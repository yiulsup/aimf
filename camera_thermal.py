import numpy as np
import threading
import time
import cv2
from flask import Flask, Response, render_template_string
from lne_tflite import interpreter as lt

# ================================
# Flask App
# ================================

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

app = Flask(__name__)

# ================================
# NPU Class
# ================================

import queue, serial, binascii

main_queue = queue.Queue()
uart = serial.Serial("/dev/ttyACM0", 115200)

def th():
    global uart
    global main_queue

    sending_1 = [0x02, 0x00, 0x04, 0x00, 0x01, 0x55, 0xaa, 0x03, 0xFA]
    sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x5, 0x03, 0x01]
    sending_3 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x00, 0x01, 0x03, 0x06]
    sending_4 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x01, 0x01, 0x03, 0x07]

    cnt = 0
    cnt1 = 0
    first = 1
    image_cnt = 0
    begin = 0

    frame = np.zeros(4800)

    time.sleep(0.1)
    uart.write(bytearray(sending_2))
    time.sleep(0.1)
    uart.write(bytearray(sending_4))
    uart.write(bytearray(sending_1))

    while True:
        try:
            line = uart.read()
            cnt1 += 1

            if begin == 0 and cnt1 == 1:
                raw = int(binascii.hexlify(line), 16)
                if raw == 2:
                    begin = 1
                else:
                    cnt1 = 0
                    continue

            if begin == 1 and cnt1 == 20:
                for i in range(9600):
                    line = uart.read()
                    raw = int(binascii.hexlify(line), 16)

                    if first == 1:
                        dec_10 = raw * 256
                        first = 2
                    else:
                        first = 1
                        frame[image_cnt] = raw + dec_10
                        image_cnt += 1

                    if image_cnt >= 4800:
                        image_cnt = 0
                        err = np.mean(frame)
                        if 7 < err < 8:
                            continue
                        main_queue.put(frame.copy())

            if cnt1 == 9638:
                begin = 0
                cnt1 = 0

        except:
            continue

def get_latest(queue):
    latest = queue.get()
    while not queue.empty():
        try:
            latest = queue.get_nowait()
        except:
            break
    return latest

class ThermalStream:
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()
        self.target_interval = 1.0 / 20   # ✅ 20 FPS 제한
        self.last_time = 0

    def work(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]  # ✅ USB+CPU 최적

        while True:
            # ✅ 항상 최신 프레임만 사용 (지연 제거 핵심)
            frame1 = get_latest(main_queue)

            max_v = np.max(frame1)
            min_v = np.min(frame1)
            if max_v == min_v:
                continue

            frame1 = (frame1 - min_v) * (255.0 / (max_v - min_v))
            image = frame1.reshape(60, 80).astype(np.uint8)

            gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gray = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_LINEAR)
            gray = cv2.flip(gray, 1)

            ret, jpeg = cv2.imencode('.jpg', gray, encode_param)
            if ret:
                with self.lock:
                    self.latest_frame = jpeg.tobytes()

            # ✅ FPS 제한 (USB Ethernet 큐 폭주 방지)
            now = time.time()
            sleep_time = self.target_interval - (now - self.last_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.last_time = time.time()




class AiMF_NPU:
    def __init__(self):
        self.cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 20)

        self.weight = "./LNE/Detection/yolov7_tiny_4x8_0_1.lne"
        self.interpreter = lt.Interpreter(self.weight)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.output_details.sort(key=lambda x: np.prod(x["shape"][-3:]))
        self.output_tensors = [x["index"] for x in self.output_details]

        self.input_shape = self.input_details[0]["shape"]

        with open("./ObjectDetection/labels/coco.names") as f:
            self.class_names = [c.strip() for c in f.readlines()]

        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255, size=(len(self.class_names), 3), dtype=np.uint8
        )

        self.raw_frame = None
        self.det_result = None
        self.frame_lock = threading.Lock()
        self.det_lock = threading.Lock()

    # ================================
    # IOU NMS
    # ================================
    def _postprocess_iou(self, scores, classes, boxes):
        pass

    def postprocess_yolov7(self, predicts):
        YOLO_ANCHORS = [
            [[116,90],[156,198],[373,326]],
            [[30,61],[62,45],[59,119]],
            [[10,13],[16,30],[33,23]]
        ]

        scores = []
        classes = []
        boxes_xywh = []   # ✅ OpenCV NMS용 (x, y, w, h)
        boxes_xyxy = []   # ✅ draw용 (x1, y1, x2, y2)

        num_cls = len(self.class_names)
        sample = 32
        GH = self.input_shape[-3] // 32
        GW = self.input_shape[-2] // 32
        nboxes = GH * GW * 3

        for p in range(len(YOLO_ANCHORS)):
            anchors = YOLO_ANCHORS[p]
            pred = predicts[p]

            for boxid in range(nboxes):
                P = pred[boxid]

                # ✅ 1차 confidence 컷
                conf = P[4]
                if conf < 0.35:
                    continue

                cls_scores = P[5:5+num_cls]
                classid = np.argmax(cls_scores)
                score = conf * cls_scores[classid]

                # ✅ 2차 최종 score 컷
                if score < 0.40:
                    continue

                a = boxid % 3
                xg = (boxid // 3) % GW
                yg = (boxid // 3) // GW

                xc = (xg + P[0] * 2.0 - 0.5) * sample
                yc = (yg + P[1] * 2.0 - 0.5) * sample
                w  = (P[2] * 2.0) ** 2 * anchors[a][0]
                h  = (P[3] * 2.0) ** 2 * anchors[a][1]

                x1 = xc - w / 2
                y1 = yc - h / 2
                x2 = xc + w / 2
                y2 = yc + h / 2

                boxes_xywh.append([x1, y1, w, h])
                boxes_xyxy.append([x1, y1, x2, y2])
                scores.append(float(score))
                classes.append(int(classid))

            sample //= 2
            GH *= 2
            GW *= 2
            nboxes *= 4

        # ✅ ✅ ✅ OpenCV C++ NMS (여기가 핵심 가속 포인트)
        if len(boxes_xywh) == 0:
            return [], [], []

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh,
            scores=scores,
            score_threshold=0.30,
            nms_threshold=0.45
        )

        if len(indices) == 0:
            return [], [], []

        final_scores = []
        final_classes = []
        final_boxes = []

        for i in indices.flatten():
            final_scores.append(scores[i])
            final_classes.append(classes[i])
            final_boxes.append(boxes_xyxy[i])

        return (
            np.array(final_scores),
            np.array(final_classes),
            np.array(final_boxes)
        )

    def post_draw(self, img, classes, scores, boxes):
        for i, c in enumerate(classes):
            color = tuple(int(x) for x in self.colors[c])
            x0, y0, x1, y1 = map(int, boxes[i][:4])

            cv2.rectangle(img, (x0,y0), (x1,y1), color, 1)
            label = f"{self.class_names[c]} {scores[i]:.2f}"
            cv2.putText(img, label, (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return img

    def capture_loop(self):
        while True:
            self.cap.grab()                  # ✅ 오래된 프레임 폐기
            ret, frame = self.cap.retrieve()
            if not ret:
                continue

            frame = cv2.resize(frame, (416,416), interpolation=cv2.INTER_NEAREST)
            with self.frame_lock:
                self.raw_frame = frame 

    def detection_loop(self):
        while True:
            with self.frame_lock:
                infer = None if self.raw_frame is None else self.raw_frame.copy()

            if infer is None:
                continue

            # ✅ 여기서 절대 self.raw_frame 다시 건드리지 않음
            frame_rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
            frame_rgb = np.expand_dims(frame_rgb, 0)

            self.interpreter.set_tensor(
                self.input_details[0]["index"], frame_rgb
            )
            self.interpreter.invoke()

            outputs = [self.interpreter.get_tensor(i) for i in self.output_tensors]
            predicts = [np.reshape(o, (-1, len(self.class_names)+5)) for o in outputs]

            scores, classes, boxes = self.postprocess_yolov7(predicts)

            with self.det_lock:
                self.det_result = (scores, classes, boxes) if len(classes) else None
            time.sleep(0.08)


# ================================
# Flask Streaming
# ================================
def generate():
    while True:
        with npu.frame_lock:
            frame = None if npu.raw_frame is None else npu.raw_frame.copy()

        if frame is None:
            time.sleep(0.002)   # ✅ lock 밖에서 sleep
            continue

        with npu.det_lock:
            det = npu.det_result

        if det is not None:
            scores, classes, boxes = det
            frame = npu.post_draw(frame, classes, scores, boxes)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        ret, jpeg = cv2.imencode(".jpg", frame, encode_param)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

        time.sleep(1/20)

thermal = ThermalStream()

def generate_thermal():
    while True:
        with thermal.lock:
            if thermal.latest_frame is None:
                time.sleep(0.01)
                continue
            jpeg_bytes = thermal.latest_frame  # already compressed ✅

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg_bytes + b'\r\n')

@app.route("/thermal")
def thermal_feed():
    return Response(generate_thermal(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# ================================
# HTML Page
# ================================
@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>NPU Dashboard</title>
<style>
body { margin:0; background:black; color:white; font-family:Arial; }
.outer { width:95vw; height:95vh; border:6px solid white; margin:2vh auto; padding:40px; box-sizing:border-box; }
.main { display:flex; gap:60px; }
.left { flex:1; align-items:center; }

.camera-box {
    width: 1280px;
    height: 800px;
    background: #7fa6d1;
    display: flex;
    align-items: center;
    justify-content: center;
}
.camera-box img {
    width: 1280px;
    height: 800px;
    object-fit: fill;
    image-rendering: pixelated;
}

.right { width:520px; }

.board-box {
    width: 260px;
    height: 230px;
    background: #7fa6d1;
    margin-top: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    overflow: hidden;
}
.board-box img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* ✅ ✅ ✅ Thermal Box 추가 */
.thermal-box {
    width: 260px;
    height: 220px;
    background: #3a3a3a;
    margin-top: 20px;
    padding: 6px;
    border-radius: 4px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.thermal-title {
    font-size: 14px;
    margin-bottom: 6px;
}

.thermal-box img {
    width: 240px;
    height: 180px;
    object-fit: fill;
    image-rendering: pixelated;
    border: 2px solid white;
}

/* 기존 로고 */
.outer {
    position: relative;
}

.emtake-logo {
    position: absolute;
    right: 60px;
    bottom: 40px;
    width: 120px;
    opacity: 0.9;
    z-index: 100;
    pointer-events: none;
}
</style>
</head>

<body>
<div class="outer">
 <div class="main">

  <div class="left">
   <div class="camera-box">
     <img src="/video">
   </div>
  </div>

  <div class="right">
   <h2>Information</h2>
   <p>SoC : LG8111</p>
   <p>NPU : 1TOPS from AiM Future</p>
   <p>Model : Yolov7-tiny</p>
   <p>Dataset : ImageNet</p>

   <div class="board-box">
    <img src="/static/eagleboard.png">
   </div>

   <!-- ✅ ✅ ✅ 여기 Thermal 추가됨 -->
   <div class="thermal-box">
     <div class="thermal-title">Thermal 80x60</div>
     <img src="/thermal">
   </div>

  </div>
 </div>

 <img src="/static/emtake.png" class="emtake-logo">
</div>
</body>
</html>
""")



@app.route("/video")
def video_feed():
    return Response(generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    npu = AiMF_NPU()

    t1 = threading.Thread(target=npu.capture_loop, daemon=True)
    t2 = threading.Thread(target=npu.detection_loop, daemon=True)

    t3 = threading.Thread(target=th, daemon=True)          # ✅ 새 UART
    t4 = threading.Thread(target=thermal.work, daemon=True)  # ✅ Thermal 영상

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    app.run(host="0.0.0.0", port=1234, threaded=True)