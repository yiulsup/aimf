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
body { margin:0; background:white; font-family:Arial; }
.outer { width:95vw; height:95vh; border:6px solid red; margin:2vh auto; padding:40px; box-sizing:border-box; }
.main { display:flex; gap:60px; }
.left { flex:1; align-items:center; }
.camera-box {
    width: 1280px;      /* ✅ 화면 표시 크기 */
    height: 800px;
    background: #7fa6d1;
    display: flex;
    align-items: center;
    justify-content: center;
}

.camera-box img {
    width: 1280px;      /* ✅ 브라우저에서만 확대 */
    height: 800px;
    object-fit: fill; /* ✅ 1:1 비율 유지 */
    image-rendering: pixelated;  /* ✅ 보간 흐림 최소화 */
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
    overflow: hidden;   /* ✅ 이미지가 박스 넘치지 않게 */
}

.board-box img {
    width: 100%;
    height: 100%;
    object-fit: contain;   /* ✅ 비율 유지하면서 박스에 맞춤 */
}
                
.outer {
    width: 95vw;
    height: 95vh;
    margin: 2vh auto;
    border: 6px solid red;
    box-sizing: border-box;
    padding: 40px;
    position: relative;   /* ✅ 이거 반드시 필요 */
}
                                  

.emtake-logo {
    position: absolute;      /* ✅ fixed → absolute (outer 기준) */
    right: 60px;
    bottom: 40px;
    width: 120px;
    opacity: 0.9;
    z-index: 100;
    pointer-events: none;
}

.outer {
    position: relative;     /* ✅ 로고 위치 기준점 */
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
  </div>
 </div>

 <!-- ✅ 빨간 보더 안쪽 우하단 로고 -->
 <img src="/static/emtake.png" class="emtake-logo">

</div>
</body>
</html>

""")


@app.route("/video")
def video_feed():
    return Response(generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame")


# ================================
# Main
# ================================
if __name__ == "__main__":
    npu = AiMF_NPU()

    t1 = threading.Thread(target=npu.capture_loop, daemon=True)
    t2 = threading.Thread(target=npu.detection_loop, daemon=True)

    t1.start()
    t2.start()

    app.run(host="0.0.0.0", port=1234, threaded=True)
