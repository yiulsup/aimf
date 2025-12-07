import numpy as np
import threading
import time
import cv2
import queue
from flask import Flask, Response, render_template_string
from lne_tflite import interpreter as lt

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

app = Flask(__name__)

# ==========================================
# NPU PIPELINE CLASS
# ==========================================
class AiMF_Pipeline:
    def __init__(self):
        # ---------------- Capture ----------------
        self.cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 20)

        # ---------------- NPU ----------------
        self.weight = "./LNE/Detection/yolov7_tiny_4x8_0_1.lne"
        self.interpreter = lt.Interpreter(self.weight)
        self.interpreter.allocate_tensors()

        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.output_details.sort(key=lambda x: np.prod(x["shape"][-3:]))
        self.output_tensors = [x["index"] for x in self.output_details]
        self.input_shape = self.input_details[0]["shape"]

        # ---------------- Labels ----------------
        with open("./ObjectDetection/labels/coco.names") as f:
            self.class_names = [c.strip() for c in f.readlines()]

        self.colors = np.random.randint(0,255,(len(self.class_names),3),dtype=np.uint8)

        # ---------------- Queues ----------------
        self.q_capture = queue.Queue(maxsize=1)
        self.q_npu     = queue.Queue(maxsize=1)

        # ---------------- Result ----------------
        self.latest_frame = None
        self.latest_det   = None
        self.result_lock  = threading.Lock()

    # ==========================================
    # YOLO POST (그대로 유지)
    # ==========================================
    def postprocess_yolov7(self, predicts):
        YOLO_ANCHORS = [
            [[116,90],[156,198],[373,326]],
            [[30,61],[62,45],[59,119]],
            [[10,13],[16,30],[33,23]]
        ]

        scores, classes = [], []
        boxes_xywh, boxes_xyxy = [], []

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
                conf = P[4]
                if conf < 0.35: continue

                cls_scores = P[5:5+num_cls]
                classid = np.argmax(cls_scores)
                score = conf * cls_scores[classid]
                if score < 0.40: continue

                a  = boxid % 3
                xg = (boxid // 3) % GW
                yg = (boxid // 3) // GW

                xc = (xg + P[0] * 2.0 - 0.5) * sample
                yc = (yg + P[1] * 2.0 - 0.5) * sample
                w  = (P[2] * 2.0) ** 2 * anchors[a][0]
                h  = (P[3] * 2.0) ** 2 * anchors[a][1]

                x1,y1 = xc-w/2, yc-h/2
                x2,y2 = xc+w/2, yc+h/2

                boxes_xywh.append([x1,y1,w,h])
                boxes_xyxy.append([x1,y1,x2,y2])
                scores.append(float(score))
                classes.append(int(classid))

            sample //= 2
            GH *= 2; GW *= 2; nboxes *= 4

        if len(boxes_xywh) == 0:
            return [],[],[]

        idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.30, 0.45)
        if len(idx) == 0:
            return [],[],[]

        final_scores, final_classes, final_boxes = [],[],[]
        for i in idx.flatten():
            final_scores.append(scores[i])
            final_classes.append(classes[i])
            final_boxes.append(boxes_xyxy[i])

        return np.array(final_scores), np.array(final_classes), np.array(final_boxes)

    def draw(self, img, classes, scores, boxes):
        for i,c in enumerate(classes):
            x0,y0,x1,y1 = map(int, boxes[i])
            color = tuple(int(v) for v in self.colors[c])
            cv2.rectangle(img,(x0,y0),(x1,y1),color,1)
            cv2.putText(img,f"{self.class_names[c]} {scores[i]:.2f}",
                        (x0,y0-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
        return img

    # ==========================================
    # Thread 1: Capture
    # ==========================================
    def capture_loop(self):
        while True:
            self.cap.grab()
            ret, frame = self.cap.retrieve()
            if not ret: continue

            frame = cv2.resize(frame,(416,416),cv2.INTER_NEAREST)
            if not self.q_capture.full():
                self.q_capture.put(frame)

    # ==========================================
    # Thread 2: NPU
    # ==========================================
    def npu_loop(self):
        while True:
            frame = self.q_capture.get()

            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            rgb = np.expand_dims(rgb,0)

            self.interpreter.set_tensor(self.input_details[0]["index"], rgb)
            self.interpreter.invoke()

            outputs = [self.interpreter.get_tensor(i) for i in self.output_tensors]
            predicts = [np.reshape(o,(-1,len(self.class_names)+5)) for o in outputs]

            if not self.q_npu.full():
                self.q_npu.put((frame, predicts))

    # ==========================================
    # Thread 3: CPU POST
    # ==========================================
    def cpu_post_loop(self):
        while True:
            frame, predicts = self.q_npu.get()

            scores, classes, boxes = self.postprocess_yolov7(predicts)

            with self.result_lock:
                self.latest_frame = frame
                self.latest_det   = (scores,classes,boxes)

    # ==========================================
    # Flask Generator
    # ==========================================
    def flask_generator(self):
        while True:
            with self.result_lock:
                if self.latest_frame is None:
                    time.sleep(0.005)
                    continue

                frame = self.latest_frame.copy()
                det   = self.latest_det

            if det is not None:
                scores, classes, boxes = det
                frame = self.draw(frame, classes, scores, boxes)

            _, jpeg = cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),50])

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpeg.tobytes() + b"\r\n")
            time.sleep(1/20)

pipeline = AiMF_Pipeline()


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
    border: 6px solid white;
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
def video():
    return Response(pipeline.flask_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=pipeline.capture_loop, daemon=True).start()
    threading.Thread(target=pipeline.npu_loop, daemon=True).start()
    threading.Thread(target=pipeline.cpu_post_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=1234, threaded=True)
