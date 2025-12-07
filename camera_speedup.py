import numpy as np
import threading
import time
import cv2
from flask import Flask, Response
from lne_tflite import interpreter as lt

# ================================
# Flask App
# ================================
app = Flask(__name__)

# ================================
# NPU Class
# ================================
class AiMF_NPU:
    def __init__(self):
        self.cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)

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
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

        # Shared buffers
        self.raw_frame = None
        self.det_result = None   # (scores, classes, boxes)
        self.frame_lock = threading.Lock()
        self.det_lock   = threading.Lock()

    # ================================
    # NMS / Postprocess
    # ================================
    def _postprocess_iou(self, scores, classes, boxes):
        valid = [True]*len(boxes)
        areas = [b[4]*b[5] for b in boxes]
        order = np.argsort(scores)[::-1]

        for i, id_i in enumerate(order):
            if not valid[id_i]:
                continue
            for id_j in order[i+1:]:
                if not valid[id_j] or classes[id_i] != classes[id_j]:
                    continue
                x0 = max(boxes[id_i][0], boxes[id_j][0])
                y0 = max(boxes[id_i][1], boxes[id_j][1])
                x1 = min(boxes[id_i][2], boxes[id_j][2])
                y1 = min(boxes[id_i][3], boxes[id_j][3])
                if x1 <= x0 or y1 <= y0:
                    continue
                inter = (x1-x0)*(y1-y0)
                union = areas[id_i] + areas[id_j] - inter
                if inter >= 0.3 * union:
                    valid[id_j] = False

        keep = [i for i in order if valid[i]]
        return np.array(scores)[keep], np.array(classes)[keep], np.array(boxes)[keep]

    def postprocess_yolov7(self, predicts):
        YOLO_ANCHORS = [
            [[116,90],[156,198],[373,326]],
            [[30,61],[62,45],[59,119]],
            [[10,13],[16,30],[33,23]]
        ]

        scores, classes, boxes = [], [], []
        sample = 32
        GH = self.input_shape[-3]//32
        GW = self.input_shape[-2]//32
        nboxes = GH*GW*3

        for p, anchors in enumerate(YOLO_ANCHORS):
            for boxid in range(nboxes):
                P = predicts[p][boxid]
                classid = np.argmax(P[5:])
                score = P[4] * P[5+classid]
                if score >= 0.3:
                    a = boxid % 3
                    xg = (boxid//3) % GW
                    yg = (boxid//3) // GW
                    xc = (xg + P[0]*2 - 0.5) * sample
                    yc = (yg + P[1]*2 - 0.5) * sample
                    w  = (P[2]*2)**2 * anchors[a][0]
                    h  = (P[3]*2)**2 * anchors[a][1]
                    boxes.append([xc-w/2, yc-h/2, xc+w/2, yc+h/2, w, h])
                    classes.append(classid)
                    scores.append(score)

            sample //= 2
            GH *= 2
            GW *= 2
            nboxes *= 4

        return self._postprocess_iou(scores, classes, boxes)

    def post_draw(self, img, classes, scores, boxes):
        h, w, _ = img.shape
        for i, c in enumerate(classes):
            color = tuple(int(x) for x in self.colors[c])
            x0, y0, x1, y1 = map(int, boxes[i][:4])
            x0 = max(0, min(w, x0))
            y0 = max(0, min(h, y0))
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))

            cv2.rectangle(img, (x0,y0), (x1,y1), color, 2)
            label = f"{self.class_names[c]} {scores[i]:.2f}"
            cv2.putText(img, label, (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    # ================================
    # Thread 1: Camera Capture
    # ================================
    def capture_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (416,416))

            with self.frame_lock:
                self.raw_frame = frame.copy()

    # ================================
    # Thread 2: NPU Detection
    # ================================
    def detection_loop(self):
        while True:
            # ✅ ONLY lock raw frame briefly
            with self.frame_lock:
                if self.raw_frame is None:
                    continue
                infer = self.raw_frame.copy()

            # ❗ NO LOCK during heavy compute
            frame_rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
            frame_rgb = frame_rgb.astype(np.float32) / 255.0
            frame_rgb = np.expand_dims(frame_rgb, 0)

            self.interpreter.set_tensor(
                self.input_details[0]['index'], frame_rgb
            )
            self.interpreter.invoke()

            outputs = [
                self.interpreter.get_tensor(i)
                for i in self.output_tensors
            ]

            predicts = [
                np.reshape(o, (-1, len(self.class_names)+5))
                for o in outputs
            ]

            scores, classes, boxes = self.postprocess_yolov7(predicts)

            # ✅ ONLY lock detection result briefly
            with self.det_lock:
                self.det_result = (
                    (scores, classes, boxes)
                    if len(classes) > 0 else None
                )



# ================================
# Flask Streaming
# ================================
def generate():
    while True:
        # ✅ Only lock frame for copy
        with npu.frame_lock:
            if npu.raw_frame is None:
                continue
            frame = npu.raw_frame.copy()

        # ✅ Only lock detection result for read
        with npu.det_lock:
            det = npu.det_result

        if det is not None:
            scores, classes, boxes = det
            frame = npu.post_draw(frame, classes, scores, boxes)

        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

from flask import Flask, Response, render_template_string

@app.route("/")
def index():
    return render_template_string("""
@app.route("/")
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NPU Camera Dashboard</title>
<style>
    body {
        margin: 0;
        background: white;
        font-family: Arial, Helvetica, sans-serif;
    }

    /* ✅ 전체 외곽 Red Border */
    .outer {
        width: 95vw;
        height: 95vh;
        margin: 2vh auto;
        border: 6px solid red;
        box-sizing: border-box;
        padding: 40px;
    }

    /* ✅ 전체 레이아웃 */
    .main {
        display: flex;
        flex-direction: row;
        height: 100%;
        gap: 60px;
    }

    /* ✅ 왼쪽 카메라 영역 */
    .left {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .camera-box {
        width: 100%;
        height: 480px;
        background: #7fa6d1;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        color: black;
        font-weight: 500;
    }

    .camera-box img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 4px;
    }

    .camera-text {
        margin-top: 12px;
        font-size: 18px;
    }

    /* ✅ 오른쪽 정보 영역 */
    .right {
        width: 520px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }

    .info {
        margin-bottom: 50px;
    }

    .info h2 {
        margin-bottom: 18px;
    }

    .info p {
        margin: 6px 0;
        font-size: 17px;
    }

    /* ✅ 보드 이미지 영역 */
    .board-box {
        width: 260px;
        height: 230px;
        background: #7fa6d1;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 4px;
        font-size: 20px;
    }

</style>
</head>

<body>
<div class="outer">
    <div class="main">

        <!-- ✅ LEFT: Camera -->
        <div class="left">
            <div class="camera-box">
                <img src="/video" />
            </div>
            <div class="camera-text">EagleBoard from Emttake</div>
        </div>

        <!-- ✅ RIGHT: Info -->
        <div class="right">
            <div class="info">
                <h2>Information</h2>
                <p>SoC : LG8111</p>
                <p>NPU : 1TOPS from AiM Future</p>
                <p>Model : Yolov7-tiny</p>
                <p>Dataset : ImageNet</p>
            </div>

            <div class="board-box">
                Eagleboard.png
            </div>
        </div>

    </div>
</div>
</body>
</html>

""")

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
