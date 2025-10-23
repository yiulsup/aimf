import os, serial, cv2, time, numpy as np, threading, binascii, queue
from flask import Flask, Response
from lne_tflite import interpreter as lt

app = Flask(__name__)

main_queue = queue.Queue()
uart = serial.Serial("/dev/ttyACM0", 921600, timeout=0.1)   # 보레이트 높이면 FPS 개선

def uart_thread():
    """UART 스레드: 열화상 프레임 수신"""
    frame = np.zeros(4800, dtype=np.uint16)

    while True:
        try:
            header = uart.read(20)
            if len(header) < 20 or header[0] != 0x02 or header[1] != 0x25 or header[2] != 0xA1:
                continue

            data = uart.read(9600)
            if len(data) < 9600:
                continue

            frame = np.frombuffer(data, dtype=np.uint16, count=4800)

            if 7 < frame.mean() < 8:
                continue

            main_queue.put(frame)

        except Exception:
            continue


class AiMF_NPU():
    def __init__(self):
        self.weight = "./LNE/Detection/yolov7.lne"
        self.interpreter = lt.Interpreter(self.weight)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.output_details.sort(key=lambda _: np.prod(_["shape"][-3:]))
        self.output_tensors = [_["index"] for _ in self.output_details]
        self.input_shape = self.input_details[0]["shape"]   # e.g. [1, 320, 320, 3] (NHWC)
        # Use dynamic H/W from the model (works for 320, 416, 640, etc.)
        # If your model is NCHW, swap indexing accordingly.
        self.height = int(self.input_shape[1])
        self.width  = int(self.input_shape[2])

        # ✅ Single class only
        self.class_names = ["kevin"]
        self.colors = np.array([[0, 255, 0]], dtype=np.uint8)   # green box

        self.latest_frame = None
        self.lock = threading.Lock()

    # -------------------------------
    # NMS functions
    # -------------------------------
    def _postprocess_iou(self, scores, classes, boxes):
        boxes_valid   = [True]*len(boxes)
        boxes_area    = [_[4]*_[5] for _ in boxes]
        score_orders  = np.argsort(scores)[::-1]

        for i, id_i in enumerate(score_orders):
            if not boxes_valid[id_i]:
                continue

            for id_j in score_orders[i+1:]:
                if (not boxes_valid[id_j]) or (classes[id_i] != classes[id_j]):
                    continue

                x0  = max(boxes[id_i][0], boxes[id_j][0])
                y0  = max(boxes[id_i][1], boxes[id_j][1])
                x1  = min(boxes[id_i][2], boxes[id_j][2])
                y1  = min(boxes[id_i][3], boxes[id_j][3])
                if (x1 <= x0 or y1 <= y0):
                    continue

                area_ixj  = (x1-x0)*(y1-y0)
                area_iuj  = boxes_area[id_i] + boxes_area[id_j] - area_ixj
                if (area_ixj >= 0.5*area_iuj):   # tightened IoU threshold
                    boxes_valid[id_j] = False

        detects = []
        for i, id_i in enumerate(score_orders):
            if i == 100:
                break
            if boxes_valid[id_i]:
                detects.append(id_i)

        return [
            np.asarray(scores )[detects],
            np.asarray(classes)[detects],
            np.asarray(boxes  )[detects]
        ]

    def postprocess_yolov7(self, predicts):
        YOLO_ANCHORS  = [
            [[116, 90], [156,198], [373,326]],    # large
            [[ 30, 61], [ 62, 45], [ 59,119]],    # medium
            [[ 10, 13], [ 16, 30], [ 33, 23]]     # small
        ]
        scores, classes, boxes  = [],[],[]

        # Use model H/W everywhere (no hardcoded 416)
        model_h = self.height
        model_w = self.width

        for p, anchors in enumerate(YOLO_ANCHORS):
            if p == 0:
                sample  = 32
                GH      = model_h // 32
                GW      = model_w // 32
                nboxes  = GH*GW*3
            else:
                sample /= 2
                GH     *= 2
                GW     *= 2
                nboxes *= 4

            for boxid in range(nboxes):
                P       = predicts[p][boxid]
                score   = P[4] * P[5]   # single class only

                if score >= 0.45:
                    a     =  boxid % 3
                    xg    = (boxid//3) % GW
                    yg    = (boxid//3) // GW
                    xc    = (xg + P[0]*2 - 0.5) * sample
                    yc    = (yg + P[1]*2 - 0.5) * sample
                    w     = (P[2]*2)**2 * anchors[a][0]
                    h     = (P[3]*2)**2 * anchors[a][1]
                    boxes.append([
                        max(xc - w/2, 0),
                        max(yc - h/2, 0),
                        min(xc + w/2, model_w),
                        min(yc + h/2, model_h),
                        w,
                        h
                    ])
                    classes.append(0)   # always class 0 ("kevin")
                    scores.append(score)

        return self._postprocess_iou(scores, classes, boxes)

    # -------------------------------
    # Drawing
    # -------------------------------
    def post_draw(self, img, classes, scores, boxes, model_input_height, model_input_width):
        img = cv2.resize(img, (500, 375))
        h, w, _ = img.shape
        h_ratio = h / model_input_height
        w_ratio = w / model_input_width

        font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        font_thick = 1

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        for i in range(len(classes)):
            label = f'kevin {scores[i]:.2f}'

            left = max(0, int(np.round(boxes[i][0] * w_ratio)))
            top = max(0, int(np.round(boxes[i][1] * h_ratio)))
            right = min(w, int(np.round(boxes[i][2] * w_ratio)))
            bottom = min(h, int(np.round(boxes[i][3] * h_ratio)))

            color = (0, 255, 0)   # green box

            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.putText(img, label, (left, top-2), font_face, font_scale, (0,0,0),
                        font_thick, cv2.LINE_AA)

        return img

    # -------------------------------
    # Main worker
    # -------------------------------
    def work(self):
        model_h = self.height
        model_w = self.width

        while True:
            frame1 = main_queue.get()
            max_val, min_val = frame1.max(), frame1.min()
            if max_val == min_val:
                continue
            frame1 = ((frame1 - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
            image = frame1.reshape(60, 80)

            grayImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # ✅ resize to model's expected size (no hardcode)
            grayImage = cv2.resize(grayImage, (model_w, model_h))

            frame_rgb = cv2.cvtColor(grayImage, cv2.COLOR_BGR2RGB).astype(np.float32)
            frame_rgb = frame_rgb / 255.0
            frame_rgb = np.expand_dims(frame_rgb, axis=0).astype(np.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], frame_rgb)
            self.interpreter.invoke()
            outputs = [self.interpreter.get_tensor(_) for _ in self.output_tensors]
            predicts = [np.reshape(_, (-1, len(self.class_names)+5)) for _ in outputs]

            # ✅ pass the real model size for proper scaling on draw
            scores, classes, boxes = self.postprocess_yolov7(predicts)
            detected_image = self.post_draw(grayImage, classes, scores, boxes, model_h, model_w)

            ret, jpeg = cv2.imencode('.jpg', detected_image)
            if ret:
                with self.lock:
                    self.latest_frame = jpeg.tobytes()


# -------------------------------
# Flask + threading
# -------------------------------
npu = AiMF_NPU()
t1 = threading.Thread(target=uart_thread, daemon=True)
t1.start()
t2 = threading.Thread(target=npu.work, daemon=True)
t2.start()

def generate():
    while True:
        with npu.lock:
            if npu.latest_frame is None:
                continue
            frame = npu.latest_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1234, threaded=True)
