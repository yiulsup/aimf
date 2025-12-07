#!/usr/bin/env python3
import os
import cv2
import sys
import time
import numpy as np
import threading
import gi

# -----------------------------
# 🔹 GStreamer RTSP Imports
# -----------------------------
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)

latest_frame = np.zeros((320, 240, 3), dtype=np.uint8)

# ============================================================
# 🔹 RTSP Server (AppSrc-based)
# ============================================================
class RTSPFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super(RTSPFactory, self).__init__()
        # ✅ 안정화된 파이프라인 (3fps, NV12, I-Frame 동기화)
        self.launch_string = (
            "appsrc name=source is-live=true block=true format=time "
            "caps=video/x-raw,format=BGR,width=320,height=240,framerate=3/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            "x264enc tune=zerolatency bitrate=2000 speed-preset=superfast key-int-max=10 ! "
            "video/x-h264,profile=baseline,level=3.0 ! "
            "rtph264pay name=pay0 pt=96 config-interval=1"
        )
        self.push_id = None
        self.set_shared(False)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.pipeline = rtsp_media.get_element()
        self.appsrc = self.pipeline.get_by_name("source")
        self.timestamp = 0
        print("✅ RTSP client connected (new appsrc active)")
        rtsp_media.connect("unprepared", self.on_unprepared)
        # ✅ 3fps (333ms per frame)
        self.push_id = GLib.timeout_add(333, self.push_frame)

    def on_unprepared(self, media):
        print("🧹 Client disconnected — stopped frame push loop")
        if self.push_id:
            GLib.source_remove(self.push_id)
            self.push_id = None
        self.appsrc = None
        self.pipeline = None
        self.timestamp = 0

    def push_frame(self):
        global latest_frame
        if self.appsrc is None:
            return True
        frame = latest_frame.copy()
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 3)
        ts = getattr(self, "timestamp", 0)
        buf.pts = buf.dts = ts
        buf.offset = ts
        self.timestamp = ts + buf.duration
        self.appsrc.emit("push-buffer", buf)
        return True


class RTSPServer(GstRtspServer.RTSPServer):
    def __init__(self):
        super(RTSPServer, self).__init__()
        factory = RTSPFactory()
        factory.set_shared(True)
        mounts = self.get_mount_points()
        mounts.add_factory("/thermal", factory)
        self.attach(None)
        print("✅ RTSP stream available at: rtsp://0.0.0.0:8554/thermal")


def start_rtsp():
    server = RTSPServer()
    loop = GLib.MainLoop()
    loop.run()

# ============================================================
# 🔹 AiMF_NPU (YOLOv7 Camera-based Inference)
# ============================================================
from lne_tflite import interpreter as lt

class AiMF_NPU:
    def __init__(self):
        # 🎥 Camera init
        self.cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 25)

        if not self.cap.isOpened():
            print("❌ Failed to open /dev/video1")
            sys.exit(1)
        print("✅ /dev/video1 opened successfully (MJPG, 640x480 @25fps)")

        # 📦 Load AiMF model
        self.weight = "./LNE/Detection/yolov7_tiny_4x8_0_1.lne"
        self.interpreter = lt.Interpreter(self.weight)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.output_details.sort(key=lambda _: np.prod(_["shape"][-3:]))
        self.output_tensors = [_["index"] for _ in self.output_details]
        self.input_shape = self.input_details[0]["shape"]
        self.height = int(self.input_shape[1])
        self.width = int(self.input_shape[2])
        print("✅ Model input:", self.input_shape)

        # 🧠 Load class names (ImageNet fallback)
        class_path = "classname.txt"
        if os.path.exists(class_path):
            with open(class_path, "r") as f:
                self.class_names = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded {len(self.class_names)} classes from {class_path}")
        else:
            print("⚠️ classname.txt not found — using dummy 1000 labels.")
            self.class_names = [f"class_{i}" for i in range(1000)]

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

    # ------------------------------------------------------------
    # 🔹 NMS
    # ------------------------------------------------------------
    def _postprocess_iou(self, scores, classes, boxes, iou_thresh=0.4):
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        boxes = np.array(boxes)
        boxes_valid = np.ones(len(boxes), dtype=bool)
        order = np.argsort(scores)[::-1]
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        for i in range(len(order)):
            if not boxes_valid[order[i]]:
                continue
            for j in range(i + 1, len(order)):
                if not boxes_valid[order[j]]:
                    continue
                if classes[order[i]] != classes[order[j]]:
                    continue
                xx1 = max(x1[order[i]], x1[order[j]])
                yy1 = max(y1[order[i]], y1[order[j]])
                xx2 = min(x2[order[i]], x2[order[j]])
                yy2 = min(y2[order[i]], y2[order[j]])
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                union = areas[order[i]] + areas[order[j]] - inter
                iou = inter / (union + 1e-6)
                if iou > iou_thresh:
                    boxes_valid[order[j]] = False
        keep = order[boxes_valid[order]]
        return np.array(scores)[keep], np.array(classes)[keep], np.array(boxes)[keep]

    # ------------------------------------------------------------
    # 🔹 YOLOv7 Postprocess
    # ------------------------------------------------------------
    def postprocess_yolov7(self, predicts):
        YOLO_ANCHORS = [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]]
        ]
        scores, classes, boxes = [], [], []
        model_w, model_h = self.width, self.height
        for p, anchors in enumerate(YOLO_ANCHORS):
            if p == 0:
                sample = 32
                GH, GW = model_h // 32, model_w // 32
                nboxes = GH * GW * 3
            else:
                sample /= 2
                GH *= 2
                GW *= 2
                nboxes *= 4
            for boxid in range(nboxes):
                P = predicts[p][boxid]
                classid = np.argmax(P[5:])
                score = P[4] * P[5 + classid]
                if score < 0.4:
                    continue
                a = boxid % 3
                xg = (boxid // 3) % GW
                yg = (boxid // 3) // GW
                xc = (xg + P[0] * 2 - 0.5) * sample
                yc = (yg + P[1] * 2 - 0.5) * sample
                w = (P[2] * 2) ** 2 * anchors[a][0]
                h = (P[3] * 2) ** 2 * anchors[a][1]
                x1, y1 = max(xc - w / 2, 0), max(yc - h / 2, 0)
                x2, y2 = min(xc + w / 2, model_w), min(yc + h / 2, model_h)
                boxes.append([x1, y1, x2, y2])
                classes.append(classid)
                scores.append(score)
        return self._postprocess_iou(scores, classes, boxes)

    # ------------------------------------------------------------
    # 🔹 Draw results
    # ------------------------------------------------------------
    def post_draw(self, img, classes, scores, boxes):
        display_w, display_h = 320, 240
        img_disp = cv2.resize(img, (display_w, display_h))
        w_ratio, h_ratio = display_w / self.width, display_h / self.height
        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]
            left, top = int(x1 * w_ratio), int(y1 * h_ratio)
            right, bottom = int(x2 * w_ratio), int(y2 * h_ratio)
            color = tuple(int(c) for c in self.colors[classes[i]])
            cv2.rectangle(img_disp, (left, top), (right, bottom), color, 2)
            label = f"{self.class_names[classes[i]]} {scores[i]:.2f}"
            cv2.putText(img_disp, label, (left, max(top - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img_disp

    # ------------------------------------------------------------
    # 🔹 Inference loop
    # ------------------------------------------------------------
    def run(self):
        global latest_frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Camera read failed")
                time.sleep(0.05)
                continue

            start = time.time()
            frame_resized = cv2.resize(frame, (self.width, self.height))
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = np.expand_dims(rgb, axis=0).astype(np.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], rgb)
            self.interpreter.invoke()
            outputs = [self.interpreter.get_tensor(i) for i in self.output_tensors]
            outputs = [self.interpreter.get_tensor(i) for i in self.output_tensors]

            output_size = outputs[0].shape[-1]
            num_classes = output_size - 5
            if len(self.class_names) != num_classes:
                print(f"⚠️ Model outputs {num_classes} classes; adjusting label list.")
                self.class_names = self.class_names[:num_classes]

            predicts = [np.reshape(o, (-1, num_classes + 5)) for o in outputs]
            
            scores, classes, boxes = self.postprocess_yolov7(predicts)
            detected = self.post_draw(frame_resized, classes, scores, boxes)

            latest_frame = detected
            print(f"✅ Inference time: {(time.time()-start)*1000:.1f} ms")
            time.sleep(0.03)

# ============================================================
# 🔹 Threads start
# ============================================================
rtsp_thread = threading.Thread(target=start_rtsp, daemon=True)
rtsp_thread.start()

npu = AiMF_NPU()
npu_thread = threading.Thread(target=npu.run, daemon=True)
npu_thread.start()

print("🔄 RTSP + AiMF_NPU Vision Inference Running... Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Gracefully exiting...")
    sys.exit(0)
