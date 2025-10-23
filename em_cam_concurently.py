#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2, time, threading, queue, numpy as np, signal, sys, os
import redis
from flask import Flask, Response
from lne_tflite import interpreter as lt
from yolo_util import postprocess_yolov7   # from your SDK utils

# =========================
# Config
# =========================
DEVICE = "/dev/video2"        # <- change to "/dev/video1" if needed
MODEL = "./LNE/Detection/yolov7_tiny_4x8_0_1.lne"
LABELS = "./ObjectDetection/labels/coco.names"
HOST = "192.168.0.120"
PORT = 1234
IN_W, IN_H = 416, 416         # model input
CAM_W, CAM_H = 640, 480       # camera capture size
CAM_FPS = 25

# =========================
# Helper: draw detections
# =========================
def draw_detections(image, classes, scores, boxes, class_names=None, colors=None):
    img = cv2.resize(image, (500, 375))
    h, w, _ = img.shape
    # scale from 416x416 coords (postprocess is on model space)
    h_ratio = h / IN_H
    w_ratio = w / IN_W

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    img = np.ascontiguousarray(img)

    for i, c in enumerate(classes):
        label = f"{int(c)}:{scores[i]:.2f}"
        if class_names and int(c) < len(class_names):
            label = f"{class_names[int(c)]}:{scores[i]:.2f}"

        # boxes are [x1,y1,x2,y2] in model space
        left   = max(0, int(np.round(boxes[i][0] * w_ratio)))
        top    = max(0, int(np.round(boxes[i][1] * h_ratio)))
        right  = min(w, int(np.round(boxes[i][2] * w_ratio)))
        bottom = min(h, int(np.round(boxes[i][3] * h_ratio)))

        color = (0, 255, 0)
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        cv2.putText(img, label, (left, max(10, top - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None

# =========================
# Camera thread
# =========================
class CameraStreamer(threading.Thread):
    def __init__(self, device=DEVICE, width=CAM_W, height=CAM_H, fps=CAM_FPS):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        # Try MJPG first
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # If not opened, try YUYV fallback
        if not self.cap.isOpened():
            print(f"[Camera] Failed MJPG open on {device}, trying default backend")
            self.cap = cv2.VideoCapture(device)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.queue = queue.Queue(maxsize=5)
        self.running = True
        self.latest = None

        print(f"[Camera] {device} opened={self.cap.isOpened()} "
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
              f"{self.cap.get(cv2.CAP_PROP_FPS):.1f} FPS")

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            # keep a latest copy (for non-blocking consumers)
            self.latest = frame

            # push resized frame for model
            if not self.queue.full():
                img = cv2.resize(frame, (IN_W, IN_H))
                self.queue.put(img)
            else:
                # drop if queue full (we want low latency, not backlog)
                pass

    def read(self, timeout=0.05):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest(self):
        return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

# =========================
# YOLO Worker (SDK style)
# =========================
class TinyYoloV7Worker(threading.Thread):
    def __init__(self, cam: CameraStreamer, redis_client=None):
        super().__init__(daemon=True)
        self.cam = cam
        self.redis = redis_client
        self.resultQ = queue.Queue(maxsize=5)
        self.running = True
        self.latest_jpeg = None

        # Load interpreter
        self.interpreter = lt.Interpreter(model_path=MODEL)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()
        self.output_detail = self.interpreter.get_output_details()
        self.output_detail.sort(key=lambda _: np.prod(_["shape"][-3:]))

        self.input_index = self.input_detail[0]["index"]
        self.output_indexes = [o["index"] for o in self.output_detail]

        # <-- postprocess_yolov7 expects these
        self.input_shape = self.input_detail[0]["shape"]  # [1,416,416,3]
        self.batch, self.h, self.w, self.c = self.input_shape
        print(f"[Worker] input shape: {self.input_shape}")

        with open(LABELS, "r") as f:
            self.class_names = [c.strip() for c in f.readlines()]

    def preprocess(self, frame):
        # frame already 416x416 BGR
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)  # [1,h,w,c]

    def run(self):
        while self.running:
            frame = self.cam.read()
            if frame is None:
                time.sleep(0.001)
                continue

            inp = self.preprocess(frame)

            t0 = time.time()
            self.interpreter.set_tensor(self.input_index, inp)
            self.interpreter.invoke()
            t1 = time.time()

            # Gather raw outputs -> reshape -> SDK postprocess
            outputs = [self.interpreter.get_tensor(i) for i in self.output_indexes]
            nclasses = len(self.class_names)
            predicts = [np.reshape(_, (-1, nclasses + 5)) for _ in outputs]

            try:
                scores, classes, boxes = postprocess_yolov7(self, predicts)
            except Exception as e:
                print(f"[WARN] postprocess error: {e}")
                continue

            if boxes is not None and len(boxes) > 0:
                jpeg = draw_detections(frame, classes, scores, boxes, self.class_names)
            else:
                # still output the raw frame if you like
                ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (500, 375)))
                jpeg = buf.tobytes() if ok else None

            if jpeg:
                self.latest_jpeg = jpeg
                if not self.resultQ.full():
                    self.resultQ.put(jpeg)

            fps = 1.0 / max(1e-6, (t1 - t0))
            if self.redis:
                self.redis.set("yolov7_tiny_fps", round(fps, 2))

    def get_result(self):
        try:
            return self.resultQ.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False

# =========================
# Flask server
# =========================
def create_app(worker: TinyYoloV7Worker, cam: CameraStreamer):
    app = Flask(__name__)

    @app.route("/")
    def stream():
        def gen():
            while True:
                frame = worker.get_result()
                if frame:
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    time.sleep(0.01)  # wait for next valid frame
                if not frame:
                    # fallback to last detection frame
                    frame = worker.latest_jpeg
                if not frame:
                    # fallback to latest raw camera frame (blank before first frame)
                    raw = cam.get_latest()
                    if raw is None:
                        blank = np.zeros((375, 500, 3), np.uint8)
                        frame = cv2.imencode(".jpg", blank)[1].tobytes()
                    else:
                        frame = cv2.imencode(".jpg", cv2.resize(raw, (500, 375)))[1].tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.005)  # reduce busy loop
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/health")
    def health():
        return "ok", 200

    return app

# =========================
# Main
# =========================
if __name__ == "__main__":
    r = redis.Redis(host="localhost", port=6379, db=0)

    cam = CameraStreamer(DEVICE, CAM_W, CAM_H, CAM_FPS)
    cam.start()

    worker = TinyYoloV7Worker(cam, redis_client=r)
    worker.start()

    app = create_app(worker, cam)

    def handler(sig, frame):
        print("Stopping...")
        worker.stop()
        cam.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print(f"➡️  Open http://{HOST}:{PORT}/")
    app.run(host=HOST, port=PORT, threaded=True)
