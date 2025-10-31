#!/usr/bin/env python3
import os
import serial
import cv2
import sys
import time
import numpy as np
import threading
import signal
import binascii
import queue
import gi

# -----------------------------
# 🔹 GStreamer RTSP Imports
# -----------------------------
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib, GObject

Gst.init(None)

# -----------------------------
# Global Variables
# -----------------------------
main_queue = queue.Queue()
uart = serial.Serial("/dev/ttyACM0", 115200)
latest_frame = np.zeros((320, 240, 3), dtype=np.uint8)
lock = threading.Lock()

# -----------------------------
# 🔹 Shared RTSP Factory (multi-client safe)
# -----------------------------
class SharedFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super().__init__()
        self.launch_string = (
            "appsrc name=source is-live=true block=true format=time "
            "caps=video/x-raw,format=BGR,width=320,height=240,framerate=10/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            "x264enc tune=zerolatency bitrate=800 speed-preset=ultrafast ! "
            "rtph264pay name=pay0 pt=96 config-interval=1"
        )
        self.appsrc = None
        self.timestamp = 0
        self.push_id = None
        self.set_shared(True)  # ✅ shared between clients

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.pipeline = rtsp_media.get_element()
        self.appsrc = self.pipeline.get_by_name("source")
        print("✅ Client connected to /thermal")
        rtsp_media.connect("unprepared", self.on_unprepared)

        # Start push loop only once
        if self.push_id is None:
            self.push_id = GLib.timeout_add(100, self.push_frame)

    def on_unprepared(self, media):
        print("🧹 A client disconnected from /thermal (stream still running)")

    def push_frame(self):
        global latest_frame, lock
        if self.appsrc is None:
            return True

        with lock:
            frame = latest_frame.copy()

        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 10)
        buf.pts = buf.dts = self.timestamp
        buf.offset = self.timestamp
        self.timestamp += buf.duration

        try:
            self.appsrc.emit("push-buffer", buf)
        except Exception:
            pass  # ignore errors if client disconnects mid-push
        return True


class RTSPServer(GstRtspServer.RTSPServer):
    def __init__(self):
        super().__init__()
        mounts = self.get_mount_points()
        factory = SharedFactory()
        mounts.add_factory("/thermal", factory)
        self.attach(None)
        print("✅ RTSP stream available at: rtsp://0.0.0.0:8554/thermal")


def start_rtsp():
    server = RTSPServer()
    loop = GLib.MainLoop()
    loop.run()


# -----------------------------
# 🔹 AiMF_NPU (YOLOv7 inference)
# -----------------------------
from lne_tflite import interpreter as lt

class AiMF_NPU:
    def __init__(self):
        self.weight = "./LNE/Detection/yolov7_body.lne"
        self.interpreter = lt.Interpreter(self.weight)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.output_details.sort(key=lambda _: np.prod(_["shape"][-3:]))
        self.output_tensors = [_["index"] for _ in self.output_details]
        self.input_shape = self.input_details[0]["shape"]
        self.height = int(self.input_shape[1])
        self.width = int(self.input_shape[2])
        self.class_names = ["kevin"]

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

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
                inter_w = max(0, xx2 - xx1)
                inter_h = max(0, yy2 - yy1)
                inter = inter_w * inter_h
                union = areas[order[i]] + areas[order[j]] - inter
                iou = inter / (union + 1e-6)
                if iou > iou_thresh:
                    boxes_valid[order[j]] = False
        keep = order[boxes_valid[order]]
        return np.array(scores)[keep], np.array(classes)[keep], np.array(boxes)[keep]

    def post_draw(self, img, classes, scores, boxes, model_input_height, model_input_width):
        display_w, display_h = 320, 240
        img_disp = cv2.resize(img, (display_w, display_h))
        w_ratio = display_w / model_input_width
        h_ratio = display_h / model_input_height
        font = cv2.FONT_HERSHEY_SIMPLEX
        img_disp = np.ascontiguousarray(img_disp.astype(np.uint8))
        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]
            left   = max(0, min(int(x1 * w_ratio), display_w - 1))
            top    = max(0, min(int(y1 * h_ratio), display_h - 1))
            right  = max(0, min(int(x2 * w_ratio), display_w - 1))
            bottom = max(0, min(int(y2 * h_ratio), display_h - 1))
            color = tuple(int(c) for c in self.colors[classes[i]])
            cv2.rectangle(img_disp, (left, top), (right, bottom), color, 2)
            label = f"{self.class_names[classes[i]]} {scores[i]:.2f}"
            cv2.putText(img_disp, label, (left, max(top - 10, 15)), font, 0.5, color, 1, cv2.LINE_AA)
        return img_disp

    def run(self):
        global latest_frame, main_queue, lock
        model_h, model_w = self.height, self.width
        while True:
            try:
                frame1 = main_queue.get(timeout=1)
            except queue.Empty:
                continue
            maxv, minv = np.max(frame1), np.min(frame1)
            if maxv <= minv:
                continue
            frame1 = ((frame1 - minv) * (255.0 / (maxv - minv))).astype(np.uint8)
            gray = cv2.cvtColor(frame1.reshape(60, 80), cv2.COLOR_GRAY2BGR)
            gray = cv2.resize(gray, (model_w, model_h))
            rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = np.expand_dims(rgb, axis=0).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], rgb)
            self.interpreter.invoke()
            outputs = [self.interpreter.get_tensor(i) for i in self.output_tensors]
            predicts = [np.reshape(o, (-1, len(self.class_names) + 5)) for o in outputs]
            scores, classes, boxes = self._postprocess_iou(
                [p[4] for p in predicts[0]],
                [np.argmax(p[5:]) for p in predicts[0]],
                [p[:4] for p in predicts[0]]
            )
            detected = self.post_draw(gray, classes, scores, boxes, self.height, self.width)
            with lock:
                latest_frame = cv2.resize(detected, (320, 240))


# -----------------------------
# 🔹 UART Thread (unchanged)
# -----------------------------
def th():
    global uart, main_queue
    sending_1 = [0x02, 0x00, 0x04, 0x00, 0x01, 0x55, 0xaa, 0x03, 0xFA]
    sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x5, 0x03, 0x01]
    sending_4 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x01, 0x01, 0x03, 0x07]
    frame = np.zeros(4800)
    uart.write(sending_2)
    time.sleep(0.1)
    uart.write(sending_4)
    print("UART ready")
    first = 1
    begin = 0
    cnt1 = 0
    image_cnt = 0
    while True:
        try:
            line = uart.read()
            cnt1 += 1
            if begin == 0 and cnt1 == 1:
                if int(binascii.hexlify(line), 16) == 2:
                    begin = 1
                else:
                    cnt1 = 0
                    continue
            if begin == 1 and cnt1 == 20:
                for _ in range(9600):
                    raw = int(binascii.hexlify(uart.read()), 16)
                    if first == 1:
                        dec_10 = raw * 256
                        first = 2
                    elif first == 2:
                        first = 1
                        dec = raw
                        frame[image_cnt] = dec + dec_10
                        image_cnt += 1
                    if image_cnt >= 4800:
                        image_cnt = 0
                        main_queue.put(frame)
        except:
            continue


# -----------------------------
# 🔹 Start Threads
# -----------------------------
uart_thread = threading.Thread(target=th, daemon=True)
uart_thread.start()

rtsp_thread = threading.Thread(target=start_rtsp, daemon=True)
rtsp_thread.start()

npu = AiMF_NPU()
npu_thread = threading.Thread(target=npu.run, daemon=True)
npu_thread.start()

print("🔄 Shared RTSP Thermal + AiMF_NPU Server Running... Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Gracefully exiting...")
    sys.exit(0)
