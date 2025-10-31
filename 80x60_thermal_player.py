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


# -----------------------------
# 🔹 RTSP Server (AppSrc-based)
# -----------------------------
class RTSPFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super(RTSPFactory, self).__init__()
        self.launch_string = (
            "appsrc name=source is-live=true block=true format=time "
            "caps=video/x-raw,format=BGR,width=320,height=240,framerate=0/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            "x264enc tune=zerolatency bitrate=800 speed-preset=ultrafast ! "
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
        self.push_id = GLib.timeout_add(100, self.push_frame)

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
        buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, 10)
        timestamp = getattr(self, "timestamp", 0)
        buf.pts = buf.dts = timestamp
        buf.offset = timestamp
        self.timestamp = timestamp + buf.duration
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
        print(self.input_details)
        self.output_details = self.interpreter.get_output_details()
        self.output_details.sort(key=lambda _: np.prod(_["shape"][-3:]))
        self.output_tensors = [_["index"] for _ in self.output_details]
        self.input_shape = self.input_details[0]["shape"]
        self.height = int(self.input_shape[1])
        self.width = int(self.input_shape[2])
        self.class_names = ["kevin"]

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.latest_frame = None

    # ----------------------------------------
    # 🔹 Corrected IoU-based NMS
    # ----------------------------------------
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

    # ----------------------------------------
    # 🔹 YOLOv7 Decode + NMS
    # ----------------------------------------
    def postprocess_yolov7(self, predicts):
        YOLO_ANCHORS = [
            [[116, 90], [156, 198], [373, 326]],  # large
            [[30, 61], [62, 45], [59, 119]],      # medium
            [[10, 13], [16, 30], [33, 23]]        # small
        ]

        scores, classes, boxes = [], [], []
        model_w, model_h = self.width, self.height

        for p, anchors in enumerate(YOLO_ANCHORS):
            if p == 0:
                sample = 32
                GH = model_h // 32
                GW = model_w // 32
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

                x1 = max(xc - w / 2, 0)
                y1 = max(yc - h / 2, 0)
                x2 = min(xc + w / 2, model_w)
                y2 = min(yc + h / 2, model_h)

                boxes.append([x1, y1, x2, y2])
                classes.append(classid)
                scores.append(score)

        return self._postprocess_iou(scores, classes, boxes)


    # ----------------------------------------
    # 🔹 Drawing
    # ----------------------------------------
    def post_draw(self, img, classes, scores, boxes, model_input_height, model_input_width):
        # ✅ 모델 입력 해상도 (320×320)
        model_h, model_w = model_input_height, model_input_width

        # ✅ RTSP / 표시 해상도 (320×240)
        display_w, display_h = 320, 240
        img_disp = cv2.resize(img, (display_w, display_h))

        w_ratio = display_w / model_w
        h_ratio = display_h / model_h

        font = cv2.FONT_HERSHEY_SIMPLEX
        img_disp = np.ascontiguousarray(img_disp.astype(np.uint8))

        for i in range(len(scores)):
            x1, y1, x2, y2 = boxes[i]

            left   = int(x1 * w_ratio)
            top    = int(y1 * h_ratio)
            right  = int(x2 * w_ratio)
            bottom = int(y2 * h_ratio)

            # 경계 제한
            left   = max(0, min(left, display_w - 1))
            top    = max(0, min(top, display_h - 1))
            right  = max(0, min(right, display_w - 1))
            bottom = max(0, min(bottom, display_h - 1))

            color = tuple(int(c) for c in self.colors[classes[i]])
            cv2.rectangle(img_disp, (left, top), (right, bottom), color, 2)
            label = f"{self.class_names[classes[i]]} {scores[i]:.2f}"
            cv2.putText(img_disp, label, (left, max(top - 10, 15)), font, 0.5, color, 1, cv2.LINE_AA)

        return img_disp




    # ----------------------------------------
    # 🔹 Main Inference Loop
    # ----------------------------------------
    def run(self):
        global latest_frame, main_queue
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

            scores, classes, boxes = self.postprocess_yolov7(predicts)
            detected = self.post_draw(gray, classes, scores, boxes, self.height, self.width)

            latest_frame = cv2.resize(detected, (320, 240))



# -----------------------------
# 🔹 UART Thread (ORIGINAL — UNCHANGED)
# -----------------------------
def th():
    global uart
    global main_queue
    sending_1 = [0x02, 0x00, 0x04, 0x00, 0x01, 0x55, 0xaa, 0x03, 0xFA]
    sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x5, 0x03, 0x01]
    sending_3 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x00, 0x01, 0x03, 0x06]
    sending_4 = [0x02, 0x00, 0x04, 0x02, 0x01, 0x01, 0x01, 0x03, 0x07]

    cnt = 0
    cnt1 = 0
    cnt2 = 0

    frame = np.zeros(4800)

    time.sleep(0.1)
    print("second command to fly")
    uart.write(sending_2)
    time.sleep(0.1)
    first = 1
    image_cnt = 0
    passFlag = np.zeros(6)
    start_frame = 0
    uart.write(sending_4)
    begin = 0
    check_cnt = 0

    uart.write(sending_1)
    while True:
        line = uart.read()
        cnt = cnt + 1
        if cnt >= 9:
            cnt = 0
            break
    uart.write(sending_4)

    while True:
        try:
            line = uart.read()
            cnt1 = cnt1 + 1
            if begin == 0 and cnt1 == 1:
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if rawDataDecimal == 2:
                    begin = 1
                else:
                    begin = 0
                    cnt1 = 0
                    continue
            if begin == 1 and cnt1 == 20:
                for i in range(0, 9600):
                    line = uart.read()
                    cnt1 = cnt1 + 1
                    rawDataHex = binascii.hexlify(line)
                    rawDataDecimal = int(rawDataHex, 16)
                    if first == 1:
                        dec_10 = rawDataDecimal * 256
                        first = 2
                    elif first == 2:
                        first = 1
                        dec = rawDataDecimal
                        frame[image_cnt] = dec + dec_10
                        image_cnt = image_cnt + 1

                    if image_cnt >= 4800:
                        image_cnt = 0
                        error = np.mean(frame)
                        if error > 7 and error < 8:
                            continue
                        main_queue.put(frame)

            if cnt1 == 2 and begin == 1:
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if rawDataDecimal == 0x25:
                    begin = 1
                else:
                    begin = 0
                    cnt1 = 0
                    continue
            if cnt1 == 3 and begin == 1:
                rawDataHex = binascii.hexlify(line)
                rawDataDecimal = int(rawDataHex, 16)
                if rawDataDecimal == 0xA1:
                    begin = 1
                else:
                    begin = 0
                    cnt1 = 0
                    continue

            if cnt1 == 9638 and begin == 1:
                begin = 0
                cnt1 = 0
            else:
                continue

        except:
            continue


# -----------------------------
# 🔹 Start Threads
# -----------------------------
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

print("🔄 RTSP Thermal + AiMF_NPU Server Running... Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Gracefully exiting...")
    sys.exit(0)

