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
# 🔹 GStreamer / RTSP Imports
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
latest_frame = np.zeros((320, 240, 3), dtype=np.uint8)  # shared RTSP frame buffer


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
        self.set_shared(False)  # ✅ ensures new pipeline per client

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.pipeline = rtsp_media.get_element()
        self.appsrc = self.pipeline.get_by_name("source")
        print("✅ RTSP client connected (new appsrc active)")

        # Clean up when client disconnects
        rtsp_media.connect("unprepared", self.on_unprepared)

        # Start pushing frames
        self.push_id = GLib.timeout_add(100, self.push_frame)

    def on_unprepared(self, media):
        """Stop pushing frames when client disconnects"""
        if self.push_id:
            GLib.source_remove(self.push_id)
            self.push_id = None
        self.appsrc = None
        print("🧹 Client disconnected — stopped frame push loop")

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
        mount_points = self.get_mount_points()
        mount_points.add_factory("/thermal", factory)
        self.attach(None)
        print("✅ RTSP stream available at: rtsp://0.0.0.0:8554/thermal")


def start_rtsp():
    server = RTSPServer()
    loop = GLib.MainLoop()
    loop.run()


# -----------------------------
# 🔹 UART Thread (unchanged)
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
# 🔹 Start UART + RTSP threads
# -----------------------------
t = threading.Thread(target=th)
t.start()

rtsp_thread = threading.Thread(target=start_rtsp, daemon=True)
rtsp_thread.start()


# -----------------------------
# 🔹 Your existing OpenCV loop
# -----------------------------
while True:
    frame1 = main_queue.get()
    max = np.max(frame1)
    min = np.min(frame1)

    nfactor = 255 / (max - min)
    pTemp = frame1 - min
    nTemp = pTemp * nfactor
    frame1 = nTemp
    image = frame1.reshape(60, 80)
    uint_img = np.array(image).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    grayImage = cv2.resize(grayImage, (320, 240))
    grayImage = cv2.flip(grayImage, 1)

    # ✅ Added line: update RTSP frame
    latest_frame = grayImage.copy()

    # Local display (unchanged)
    #cv2.imshow('d', grayImage)
    cv2.waitKey(1)
