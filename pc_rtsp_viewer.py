#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

RTSP_URL = "rtsp://192.168.2.2:8554/thermal"


class RTSPViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTSP Thermal Stream")
        self.setGeometry(100, 100, 640, 480)

        # QLabel for video
        self.label = QLabel(self)
        self.label.setScaledContents(True)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # OpenCV capture
        self.cap = cv2.VideoCapture(RTSP_URL)
        if not self.cap.isOpened():
            print("❌ Failed to open RTSP stream.")
            sys.exit(1)

        # Timer to fetch frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ≈ 30 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        # Optional: resize or convert color
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = RTSPViewer()
    viewer.show()
    sys.exit(app.exec_())
