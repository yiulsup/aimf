import sys
import cv2
import numpy as np
import threading
from PyQt5 import QtWidgets, QtGui, QtCore, uic


class ViewerDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("viewer.ui", self)
        self.show()

        # Connect buttons
        self.pButton_0.clicked.connect(lambda: self.open_video(0))
        self.pButton_1.clicked.connect(lambda: self.open_video(1))
        self.pButton_2.clicked.connect(lambda: self.open_video(2))

        # Capture holders
        self.caps = [None, None, None]
        self.threads = [None, None, None]
        self.running = [False, False, False]

    def open_video(self, idx):
        url_line = [self.lE_0, self.lE_1, self.lE_2][idx]
        label = [self.l_0, self.l_1, self.l_2][idx]
        url = url_line.text().strip()

        if not url:
            QtWidgets.QMessageBox.warning(self, "Warning", f"URL for camera {idx} is empty")
            return

        self.caps[idx] = cv2.VideoCapture(url)
        if not self.caps[idx].isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot open camera {idx}: {url}")
            return

        # Start thread if not running
        if not self.running[idx]:
            self.running[idx] = True
            t = threading.Thread(target=self.stream_thread, args=(idx, label), daemon=True)
            self.threads[idx] = t
            t.start()
            print(f"[INFO] Camera {idx} thread started -> {url}")

    def stream_thread(self, idx, label):
        cap = self.caps[idx]
        while self.running[idx] and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            # update QLabel safely in GUI thread
            label.setPixmap(pixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio))

        cap.release()

    def closeEvent(self, event):
        # Stop all threads
        for i in range(3):
            self.running[i] = False
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dlg = ViewerDialog()
    dlg.exec_()
