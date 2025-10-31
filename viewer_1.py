import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore, uic


class VideoThread(QtCore.QThread):
    frame_received = QtCore.pyqtSignal(int, np.ndarray)

    def __init__(self, idx, url):
        super().__init__()
        self.idx = idx
        self.url = url
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffering delay

        if not cap.isOpened():
            print(f"[ERROR] Failed to open stream {self.idx}: {self.url}")
            return

        print(f"[INFO] Started streaming {self.idx}: {self.url}")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Reconnect on failure
                print(f"[WARN] Stream {self.idx} lost, reconnecting...")
                cap.release()
                QtCore.QThread.msleep(1000)
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue

            self.frame_received.emit(self.idx, frame)

        cap.release()
        print(f"[INFO] Stopped thread {self.idx}")

    def stop(self):
        self.running = False
        self.wait(1000)


class ViewerDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("viewer.ui", self)
        self.show()

        # Button bindings
        self.pButton_0.clicked.connect(lambda: self.start_camera(0))
        self.pButton_1.clicked.connect(lambda: self.start_camera(1))
        self.pButton_2.clicked.connect(lambda: self.start_camera(2))

        # Thread placeholders
        self.threads = [None, None, None]

    def start_camera(self, idx):
        url_edit = [self.lE_0, self.lE_1, self.lE_2][idx]
        label = [self.l_0, self.l_1, self.l_2][idx]
        url = url_edit.text().strip()

        if not url:
            QtWidgets.QMessageBox.warning(self, "Warning", f"Camera {idx} URL is empty")
            return

        # Stop existing thread if running
        if self.threads[idx] and self.threads[idx].isRunning():
            self.threads[idx].stop()

        # Start new thread
        thread = VideoThread(idx, url)
        thread.frame_received.connect(lambda i, f, lab=label: self.update_frame(i, f, lab))
        thread.start()
        self.threads[idx] = thread
        print(f"[INFO] Camera {idx} started")

    def update_frame(self, idx, frame, label):
        # Convert to QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio))

    def closeEvent(self, event):
        for t in self.threads:
            if t and t.isRunning():
                t.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dlg = ViewerDialog()
    dlg.exec_()
