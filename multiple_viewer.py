#!/usr/bin/env python3
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore


class VideoThread(QtCore.QThread):
    frame_received = QtCore.pyqtSignal(int, np.ndarray)

    def __init__(self, idx, url):
        super().__init__()
        self.idx = idx
        self.url = url
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open stream {self.idx}: {self.url}")
            return

        print(f"[INFO] Started streaming {self.idx}: {self.url}")
        while self.running:
            ret, frame = cap.read()
            if not ret:
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


class MultiViewer(QtWidgets.QDialog):
    def __init__(self, num_windows=4):
        super().__init__()
        self.setWindowTitle("Multi Camera Viewer")
        self.resize(1600, 900)

        self.num_windows = num_windows
        self.threads = []
        self.labels = []
        self.url_edits = []
        self.buttons = []

        # --- Layout setup ---
        grid = QtWidgets.QGridLayout(self)
        self.setLayout(grid)

        rows = int(np.sqrt(num_windows))
        cols = int(np.ceil(num_windows / rows))

        for i in range(num_windows):
            frame = QtWidgets.QFrame()
            frame.setStyleSheet("background-color: black; border: 2px solid #333;")
            vbox = QtWidgets.QVBoxLayout(frame)

            label = QtWidgets.QLabel()
            label.setFixedSize(400, 300)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setText(f"window_{i}")
            label.setStyleSheet("background-color: #222; color: white;")
            self.labels.append(label)

            url_edit = QtWidgets.QLineEdit()
            url_edit.setPlaceholderText(f"rtsp://192.168.2.{i+5}:8554/thermal")
            url_edit.setStyleSheet("background-color: #eee;")
            self.url_edits.append(url_edit)

            btn = QtWidgets.QPushButton(f"Start {i}")
            btn.clicked.connect(lambda _, j=i: self.start_camera(j))
            self.buttons.append(btn)

            vbox.addWidget(label)
            vbox.addWidget(url_edit)
            vbox.addWidget(btn)
            grid.addWidget(frame, i // cols, i % cols)

        # --- Control buttons ---
        ctrl_box = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["2", "4", "8", "16"])
        self.combo.setCurrentText(str(num_windows))
        self.combo.currentIndexChanged.connect(self.rebuild_layout)

        stop_all = QtWidgets.QPushButton("Stop All")
        stop_all.clicked.connect(self.stop_all)

        ctrl_box.addWidget(QtWidgets.QLabel("Layout:"))
        ctrl_box.addWidget(self.combo)
        ctrl_box.addWidget(stop_all)
        grid.addLayout(ctrl_box, rows, 0, 1, cols)

    # ----------------------------------------------------------------
    def rebuild_layout(self):
        """Recreate the window with the selected layout."""
        new_count = int(self.combo.currentText())
        self.stop_all()
        self.close()
        dlg = MultiViewer(num_windows=new_count)
        dlg.show()
        dlg.exec_()

    # ----------------------------------------------------------------
    def start_camera(self, idx):
        url = self.url_edits[idx].text().strip()
        if not url:
            QtWidgets.QMessageBox.warning(self, "Warning", f"Camera {idx} URL is empty")
            return

        if idx < len(self.threads) and self.threads[idx] and self.threads[idx].isRunning():
            self.threads[idx].stop()

        thread = VideoThread(idx, url)
        thread.frame_received.connect(lambda i, f, lab=self.labels[idx]: self.update_frame(i, f, lab))
        thread.start()

        if idx < len(self.threads):
            self.threads[idx] = thread
        else:
            self.threads.append(thread)

        print(f"[INFO] Camera {idx} started")

    # ----------------------------------------------------------------
    def update_frame(self, idx, frame, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio))

    # ----------------------------------------------------------------
    def stop_all(self):
        for t in self.threads:
            if t and t.isRunning():
                t.stop()

    def closeEvent(self, event):
        self.stop_all()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dlg = MultiViewer(num_windows=4)  # Default layout: 4 windows
    dlg.exec_()
