import numpy as np 
import os 
import threading 
import time
from lne_tflite import interpreter as lt
import cv2 

class AiMF_NPU():
    def __init__(self):
        self.cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # FPS: 10
        self.cap.set(cv2.CAP_PROP_FPS, 10)

        # 확인용 출력
        print("Width :", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Height:", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS   :", self.cap.get(cv2.CAP_PROP_FPS))
        self.weight = "./LNE/Detection/yolov7_tiny_4x8_0_1.lne"
        self.interpreter = lt.Interpreter(self.weight)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.output_details.sort(key=lambda _: np.prod(_["shape"][-3:]))
        self.output_tensors = [_["index"] for _ in self.output_details]
        self.input_shape = self.input_details[0]["shape"]
        self.input_tensor = self.input_details[0]["index"]

        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

        with open("./ObjectDetection/labels/coco.names") as f:
            self.class_names = [c.strip() for c in f.readlines()]

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

        # GStreamer UDP 송출 파이프라인 (UDP only)
        gst_str = f"""
            appsrc name=src is-live=true block=true format=GST_FORMAT_TIME !
            video/x-raw,format=RGB,width=640,height=480,framerate=30/1 !
            videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast !
            rtph264pay config-interval=1 pt=96 !
            udpsink host="192.168.0.113" port=10000
            """
        
        self.udp_out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30.0, (500, 375))

    def postprocess_fast(self, outputs, conf_thresh=0.3, nms_thresh=0.4):
        """벡터화된 후처리"""
        boxes, confidences, class_ids = [], [], []
        for out in outputs:
            for det in out:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = det[4] * scores[class_id]
                if confidence > conf_thresh:
                    cx, cy, w, h = det[0:4]
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
        final_boxes, final_scores, final_classes = [], [], []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                final_boxes.append([x, y, x + w, y + h])
                final_scores.append(confidences[i])
                final_classes.append(class_ids[i])
        return final_scores, final_classes, final_boxes

    def post_draw(self, img, classes, scores, boxes):
        img = cv2.resize(img, (500, 375))
        for i, c in enumerate(classes):
            label = '{} {:.2f}'.format(self.class_names[c], scores[i])
            left, top, right, bottom = boxes[i]
            color = (int(self.colors[c][2]), int(self.colors[c][1]), int(self.colors[c][0]))
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            cv2.putText(img, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 2, cv2.LINE_AA)
        return img

    def work(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 전처리 최적화
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
            blob = np.transpose(blob, (0, 2, 3, 1))   # (1,416,416,3)으로 변환
            self.interpreter.set_tensor(self.input_tensor, blob.astype(np.float32))

            self.interpreter.invoke()

            outputs = [self.interpreter.get_tensor(_) for _ in self.output_tensors]
            outputs = [np.reshape(_, (-1, len(self.class_names)+5)) for _ in outputs]

            scores, classes, boxes = self.postprocess_fast(outputs)
            detected_image = self.post_draw(frame, classes, scores, boxes)

            # (UDP 전송)
            if self.udp_out.isOpened():
                self.udp_out.write(detected_image)

if __name__ == "__main__":
    npu = AiMF_NPU()
    t = threading.Thread(target=npu.work, daemon=True)
    t.start()

    # 메인 스레드는 계속 대기
    while True:
        time.sleep(1)
