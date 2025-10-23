import os, threading, time
import numpy as np
import cv2
from lne_tflite import interpreter as lt
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# ---------------------------------
# GStreamer streamer class
# ---------------------------------
Gst.init(None)

class GstStreamer:
    def __init__(self, host="192.168.0.113", port=10000, width=500, height=375, fps=3):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0  # monotonic timestamp counter

        pipeline_str = (
            f'appsrc name=mysrc is-live=true block=true format=time '
            f'caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 '
            '! videoconvert '
            '! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast key-int-max=30 byte-stream=true '
            '! rtph264pay config-interval=1 pt=96 '
            f'! udpsink host={host} port={port}'
        )

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name("mysrc")
        self.pipeline.set_state(Gst.State.PLAYING)

    def push(self, frame):
        frame = cv2.resize(frame, (self.width, self.height))
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        data = frame.tobytes()

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        frame_duration = Gst.SECOND // self.fps
        buf.pts = buf.dts = self.frame_count * frame_duration
        buf.duration = frame_duration
        self.frame_count += 1

        self.appsrc.emit("push-buffer", buf)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


# ---------------------------------
# AiMF_NPU class
# ---------------------------------
class AiMF_NPU():
    def __init__(self):
        self.cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)                    # V4L2 Video Stream Open
        self.weight = "./LNE/Detection/yolov7_tiny_4x8_0_1.lne"                     # model weight load 위치
        self.interpreter = lt.Interpreter(self.weight)                              # model load
        self.interpreter.allocate_tensors()                                         # model input/ output memory 할당
        self.input_details = self.interpreter.get_input_details()                   # model input에 대한 정보
        self.output_details = self.interpreter.get_output_details()                 # model output에 대한 정보

        self.output_details.sort( key=lambda _:  np.prod(_["shape"][-3:]))          # [13, 13], [26, 26], [52, 52] 순으로 정렬
        self.output_tensors = [_["index"] for _ in self.output_details]             # 모델의 추론후 값 출력에 대한 index
        self.input_shape      = self.input_details[0]["shape"]                      # input image 해상도와 채널
        self.input_tensor     = self.input_details[0]["index"]                      # input image가 저장될 버퍼 index
        self.height = self.input_shape[1]                                           # input height resolution
        self.width = self.input_shape[2]                                            # input width resolution

        self.class_names = {}                                                       # 학습한 모델의 class들을 저장할 공간
        with open("./ObjectDetection/labels/coco.names") as f:
            self.class_names = f.readlines()
            self.class_names = [c.strip() for c in self.class_names]

        # 색상 팔레트
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

        self.lock = threading.Lock()
        self.streamer = None  # streamer will be injected from outside

    def _process_iou(self, boxes, classes, scores, thres_iou):
        nboxes      = len(scores)
        orders      = np.argsort(scores)[::-1]

        ##  Find the area of boxes.
        boxes_area  = np.ndarray(nboxes, dtype=np.float32)
        valids      = np.ndarray(nboxes, dtype=int)
        for i, box in enumerate(boxes):
            box_w       = box[2]-box[0]
            box_h       = box[3]-box[1]
            valids[i]   = (box_w>0) and (box_h>0)
            if (valids[i]): boxes_area[i] = box_w*box_h

        ##  IOU operation.
        for i, id_i in enumerate(orders):
            if not(valids[id_i]):  
                continue

            for id_j in orders[i+1:]:
                if (not(valids[id_j]) or classes[id_i]!=classes[id_j]): 
                    continue

                x1        = max(boxes[id_i][0], boxes[id_j][0])
                y1        = max(boxes[id_i][1], boxes[id_j][1])
                x2        = min(boxes[id_i][2], boxes[id_j][2])
                y2        = min(boxes[id_i][3], boxes[id_j][3])
                if (x2<=x1 or y2<=y1):  
                    continue

                area_ixj  = (x2-x1)*(y2-y1)
                area_iuj  = boxes_area[id_i]+boxes_area[id_j]-area_ixj
                if (area_ixj>=thres_iou*area_iuj):  valids[id_j] = 0

        ##  Detection outputs.
        detects     = []
        for id_i in orders:
            if (valids[id_i]):   detects.append(id_i)

        return [
            len(detects),                       # number of detections
            np.asarray(classes)[detects],       # detection classes
            np.asarray(scores )[detects],       # detection scores
            np.asarray(boxes  )[detects]        # detection boxes
        ]

    def _postprocess_iou(self,
        scores,                             # scores  after score_thresh
        classes,                            # classes after score_thresh
        boxes                               # boxes   after score_thresh
    ):
        boxes_valid   = [True]*len(boxes)           # initialize all boxes valid
        boxes_area    = [_[4]*_[5] for _ in boxes]  # precompute the box area
        score_orders  = np.argsort(scores)[::-1]    # set the NMS order

        for i, id_i in enumerate(score_orders):
            if not(boxes_valid[id_i]):  
                continue

            for id_j in score_orders[i+1:]:
                if (not(boxes_valid[id_j]) or classes[id_i]!=classes[id_j]): 
                    continue

                x0  = max(boxes[id_i][0], boxes[id_j][0])
                y0  = max(boxes[id_i][1], boxes[id_j][1])
                x1  = min(boxes[id_i][2], boxes[id_j][2])
                y1  = min(boxes[id_i][3], boxes[id_j][3])
                if (x1<=x0 or y1<=y0):  continue

                area_ixj  = (x1-x0)*(y1-y0)
                area_iuj  = boxes_area[id_i]+boxes_area[id_j]-area_ixj
                if (area_ixj>=0.3*area_iuj): boxes_valid[id_j] = False

        ##  Detection outputs.
        detects     = []
        for i, id_i in enumerate(score_orders):
            if (i==100):            
                break
            if (boxes_valid[id_i]): detects.append(id_i)

        return [
            np.asarray(scores )[detects],     # post-NMS scores
            np.asarray(classes)[detects],     # post-NMS classes
            np.asarray(boxes  )[detects]      # post-NMS boxes in XYWH format
        ]

    def postprocess_yolov7(self, predicts):
        YOLO_ANCHORS  = [
            [[116, 90], [156,198], [373,326]],    # large  box
            [[ 30, 61], [ 62, 45], [ 59,119]],    # medium box
            [[ 10, 13], [ 16, 30], [ 33, 23]]     # small  box
        ]
        scores,classes,boxes  = [],[],[]

        for p, anchors in enumerate(YOLO_ANCHORS):
            ##  Configure hyperparameters for each pyramid level.
            if (p==0):
                sample  = 32
                GH      = self.input_shape[-3]//32  # grid height
                GW      = self.input_shape[-2]//32  # grid width
                nboxes  = GH*GW*3                   # number of boxes total
            else:
                sample /= 2
                GH     *= 2
                GW     *= 2
                nboxes *= 4

            ##  Decode the coordinates for boxes exceeding the score threshold.
            for boxid in range(nboxes):
                P       = predicts[p][boxid]
                classid = np.argmax(P[5:])      # find the top-1 class
                score   = P[4]*P[5+classid]

                if (score>=0.3):
                    a     =  boxid% 3             # anchor index
                    xg    = (boxid//3)% GW        # grid reference xg
                    yg    = (boxid//3)//GW        # grid reference yg
                    xc    = (xg+P[0]*2-0.5)*sample       
                    yc    = (yg+P[1]*2-0.5)*sample       
                    w     = (   P[2]*2)**2 *anchors[a][0]
                    h     = (   P[3]*2)**2 *anchors[a][1]
                    boxes  .append([              
                        max(xc-w/2, 0),
                        max(yc-h/2, 0),
                        min(xc+w/2, 416),
                        min(yc+h/2, 416),
                        w,
                        h
                    ])
                    classes.append(classid)
                    scores .append(score  )

        return self._postprocess_iou(scores, classes, boxes)
    
    def post_draw(self, img, classes, scores, boxes, model_input_height, model_input_width):
        img = cv2.resize(img, (500,375))
        h, w, _ = img.shape
        h_ratio = h / model_input_height
        w_ratio = w / model_input_width

        font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 1
        font_thick = 1

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)

        for i, c in enumerate(classes):
            label = '{} {:.2f}'.format(self.class_names[c], scores[i])

            left = max(0, int(np.round(boxes[i][0] * w_ratio)))
            top = max(0, int(np.round(boxes[i][1] * h_ratio)))
            right = min(w, int(np.round(boxes[i][2] * w_ratio)))
            bottom = min(h, int(np.round(boxes[i][3] * h_ratio)))

            # 색상 변환 (BGR)
            color = (int(self.colors[c][2]), int(self.colors[c][1]), int(self.colors[c][0]))

            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            label_size = cv2.getTextSize(label, font_face, font_scale, font_thick)[0]
            img = cv2.rectangle(img, (left - 1, top - label_size[1] - 1), 
                                (left + label_size[0] + 1, top), color, -1)
            img = cv2.putText(img, label, (left, top - 2), font_face, 
                            font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

        return img

    def work(self):
        target_interval = 1.0 / 15.0   # 3 FPS
        last_time = time.time()

        while True:
            ret, frame = self.cap.read()                                                
            if not ret:
                continue

            now = time.time()
            if now - last_time < target_interval:
                continue
            last_time = now

            # handle grayscale input
            if len(frame.shape) == 2:   
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            h, w, _ = frame.shape                                                       
            frame_resized = cv2.resize(frame, (416, 416))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32)   
            frame_rgb = frame_rgb / 255.0   
            frame_rgb = np.expand_dims(frame_rgb, axis=0).astype(np.float32)  

            self.interpreter.set_tensor(self.input_details[0]['index'], frame_rgb)           
            self.interpreter.invoke()                                                   
            outputs   = [self.interpreter.get_tensor(_) for _ in self.output_tensors]   
            predicts  = [np.reshape(_, (-1,len(self.class_names)+5)) for _ in outputs]  

            max_conf = max([p[:,4].max() for p in predicts])
            print(f"[DEBUG] Max objectness: {max_conf:.4f}")

            scores, classes, boxes = self.postprocess_yolov7(predicts)                  
            detected_image = self.post_draw(frame_resized, classes, scores, boxes, 416, 416)    

            # Push frame to GStreamer
            if self.streamer:
                self.streamer.push(detected_image)


# ---------------------------------
# Main
# ---------------------------------
if __name__ == "__main__":
    streamer = GstStreamer(host="192.168.0.113", port=10001)
    npu = AiMF_NPU()
    npu.streamer = streamer

    t = threading.Thread(target=npu.work, daemon=True)
    t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        streamer.stop()
        print("Stopped streaming.")

