import os, serial, cv2, sys, time, numpy as np, threading, binascii, queue
from flask import Flask, Response

main_queue = queue.Queue()
uart = serial.Serial("/dev/ttyACM0", 921600, timeout=0.1)

app = Flask(__name__)

latest_frame = None
lock = threading.Lock()

def th():
    global uart, main_queue
    sending_1 = bytes([0x02,0x00,0x04,0x00,0x01,0x55,0xaa,0x03,0xFA])
    sending_2 = bytes([0x02,0x00,0x04,0x01,0x01,0x00,0x05,0x03,0x01])
    sending_4 = bytes([0x02,0x00,0x04,0x02,0x01,0x01,0x01,0x03,0x07])

    frame = np.zeros(4800, dtype=np.uint16)

    time.sleep(0.1)
    print("second command to fly")
    uart.write(sending_2)
    time.sleep(0.1)
    uart.write(sending_4)

    # 초기 sync
    uart.write(sending_1)
    uart.read(9)  # 9바이트 버림
    uart.write(sending_4)

    first = True
    image_cnt = 0

    while True:
        try:
            header = uart.read(20)  # 한번에 읽기
            if len(header) < 20 or header[0] != 0x02 or header[1] != 0x25 or header[2] != 0xA1:
                continue

            # 이미지 데이터 9600 바이트 bulk read
            data = uart.read(9600)
            if len(data) < 9600:
                continue

            # 16비트 단위 변환
            frame = np.frombuffer(data, dtype=np.uint16, count=4800)

            # 에러 필터
            if 7 < frame.mean() < 8:
                continue

            main_queue.put(frame)

        except Exception as e:
            continue

def frame_worker():
    global latest_frame
    while True:
        frame1 = main_queue.get()
        max_val, min_val = frame1.max(), frame1.min()
        if max_val == min_val:
            continue

        # normalize to [0,255]
        frame1 = ((frame1 - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
        image = frame1.reshape(60, 80)

        grayImage = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        grayImage = cv2.resize(grayImage, (320, 240), interpolation=cv2.INTER_NEAREST)
        grayImage = cv2.flip(grayImage, 1)

        ret, jpeg = cv2.imencode('.jpg', grayImage, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            with lock:
                latest_frame = jpeg.tobytes()

def generate():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t1 = threading.Thread(target=th, daemon=True)
    t1.start()
    t2 = threading.Thread(target=frame_worker, daemon=True)
    t2.start()
    app.run(host="0.0.0.0", port=1234, threaded=True)
