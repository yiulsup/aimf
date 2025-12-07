import cv2
import time
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# --- GStreamer 초기화 ---
Gst.init(None)

# --- 기존에 잘 동작하던 launch_string 그대로 ---
launch_string = (
    "appsrc name=source is-live=true block=true format=time "
    "caps=video/x-raw,format=BGR,width=320,height=240,framerate=0/1 ! "
    "videoconvert ! video/x-raw,format=I420 ! "
    "x264enc tune=zerolatency bitrate=800 speed-preset=ultrafast ! "
    "rtph264pay name=pay0 pt=96 config-interval=1" ! 
    "udpsink host=192.168.0.91 port=5000"
)

# --- 파이프라인 생성 ---
pipeline = Gst.parse_launch(launch_string)
appsrc = pipeline.get_by_name("source")

# --- 파이프라인 실행 ---
pipeline.set_state(Gst.State.PLAYING)

# --- 카메라 입력 ---
cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("✅ 시작: OpenCV → appsrc feeding")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 프레임 읽기 실패")
            break

        # 입력 해상도 맞추기
        frame = cv2.resize(frame, (320, 240))

        # BGR 포맷 그대로 전달
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)

        # 타임스탬프 설정 (임의로 현재 시간 기준)
        buf.pts = buf.dts = int(time.time() * Gst.SECOND)
        buf.duration = Gst.util_uint64_scale(1, Gst.SECOND, 30)

        # GStreamer로 전송
        retval = appsrc.emit("push-buffer", buf)
        if retval != Gst.FlowReturn.OK:
            print("⚠️ push-buffer 오류:", retval)

        # 프레임 간 딜레이 (적절히 조정 가능)
        time.sleep(0.03)

except KeyboardInterrupt:
    print("\n🛑 종료 중...")

finally:
    pipeline.set_state(Gst.State.NULL)
    cap.release()
    print("✅ 종료 완료.")
