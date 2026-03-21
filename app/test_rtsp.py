import cv2

url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"

print("Connecting...")

cap = cv2.VideoCapture("rtsp://localhost:8554/live")
if not cap.isOpened():
    print("❌ Cannot open RTSP")
    exit()

print("✅ Connected!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not received")
        break

    cv2.imshow("RTSP TEST", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()