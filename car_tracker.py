import os
import cv2
import math

# --------------------------------------------------------------------------------
# مسیرهای ورودی/خروجی را تغییر دهید
# --------------------------------------------------------------------------------
INPUT_VIDEO = os.path.join("videos", "cars.mp4")
TRACKED_VIDEO_OUTPUT = os.path.join("output", "tracked_video.mp4")
MODEL_ADDRESS = os.path.join("models", "myhaar.xml")

# --------------------------------------------------------------------------------
# بارگذاری مدل Haar
# --------------------------------------------------------------------------------
carCascade = cv2.CascadeClassifier(MODEL_ADDRESS)
if carCascade.empty():
    raise IOError(f"Failed to load Haar Cascade model at: {MODEL_ADDRESS}")

# --------------------------------------------------------------------------------
# ویدیوی ورودی
# --------------------------------------------------------------------------------
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {INPUT_VIDEO}")

# اطلاعات ویدیو: فریم‌ریت و ابعاد فریم
input_fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --------------------------------------------------------------------------------
# تنظیماتی برای ردیابی ویدیوی خروجی
# --------------------------------------------------------------------------------

# SKIP_FACTOR: هر چند فریم یک‌بار در فایل نهایی ثبت شود
# مثال: SKIP_FACTOR=2 یعنی در عمل نصف فریم‌ها در فایل خروجی ذخیره شوند
SKIP_FACTOR = 2

# fps خروجی => اگر ورودی 30 باشد و بخواهید نیمی از فریم‌ها را بنویسید، می‌شود 15
output_fps = input_fps / SKIP_FACTOR

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(TRACKED_VIDEO_OUTPUT, fourcc, output_fps, (width, height))

# --------------------------------------------------------------------------------
# متغیرهای ردیابی
# --------------------------------------------------------------------------------
rectangleColor = (0, 255, 0)
carTracker = {}
carLocation1 = {}
carLocation2 = {}
currentCarID = 0
speed_dict = {}

# هر n فریم یکبار دیتکشن شود
DETECTION_INTERVAL = 10

def estimateSpeed(location1, location2, fps_used):
    """
    محاسبه سرعت بر اساس موقعیت پیکسل‌ها.
    در اینجا fps_used = input_fps است، چون ما نمی‌خواهیم فرمول سرعت با کاهش فریم‌ خروجی خراب شود.
    """
    (x1, y1) = location1
    (x2, y2) = location2
    d_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    ppm = 8.8  # پیکسل بر متر (مثال)
    d_meters = d_pixels / ppm

    # تبدیل متر/ثانیه به کیلومتر بر ساعت:
    km_per_hr = d_meters * fps_used * 3.6
    return km_per_hr

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # کپی برای رسم نتیجه
    result_frame = frame.copy()

    # هر DETECTION_INTERVAL فریم، تشخیص خودرو
    if frame_idx % DETECTION_INTERVAL == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = carCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        print(f"Frame {frame_idx}: Detected {len(cars)} cars.")

        for (x, y, w, h) in cars:
            matchCarID = None
            for carID in list(carTracker.keys()):
                success, bbox = carTracker[carID].update(frame)
                if success:
                    t_x, t_y, t_w, t_h = map(int, bbox)
                    carLocation2[carID] = [t_x, t_y, t_w, t_h]
                    cv2.rectangle(result_frame, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)
                    
                    # بررسی اینکه خودرو جدید در محدوده Tracker فعلی هست یا خیر
                    if (t_x <= x + w//2 <= t_x + t_w) and (t_y <= y + h//2 <= t_y + t_h):
                        matchCarID = carID
                        break
                else:
                    # ردیابی ضعیف => حذف
                    print(f"Removing carID {carID} due to poor tracking quality.")
                    carTracker.pop(carID, None)
                    carLocation1.pop(carID, None)
                    carLocation2.pop(carID, None)

            # اگر هیچ ردیابی منطبق نبود => ایجاد Tracker جدید
            if matchCarID is None:
                tracker = cv2.legacy.TrackerKCF_create()
                bbox = (x, y, w, h)
                tracker.init(frame, bbox)
                carTracker[currentCarID] = tracker
                carLocation1[currentCarID] = [x, y, w, h]
                currentCarID += 1
    else:
        # آپدیت Tracker در فریم‌های غیردیتکشن
        for carID in list(carTracker.keys()):
            success, bbox = carTracker[carID].update(frame)
            if not success:
                print(f"Removing carID {carID} due to poor tracking quality.")
                carTracker.pop(carID, None)
                carLocation1.pop(carID, None)
                carLocation2.pop(carID, None)
            else:
                t_x, t_y, t_w, t_h = map(int, bbox)
                carLocation2[carID] = [t_x, t_y, t_w, t_h]
                cv2.rectangle(result_frame, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 2)

    # محاسبهٔ سرعت برای هر خودرو
    for carID in carTracker.keys():
        if carID in carLocation1 and carID in carLocation2:
            x1, y1, w1, h1 = carLocation1[carID]
            x2, y2, w2, h2 = carLocation2[carID]

            # اگر واقعاً مکان تغییر کرده باشد
            if (x1, y1) != (x2, y2):
                # می‌توانید شرط خاص y را حذف یا تغییر دهید
                if speed_dict.get(carID) is None:
                    speed_dict[carID] = estimateSpeed([x1, y1], [x2, y2], input_fps)
                else:
                    # اگر می‌خواهید هر بار سرعت آپدیت شود، می‌توانید اینجا دوباره محاسبه کنید
                    speed_dict[carID] = estimateSpeed([x1, y1], [x2, y2], input_fps)

            # بروزرسانی مکان قبلی
            carLocation1[carID] = [x2, y2, w2, h2]

            # نمایش سرعت
            if speed_dict.get(carID) is not None:
                speed_text = f"{int(speed_dict[carID])} km/hr"
                cv2.putText(
                    result_frame,
                    speed_text,
                    (x2, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

    # --------------------------------------------------------------------------------
    # نوشتن نتیجه در خروجی با فریم‌ریت کمتر
    # --------------------------------------------------------------------------------
    # هر SKIP_FACTOR فریم یک‌بار بنویس
    if frame_idx % SKIP_FACTOR == 0:
        out.write(result_frame)

    frame_idx += 1

cap.release()
out.release()
print(f"Done! Output video saved at: {TRACKED_VIDEO_OUTPUT}")
