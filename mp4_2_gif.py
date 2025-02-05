import os
import cv2

# Paths
VIDEO_ADDRESS = os.path.join("videos", "cars.mp4")
OUTPUT_VIDEO_ADDRESS = os.path.join("output", "tracked_cars.mp4")

# Load the video
video = cv2.VideoCapture(VIDEO_ADDRESS)
if not video.isOpened():
    raise IOError(f"Failed to open video file at {VIDEO_ADDRESS}")

fps = video.get(cv2.CAP_PROP_FPS)  # Get the original FPS from the video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the cascade for vehicle detection
carCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Variables for tracking
carTracker = {}
carLocation1 = {}
carLocation2 = {}
currentCarID = 0

# Create the video writer for output (use original FPS to avoid changing video speed)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_ADDRESS, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale for car detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = carCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Process each detected car
    for (x, y, w, h) in cars:
        matchCarID = None
        for carID in carTracker.keys():
            success, bbox = carTracker[carID].update(frame)
            if success:
                t_x, t_y, t_w, t_h = map(int, bbox)
                if (t_x <= x + w // 2 <= t_x + t_w) and (t_y <= y + h // 2 <= t_y + t_h):
                    matchCarID = carID

        if matchCarID is None:
            # Use CSRT tracker (better for complex scenes)
            tracker = cv2.legacy.TrackerCSRT_create()  # Try CSRT instead of KCF for better tracking
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)
            carTracker[currentCarID] = tracker
            carLocation1[currentCarID] = [x, y, w, h]
            currentCarID += 1

    # Update trackers and draw bounding boxes
    for carID in list(carTracker.keys()):
        success, bbox = carTracker[carID].update(frame)
        if not success:
            print(f"Removing carID {carID} due to poor tracking quality.")
            del carTracker[carID]
            del carLocation1[carID]
            continue

        t_x, t_y, t_w, t_h = map(int, bbox)
        carLocation2[carID] = [t_x, t_y, t_w, t_h]
        cv2.rectangle(frame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release the video capture and writer
video.release()
out.release()
print(f"Tracked video saved at {OUTPUT_VIDEO_ADDRESS}")
