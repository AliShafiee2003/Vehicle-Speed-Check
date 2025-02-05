# Real-Time Car Detection and Tracking

This project simulates real-time video processing and vehicle detection in a video using the **Haar Cascade Classifier** and **KCF Tracker**. In this project, vehicles are detected and tracked in the video with green bounding boxes. Additionally, the speed of each vehicle is calculated and displayed in kilometers per hour.

## Features

- Vehicle detection using the **Haar Cascade** model.
- Vehicle tracking in video using the **KCF Tracker** algorithm.
- Speed estimation for each vehicle based on its movement.
- Saving the output video with reduced frame rate to decrease file size.

## Prerequisites

To run this project, you'll need to install the following Python libraries:

- Python 3.x
- OpenCV
- Numpy

To install the necessary dependencies, use the following command:

```bash
pip install opencv-python numpy
```

## How to Use

### 1. Set File Paths

Before running the project, you'll need to set the correct file paths for your input video and output video files. The file paths are defined in the code like this:

```python
INPUT_VIDEO = os.path.join("videos", "cars.mp4")
TRACKED_VIDEO_OUTPUT = os.path.join("output", "tracked_video.mp4")
MODEL_ADDRESS = os.path.join("models", "myhaar.xml")
```

- **INPUT_VIDEO**: Path to the input video containing vehicles.
- **TRACKED_VIDEO_OUTPUT**: Path where the output video with tracked vehicles will be saved.
- **MODEL_ADDRESS**: Path to the Haar Cascade model used for vehicle detection.

### 2. Load the Haar Model

The Haar model must be loaded before running the code. If the model file is not found in the specified path, the program will raise an error.

### 3. Running the Code

You can run the code by executing the following command:

```bash
python car_tracking.py
```

### 4. How It Works

- The program processes the video frame by frame, detecting vehicles every **DETECTION_INTERVAL** frames (this can be adjusted).
- A **Tracker** is created for each detected vehicle and used to track it across frames.
- Every **SKIP_FACTOR** frames (which can be adjusted), the tracked video is saved to the output.
- The speed of each vehicle is calculated based on its movement and displayed as text on the video.

### 5. Adjustable Settings

- **SKIP_FACTOR**: This variable determines how many frames are skipped before saving to the output video. For example, if set to 2, only half of the frames are saved.
- **DETECTION_INTERVAL**: This variable specifies how many frames to skip before performing vehicle detection.

## Example

To run the project, use the following command:

```bash
python car_tracking.py
```

This will process the video and save the tracked video at the specified output path.

## Limitations and Notes

- This project uses **Haar Cascade Classifier** for vehicle detection, which may not perform well in low-light conditions or with unusual camera angles.
- The speed calculation is based on pixel movement, and for more accurate results, you should adjust the **ppm** value correctly.

## Files

- `car_tracking.py`: The main script that performs car detection and tracking.
- `videos/`: Folder containing the input videos.
- `models/`: Folder containing the Haar Cascade model file.
- `output/`: Folder where the output tracked video is saved.

### Video Files

To test the project, you can use the example video provided in the `videos/` folder. The input video should be named **`cars.mp4`**.
