import cv2
import os
import sys
import glob
import numpy as np

OUTPUT_IMAGE = "capture.png"
BRIGHTNESS_REDUCTION = 50  # Adjust this value as needed (0–255)

def find_first_camera_device() -> str:
    video_devices = sorted(glob.glob("/dev/video*"))
    for device in video_devices:
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            cap.release()
            print(f"✅  Using camera: {device}")
            return device
    sys.exit("❌  No accessible video devices found.")

def capture():
    device_path = find_first_camera_device()

    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        sys.exit(f"❌  Could not open camera at {device_path}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        sys.exit("⚠️  Frame grab failed; exiting.")

    # Lower brightness by subtracting a value from each pixel
    frame_darker = cv2.subtract(frame, np.full(frame.shape, BRIGHTNESS_REDUCTION, dtype=np.uint8))

    success = cv2.imwrite(OUTPUT_IMAGE, frame_darker)
    if success:
        print(f"✅  Darkened frame saved to {OUTPUT_IMAGE}")
    else:
        sys.exit("❌  Failed to save image.")

if __name__ == "__main__":
    capture()
