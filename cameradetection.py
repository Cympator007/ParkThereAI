import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import cv2
import os
import sys
import glob
# Define constants for visualization
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # Red for detected objects
FREE_COLOR = (0, 255, 0) # Green for free spaces
OCCUPIED_COLOR = (0, 0, 255) # Blue for occupied spaces
DEVICE_PATH = "/dev/video0"          # change if your camera uses /dev/video1, etc.
WINDOW_NAME = "USB Webcam (/dev/video0)"

def calculate_midpoints(park_spaces_coords):
    """Calculates the midpoint for each parking space bounding box."""
    midpoints = []
    for space in park_spaces_coords:
        # space format is [x_start, x_end, y_start, y_end]
        mid_x = space[0] + (space[1] - space[0]) / 2
        mid_y = space[2] + (space[3] - space[2]) / 2
        midpoints.append([mid_x, mid_y])
    return midpoints

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

    success = cv2.imwrite("capture.png", frame)
    if success:
        print("✅  Frame saved to capture.png")
    else:
        sys.exit("❌  Failed to save image.")


def check_overlap(park_coords, car_coords):
    """
    Checks if two rectangular bounding boxes overlap.
    
    Args:
        park_coords: A list [x_start, x_end, y_start, y_end] for the parking space.
        car_coords: A list [x_start, x_end, y_start, y_end] for the detected car.
        
    Returns:
        True if the boxes overlap, False otherwise.
    """
    park_x_start, park_x_end, park_y_start, park_y_end = park_coords
    car_x_start, car_x_end, car_y_start, car_y_end = car_coords

    # Check for non-overlap on the X-axis
    if park_x_end < car_x_start or car_x_end < park_x_start:
        return False

    # Check for non-overlap on the Y-axis
    if park_y_end < car_y_start or car_y_end < park_y_start:
        return False

    # If they don't fall into any non-overlap case, they must overlap.
    return True


def update_parking_status(detection_result, parking_spaces):
    """
    Updates the status of parking spaces based on detection results.

    Args:
        detection_result: The result from the MediaPipe ObjectDetector.
        parking_spaces: A list of dictionaries, where each dictionary represents a parking space.

    Returns:
        The updated list of parking spaces.
    """
    # 1. Reset all parking spaces to "free" initially.
    for space in parking_spaces:
        space["status"] = "free"

    # 2. Get a list of all detected car bounding boxes.
    detected_cars_boxes = []
    for detection in detection_result.detections:
        # Consider only relevant categories
        if detection.categories[0].category_name in ["car", "truck", "bus"]:
            bbox = detection.bounding_box
            car_box = [
                bbox.origin_x, 
                bbox.origin_x + bbox.width, 
                bbox.origin_y, 
                bbox.origin_y + bbox.height
            ]
            detected_cars_boxes.append(car_box)

    # 3. For each parking space, check if it overlaps with any detected car.
    for space in parking_spaces:
        for car_box in detected_cars_boxes:
            if check_overlap(space.get("coords"), car_box):
                space["status"] = "occupied"
                # Once a car is found in the space, no need to check other cars for this space.
                break 
    
    return parking_spaces


def visualize(image, detection_result, parking_spaces) -> np.ndarray:
    """
    Draws bounding boxes for parking spaces and detected objects on the image.
    
    Args:
        image: The input RGB image as a NumPy array.
        detection_result: The list of all "Detection" entities to be visualized.
        parking_spaces: The list of parking space dictionaries with their updated status.
        
    Returns:
        Image with bounding boxes drawn on it.
    """
    # 1. Draw the parking space boxes first
    for p in parking_spaces:
        start_point = p.get("coords")[0], p.get("coords")[2]
        end_point = p.get("coords")[1], p.get("coords")[3]
        
        # Determine color and text based on status
        if p.get("status") == "occupied":
            color = OCCUPIED_COLOR
            status_text = "Occupied"
        else:
            color = FREE_COLOR
            status_text = "Free"
            
        # Draw the rectangle for the parking space
        cv2.rectangle(image, start_point, end_point, color, 3)
        
        # Define text location relative to the parking space box
        text_location = (start_point[0] + 5, start_point[1] + 20)
        cv2.putText(image, status_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, color, FONT_THICKNESS)

    # 2. Draw the detected object boxes on top
    for detection in detection_result.detections:
        # Filter for vehicles
        #if detection.categories[0].category_name in ["car", "truck", "bus", "motorcycle"]:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            
        # Draw bounding_box for the detected vehicle
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

            # Prepare and draw label with score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f'{category_name} ({probability})'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


def check(scheduler):
    
    scheduler.enter(10, 1, check, (scheduler,))
    """
    Main function to run the parking space detection and visualization.
    """
    # --- Configuration ---
    capture()
    IMAGE_FILE = 'capture.png'
    MODEL_FILE = './model.tflite'
    
    # Coordinates for each parking space: [x_start, x_end, y_start, y_end]
    ParkSpacesCoords = [[243, 435, 169, 215], [313, 550, 227, 308], [555, 1142, 294, 572], [555, 1142, 294, 572],  [555, 1142, 294, 572],  [555, 1142, 294, 572],  [555, 1142, 294, 572],  [555, 1142, 294, 572]]
    
    # Corresponding map coordinates (example data)
    CoordsMap = [[49.736658, 13.381767], [49.736641, 13.381762], [49.736618, 13.381754], [49.736593, 13.381742], [49.736572, 13.381731], [49.736544, 13.381722], [49.736526, 13.381717], [49.736502, 13.381707]]

    # --- Initialization ---
    
    # Calculate midpoints for the parking spaces
    midpoints = calculate_midpoints(ParkSpacesCoords)
    
    # Create the main list of parking space objects (PS)
    PS = []
    for i in range(len(ParkSpacesCoords)):
        PS.append({
            "index": i,
            "coords": ParkSpacesCoords[i],
            "mpoint": midpoints[i],
            "status": "free",  # Initial status
            "coordsmap": CoordsMap[i]
        })

    # --- MediaPipe Object Detection ---

    # STEP 1: Create an ObjectDetector object.
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5  # Set a confidence threshold
        )
        detector = vision.ObjectDetector.create_from_options(options)
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return None

    # STEP 2: Load the input image.
    try:
        image = mp.Image.create_from_file(IMAGE_FILE)
        img_for_drawing = cv2.imread(IMAGE_FILE)
        if img_for_drawing is None:
            raise FileNotFoundError(f"Could not read image file: {IMAGE_FILE}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    # STEP 3: Detect objects in the input image.
    detection_result = detector.detect(image)
    
    # --- Process and Visualize ---

    # STEP 4: Update the status of parking spaces based on detections.
    PS = update_parking_status(detection_result, PS)
    print("Final Parking Status:", PS)

    # STEP 5: Visualize the results.
    annotated_image = visualize(img_for_drawing, detection_result, PS)
    
    # Save the final image
    try:
        cv2.imwrite("annotated_image.png", annotated_image)
        print("Successfully saved annotated_image.png")
    except Exception as e:
        print(f"Error saving image: {e}")
    with open("data.txt", "w") as outfile:
        print(PS)
        outfile.write(json.dumps(PS))
    return PS

# --- Run the main function ---
