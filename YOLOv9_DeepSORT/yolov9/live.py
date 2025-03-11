import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

# Configurations
YOLO_WEIGHTS = 'weights/yolov9-s.pt'  # Path to YOLO model weights
COCO_CLASSES_PATH = '../configs/coco.names'  # Path to class labels
CONFIDENCE_THRESHOLD = 0.45  # Confidence threshold for detections
BLUR_PEOPLE = False  # Set True to blur detected people (class ID 0)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    
    # Load YOLO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights=YOLO_WEIGHTS, device=device)
    model = AutoShape(model)
    
    # Load COCO class labels
    with open(COCO_CLASSES_PATH, 'r') as f:
        class_names = f.read().strip().split("\n")
    
    # Generate random color for "person" class (ID 0)
    person_color = tuple(np.random.randint(0, 255, 3).tolist())
    
    # Initialize DeepSort tracker
    tracker = DeepSort(
        max_iou_distance=0.8,
        max_age=10,
        n_init=3,
        nms_max_overlap=0.5,
        max_cosine_distance=0.35,
        nn_budget=100)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO model on the frame
        results = model(frame)
        detections = results.xyxy[0]  # YOLO detections as (x1, y1, x2, y2, conf, class_id)
        
        detect = []
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = map(float, det)
            class_id = int(class_id)
            
            # Filter detections for "person" class (ID 0) and confidence threshold
            if class_id == 0 and confidence >= CONFIDENCE_THRESHOLD:
                detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
        
        # Update tracks with detections
        tracks = tracker.update_tracks(detect, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            # Extract tracking details
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            
            # Draw bounding box and label with ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), person_color, 2)
            label = f"ID {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 2)
            
            # Apply blur if enabled
            if BLUR_PEOPLE:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 30)
        
        # Show the frame
        cv2.imshow('Live Person Detection with ID', frame)
        
        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()