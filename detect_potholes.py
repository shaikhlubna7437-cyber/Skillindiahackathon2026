import cv2
import time
import random
from ultralytics import YOLO

def get_fleet_gps_telemetry():
    """
    Simulates real-time telemetry from a transit bus or fleet vehicle's GPS.
    In production, replace this with a live API pull, serial port reading (NMEA),
    or an MQTT subscription to your fleet's IoT gateway.
    """
    # Mock coordinates around an urban route
    mock_lat = 12.9716 + random.uniform(-0.005, 0.005)
    mock_lng = 77.5946 + random.uniform(-0.005, 0.005)
    return round(mock_lat, 6), round(mock_lng, 6)

def process_urban_fleet_video(video_path, model_path='yolov8n.pt'):
    """
    Processes video input from a transit fleet dashcam, detects potholes in real time,
    and logs their physical geolocation coordinates.
    """
    # 1. Load your trained model (e.g., fine-tuned on custom pothole datasets)
    # If you have a custom weights file, replace 'yolov8n.pt' with 'best.pt'
    print(f"[INFO] Initializing Urban Intelligence AI Engine with {model_path}...")
    model = YOLO(model_path)
    
    # 2. Initialize Video Stream (Accepts file path or live RTSP network stream)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file or stream at: {video_path}")
        return

    print("[INFO] Processing stream. Press 'q' to stop.")
    
    pothole_class_id = 0  # Adjust according to your custom dataset class map index
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video file or stream disconnect.")
            break
            
        # 3. Object Detection Inference
        # stream=True optimizes memory efficiency for continuous video data
        results = model(frame, stream=True, verbose=False)
        
        # 4. Fetch the real-time vehicle spatial telemetry at the exact frame interval
        current_lat, current_lng = get_fleet_gps_telemetry()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Filter results strictly to your pothole class
                cls = int(box.cls[0])
                if cls == pothole_class_id:
                    conf = float(box.conf[0])
                    
                    # Optional: Set a confidence threshold rule to eliminate false positives
                    if conf > 0.50:
                        # Extract bounding box matrix mapping
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Real-Time Console Reporting for the Fleet Management Dashboard
                        print(f"[{current_time}] ⚠️ POTHOLE DETECTED | Conf: {conf:.2f} | Fleet Location: Lat {current_lat}, Lng {current_lng}")
                        
                        # 5. Visual Overlays for Edge Devices / Diagnostic Logs
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label = f"Pothole: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0, 0, 255), 2)
                        
                        # Displaying metadata coordinates directly on live UI feed
                        telemetry_str = f"GPS: {current_lat}, {current_lng}"
                        cv2.putText(frame, telemetry_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 6. Show the frame output visually
        cv2.imshow("Urban Fleet Intelligence - Real-Time Pothole Engine", frame)
        
        # Breakdown flag to safely terminate the process loop using the escape key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Processing loop closed successfully.")
if __name__ == "__main__":
    # Provide the target video path (or pass 0 to leverage a local connected hardware camera)
    VIDEO_SOURCE = "road_surface_footage.mp4" 
    
    # Run the pipeline
    process_urban_fleet_video(VIDEO_SOURCE)