import cv2  
import numpy as np  
import supervision as sv  
from ultralytics import YOLO  

def main():
    
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")  

    video_path = "traffic_sample.mp4"  
    output_path = "output_traffic_count.mp4"  

    video_info = sv.VideoInfo.from_video_path(video_path)
    print(f"[INFO] Processing video: {video_info.width}x{video_info.height} @ {video_info.fps:.1f} FPS")

    
    tracker = sv.ByteTrack()  

   
    start_point = sv.Point(x=50, y=int(video_info.height * 0.6))  
    end_point = sv.Point(x=video_info.width - 50, y=int(video_info.height * 0.6))  
    line_zone = sv.LineZone(start=start_point, end=end_point)  

    
    box_annotator = sv.BoxAnnotator(thickness=2)  
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)  
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.8)  

    
    def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
        
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)  

       
        vehicle_classes = [2, 3, 5, 7]
        if len(detections) > 0 and detections.class_id is not None:
            mask = np.isin(detections.class_id, vehicle_classes)
            detections = detections[mask]  
        
        detections = tracker.update_with_detections(detections)

        line_zone.trigger(detections)

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        
       
        annotated_frame = line_annotator.annotate(annotated_frame, line_counter=line_zone)

        return annotated_frame  

    print("[INFO] Running detection & tracking pipeline...")
   
    sv.process_video(
        source_path=video_path,
        target_path=output_path,
        callback=process_frame
    )
    print(f"[SUCCESS] Processed video saved to '{output_path}'!")

if __name__ == "__main__":
    main()