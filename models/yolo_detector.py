from ultralytics import YOLO
model = YOLO("yolov8n.pt")

def detect_persons(frame):
  results = model(frame)
  detections = []
  
  for result in results:

    for box in result.boxes:

      class_id = int(box.cls[0])
      
      # we only keep person
      if class_id == 0:
        x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )
        detections.append(
                    (x1, y1, x2, y2)
                )
  return detections