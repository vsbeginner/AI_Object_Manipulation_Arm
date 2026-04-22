import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Use your DroidCam index (usually 1)
cap = cv2.VideoCapture(1)

target = input("Enter object to detect (e.g. bottle, cell phone, book): ").lower()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    found = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id].lower()
        conf = float(box.conf[0])

        if conf < 0.5:
            continue

        if class_name == target:
            found = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            # Compute center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({cx},{cy})",
                        (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            # Print coordinates in terminal
            print("Target position (pixels):", cx, cy)

    if not found:
        cv2.putText(frame, "Target not found",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("YOLO Target Coordinates", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
