
  import cv2
from ultralytics import YOLO

# === Load your trained model ===
model = YOLO("best.pt")  # Replace with your .pt path

def get_bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)

def estimate_offset_point(lane_point, direction, offset_px):
    return (lane_point[0] + direction * offset_px, lane_point[1])

def blend_midline(prev, new, alpha=0.2):
    if prev is None:
        return new
    return (
        alpha * new[0] + (1 - alpha) * prev[0],
        alpha * new[1] + (1 - alpha) * prev[1]
    )

def compute_smooth_midline(boxes, classes, prev_midline=None, lane_width_px=160, alpha=0.2):
    solid_pts = []
    broken_pts = []

    for i, cls in enumerate(classes):
        box = boxes[i]
        center = get_bottom_center(box)
        if cls == "Solid_Line_Lane":
            solid_pts.append(center)
        elif cls == "Broken_Line_Lane":
            broken_pts.append(center)

    solid_pts = sorted(solid_pts, key=lambda p: p[1], reverse=True)
    broken_pts = sorted(broken_pts, key=lambda p: p[1], reverse=True)

    solid_pt = solid_pts[0] if solid_pts else None
    broken_pt = broken_pts[0] if broken_pts else None

    if solid_pt and broken_pt:
        new_midline = (
            (solid_pt[0] + broken_pt[0]) / 2,
            (solid_pt[1] + broken_pt[1]) / 2
        )
    elif solid_pt:
        new_midline = estimate_offset_point(solid_pt, direction=+1, offset_px=lane_width_px/2)
    elif broken_pt:
        new_midline = estimate_offset_point(broken_pt, direction=-1, offset_px=lane_width_px/2)
    else:
        return prev_midline or (320, 480)  # fallback

    return blend_midline(prev_midline, new_midline, alpha)

# === Run on video ===
video_path = "input_video.mp4"  # Change to your input
cap = cv2.VideoCapture(video_path)
prev_midline = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    class_names = [model.names[int(c)] for c in class_ids]

    midline_point = compute_smooth_midline(boxes, class_names, prev_midline)
    prev_midline = midline_point

    if midline_point:
        cv2.circle(frame, (int(midline_point[0]), int(midline_point[1])), 6, (0, 255, 0), -1)

    cv2.imshow("Drive Path", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
