import cv2
import os

# ============================================
# CHANGE THIS for each gesture you collect:
# "thumbs_up"   or   "thumbs_down"   or   "open_hand"
gesture_name = "open_hand"
# ============================================

save_folder = f"dataset/raw/{gesture_name}"
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 15
total_needed = 100

print(f"Collecting images for: {gesture_name}")
print("Press S to save an image")
print("Press Q to quit")

while count < total_needed:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found! Try changing VideoCapture(0) to VideoCapture(1)")
        break

    display = frame.copy()
    cv2.putText(display, f"Gesture: {gesture_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, f"Saved: {count}/{total_needed}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, "Press S to save | Q to quit", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Collecting Dataset", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = f"{save_folder}/{gesture_name}_{count:03d}.jpg"
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Saved {count}/{total_needed}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Collected {count} images for {gesture_name}")