import cv2
import csv
import time
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

# ouvrir le fichier UNE seule fois
file = open("eye_positions.csv", "a", newline="")
writer = csv.writer(file)
writer.writerow(['time', 'fix_x', 'fix_y'])

while True:
    ret, frame = webcam.read()
    if not ret:
        continue

    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    x = gaze.horizontal_ratio()
    y = gaze.vertical_ratio()

    # si pas de visage détecté
    if x is None or y is None:
        cv2.imshow("Demo", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    text = f"x={x:.2f} y={y:.2f}"

    cv2.putText(frame, text, (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # export propre
    writer.writerow([time.time(), x, y])

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

# cleanup
webcam.release()
file.close()
cv2.destroyAllWindows()
