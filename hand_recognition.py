import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque, Counter

GESTURES_PATH = 'gestures.csv'
K_NEIGHBORS = 5
MATCH_THRESHOLD = 0.35      # max normalized distance to accept a match
SMOOTHING_WINDOW = 7        # last N predictions used for majority vote

# initializing mediapipe
mpHands = mp.solutions.hands    # this performs the hand recognition
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)    #this line configures the model
mpDraw = mp.solutions.drawing_utils #this line draws the detected keypoints


def normalize_landmarks(points):
    """Translate to wrist origin and scale by max distance from wrist.

    points: (21, 2) array of (x, y).
    Returns a flat (42,) vector invariant to position and hand size.
    """
    pts = np.asarray(points, dtype=np.float32)
    pts = pts - pts[0]  # wrist as origin
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts = pts / scale
    return pts.flatten()


def load_gesture_db(path):
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    labels = df['gesture_name'].to_numpy()
    coords = df.drop(columns=['gesture_name']).to_numpy(dtype=np.float32)
    coords = coords.reshape(-1, 21, 2)
    features = np.stack([normalize_landmarks(row) for row in coords])
    return features, labels


def predict_gesture(feature, db_features, db_labels):
    if db_features is None or len(db_features) == 0:
        return None, None
    dists = np.linalg.norm(db_features - feature, axis=1)
    k = min(K_NEIGHBORS, len(dists))
    idx = np.argpartition(dists, k - 1)[:k]
    nearest_labels = db_labels[idx]
    nearest_dists = dists[idx]
    values, counts = np.unique(nearest_labels, return_counts=True)
    label = values[counts.argmax()]
    mean_dist = float(nearest_dists[nearest_labels == label].mean())
    if mean_dist > MATCH_THRESHOLD:
        return None, mean_dist
    return label, mean_dist


def draw_top_banner(frame, text, color):
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


db_features, db_labels = load_gesture_db(GESTURES_PATH)
if db_features is None:
    print(f'Warning: {GESTURES_PATH} not found. Run save_gestures.py first.')
else:
    print(f'Loaded {len(db_features)} samples across '
          f'{len(set(db_labels))} gestures: {sorted(set(db_labels))}')

prediction_history = deque(maxlen=SMOOTHING_WINDOW)

# initializing webcam for video capture
cap = cv2.VideoCapture(2)

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)  # flip frame vertically

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)   # get hand landmark predictions

    current_label = None
    current_dist = None

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            points = []
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                points.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(3, 252, 244), thickness=2, circle_radius=2),
                                  mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            feature = normalize_landmarks(points)
            label, dist = predict_gesture(feature, db_features, db_labels)

            # use only the first hand for the global banner / smoothing
            if current_label is None and current_dist is None:
                current_label = label
                current_dist = dist

            wrist_x, wrist_y = points[0]
            per_hand_text = label if label is not None else '...'
            per_hand_color = (0, 255, 0) if label is not None else (0, 0, 255)
            cv2.putText(frame, per_hand_text, (wrist_x, max(wrist_y - 20, 30)),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.0, per_hand_color, 2)

    # temporal smoothing across recent frames
    prediction_history.append(current_label)
    if prediction_history:
        most_common, count = Counter(prediction_history).most_common(1)[0]
    else:
        most_common, count = None, 0

    if db_features is None:
        draw_top_banner(frame, "No gestures.csv — run save_gestures.py first",
                        (0, 0, 255))
    elif most_common is not None and count >= (SMOOTHING_WINDOW // 2 + 1):
        dist_text = f"  (dist={current_dist:.2f})" if current_dist is not None else ""
        draw_top_banner(frame, f"Gesture: {most_common}{dist_text}",
                        (0, 255, 0))
    elif current_dist is not None:
        draw_top_banner(frame, f"No match  (dist={current_dist:.2f})",
                        (0, 165, 255))
    else:
        draw_top_banner(frame, "Show your hand to the camera",
                        (200, 200, 200))

    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
