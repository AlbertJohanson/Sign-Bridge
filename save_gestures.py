import os
import pandas as pd
import cv2
import mediapipe as mp
import argparse


GESTURES_PATH = 'gestures.csv'
TOTAL_DATAPOINTS = 50
WARMUP_FRAMES = 50

parser = argparse.ArgumentParser(
    description='Save gesture keypoints in gestures.csv')
parser.add_argument('-n', '--new', action='store_true',
                    help='Overwrite the previously collected data.')

args = parser.parse_args()

if args.new:
    if os.path.exists(GESTURES_PATH):
        os.remove(GESTURES_PATH)

mpHands = mp.solutions.hands    # this performs the hand recognition
# this line configures the model
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# initializing webcam for video capture
cap = cv2.VideoCapture(2)

handpoints = ['HandLandmark.WRIST_lmx', 'HandLandmark.WRIST_lmy', 'HandLandmark.THUMB_CMC_lmx', 'HandLandmark.THUMB_CMC_lmy', 'HandLandmark.THUMB_MCP_lmx', 'HandLandmark.THUMB_MCP_lmy', 'HandLandmark.THUMB_IP_lmx', 'HandLandmark.THUMB_IP_lmy', 'HandLandmark.THUMB_TIP_lmx', 'HandLandmark.THUMB_TIP_lmy', 'HandLandmark.INDEX_FINGER_MCP_lmx', 'HandLandmark.INDEX_FINGER_MCP_lmy', 'HandLandmark.INDEX_FINGER_PIP_lmx', 'HandLandmark.INDEX_FINGER_PIP_lmy', 'HandLandmark.INDEX_FINGER_DIP_lmx', 'HandLandmark.INDEX_FINGER_DIP_lmy', 'HandLandmark.INDEX_FINGER_TIP_lmx', 'HandLandmark.INDEX_FINGER_TIP_lmy', 'HandLandmark.MIDDLE_FINGER_MCP_lmx', 'HandLandmark.MIDDLE_FINGER_MCP_lmy', 'HandLandmark.MIDDLE_FINGER_PIP_lmx',
              'HandLandmark.MIDDLE_FINGER_PIP_lmy', 'HandLandmark.MIDDLE_FINGER_DIP_lmx', 'HandLandmark.MIDDLE_FINGER_DIP_lmy', 'HandLandmark.MIDDLE_FINGER_TIP_lmx', 'HandLandmark.MIDDLE_FINGER_TIP_lmy', 'HandLandmark.RING_FINGER_MCP_lmx', 'HandLandmark.RING_FINGER_MCP_lmy', 'HandLandmark.RING_FINGER_PIP_lmx', 'HandLandmark.RING_FINGER_PIP_lmy', 'HandLandmark.RING_FINGER_DIP_lmx', 'HandLandmark.RING_FINGER_DIP_lmy', 'HandLandmark.RING_FINGER_TIP_lmx', 'HandLandmark.RING_FINGER_TIP_lmy', 'HandLandmark.PINKY_MCP_lmx', 'HandLandmark.PINKY_MCP_lmy', 'HandLandmark.PINKY_PIP_lmx', 'HandLandmark.PINKY_PIP_lmy', 'HandLandmark.PINKY_DIP_lmx', 'HandLandmark.PINKY_DIP_lmy', 'HandLandmark.PINKY_TIP_lmx', 'HandLandmark.PINKY_TIP_lmy']

# state machine
MODE_IDLE = 'idle'
MODE_NAMING = 'naming'
MODE_WARMUP = 'warmup'
MODE_CAPTURING = 'capturing'

mode = MODE_IDLE
gesture_name = ''
typed_name = ''
warmup_count = 0
captured = 0
landmarks_buffer = []
gesture_data = []
status_message = "Press 'c' to add a gesture, 'q' to save and quit."


def draw_overlay(frame, lines, origin=(20, 40), color=(127, 255, 255), scale=0.8):
    y = origin[1]
    for line in lines:
        cv2.putText(frame, line, (origin[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += int(35 * scale + 10)


while True:
    ok, frame = cap.read()
    if not ok:
        continue
    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)  # flip frame vertically

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)   # get hand landmark predictions

    keypress = cv2.waitKey(1) & 0xFF

    # ---- key handling per mode ----
    if mode == MODE_IDLE:
        if keypress == ord('c'):
            mode = MODE_NAMING
            typed_name = ''
        elif keypress == ord('q'):
            break

    elif mode == MODE_NAMING:
        if keypress == 13:  # Enter
            if typed_name.strip():
                gesture_name = typed_name.strip()
                mode = MODE_WARMUP
                warmup_count = 0
                captured = 0
                landmarks_buffer = []
        elif keypress == 27:  # Esc -> cancel
            mode = MODE_IDLE
            typed_name = ''
        elif keypress in (8, 127):  # Backspace / Delete
            typed_name = typed_name[:-1]
        elif 32 <= keypress <= 126:  # printable ASCII
            typed_name += chr(keypress)

    elif mode in (MODE_WARMUP, MODE_CAPTURING):
        if keypress == 27 or keypress == ord('c'):
            # cancel current capture cleanly — discard partials
            mode = MODE_IDLE
            warmup_count = 0
            captured = 0
            landmarks_buffer = []
            status_message = f"Cancelled '{gesture_name}'. Press 'c' for next."

    # ---- mode transitions ----
    if mode == MODE_WARMUP:
        warmup_count += 1
        if warmup_count >= WARMUP_FRAMES:
            mode = MODE_CAPTURING

    # ---- landmark drawing + capture ----
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(
                frame, handslms, mpHands.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(3, 252, 244), thickness=2, circle_radius=2),
                mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            if mode == MODE_CAPTURING:
                lmks = []
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    lmks += [lmx, lmy]
                landmarks_buffer.append(lmks + [gesture_name])
                captured += 1

                if captured >= TOTAL_DATAPOINTS:
                    gesture_data += landmarks_buffer
                    landmarks_buffer = []
                    status_message = (
                        f"Saved '{gesture_name}' "
                        f"({TOTAL_DATAPOINTS} samples). Press 'c' for next.")
                    mode = MODE_IDLE

    # ---- overlay ----
    if mode == MODE_IDLE:
        draw_overlay(frame, [status_message,
                             f"Total saved gestures so far: {len(gesture_data)} samples"])
    elif mode == MODE_NAMING:
        draw_overlay(frame, [
            f"Gesture name: {typed_name}_",
            "Type name, Enter to confirm, Esc to cancel"
        ], color=(127, 255, 127))
    elif mode == MODE_WARMUP:
        remaining = max(WARMUP_FRAMES - warmup_count, 0)
        draw_overlay(frame, [
            f"[{gesture_name}] Get ready... {remaining}",
            "Press 'c' or Esc to cancel"
        ], color=(0, 200, 255))
    elif mode == MODE_CAPTURING:
        draw_overlay(frame, [
            f"Capturing [{gesture_name}] {captured}/{TOTAL_DATAPOINTS}",
            "Press 'c' or Esc to cancel"
        ], color=(0, 0, 255))

    cv2.imshow('Output', frame)


cap.release()
cv2.destroyAllWindows()

if gesture_data:
    handpoints_with_label = handpoints + ['gesture_name']
    df = pd.DataFrame(gesture_data, columns=handpoints_with_label)
    if os.path.exists(GESTURES_PATH):
        df = pd.concat([pd.read_csv(GESTURES_PATH), df], ignore_index=True)
    df.to_csv(GESTURES_PATH, index=False)
    print(f"Saved {len(gesture_data)} new samples to {GESTURES_PATH}")
else:
    print("No gestures captured.")
