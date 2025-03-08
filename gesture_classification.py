import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

BUFFER_SIZE = 7  # Track motion over last N frames
INSTANCE_BUFFER_SIZE = 10  # Number of instances to establish gesture
SWIPE_THRESHOLD = 0.15  # Movement threshold for swipes
SELECT_THRESHOLD = 0.03  # Minimal movement for select

FRAME_BUFFER = deque(maxlen=BUFFER_SIZE)
INSTANCE_BUFFER = deque(maxlen=INSTANCE_BUFFER_SIZE)

def classify_gesture():
    counts = {"Swipe Left": 0, "Swipe Right": 0, "Swipe Up": 0, "Swipe Down": 0, "Select": 0, "No Operation": 0}
    for g in INSTANCE_BUFFER:
        counts[g] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "No Operation"

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gesture = "No Operation"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                FRAME_BUFFER.append((index_tip.x, index_tip.y))
                
                if len(FRAME_BUFFER) >= 2:
                    dx = FRAME_BUFFER[-1][0] - FRAME_BUFFER[0][0]
                    dy = FRAME_BUFFER[-1][1] - FRAME_BUFFER[0][1]
                    
                    if abs(dx) > SWIPE_THRESHOLD:
                        gesture = "Swipe Right" if dx < 0 else "Swipe Left"
                    elif abs(dy) > SWIPE_THRESHOLD:
                        gesture = "Swipe Down" if dy < 0 else "Swipe Up"
                    elif thumb_tip.y < index_tip.y:  # Thumbs up detected
                        gesture = "Select"
                
                INSTANCE_BUFFER.append(gesture)
        else:
            INSTANCE_BUFFER.append("No Operation")

        final_gesture = classify_gesture()
        cv2.putText(image, f"Gesture: {final_gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Gesture Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
