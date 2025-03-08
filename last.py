import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import google.generativeai as genai
import os


# Configure Gemini AI
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Constants
BUFFER_SIZE = 7
INSTANCE_BUFFER_SIZE = 10
SWIPE_THRESHOLD = 0.15
SELECT_THRESHOLD = 0.03

# Buffers
frame_buffer = deque(maxlen=BUFFER_SIZE)
instance_buffer = deque(maxlen=INSTANCE_BUFFER_SIZE)

def classify_gesture():
    counts = {"Swipe Left": 0, "Swipe Right": 0, "Swipe Up": 0, "Swipe Down": 0, "Select": 0, "No Operation": 0}
    for g in instance_buffer:
        counts[g] += 1
    return max(counts, key=counts.get) if any(counts.values()) else "No Operation"

def process_gesture(gesture):
    # Process gesture with Gemini AI
    prompt = f"Given the gesture '{gesture}', suggest an appropriate action for a web interface."
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("Gesture Recognition Interface")
    
    # Sidebar controls
    st.sidebar.title("Controls")
    start_camera = st.sidebar.button("Start Camera")
    stop_camera = st.sidebar.button("Stop Camera")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Camera Feed")
        camera_placeholder = st.empty()
    
    with col2:
        st.subheader("Gesture Information")
        gesture_placeholder = st.empty()
        action_placeholder = st.empty()
    
    if start_camera:
        cap = cv2.VideoCapture(0)
        
        with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                
                gesture = "No Operation"
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                        
                        index_tip = hand_landmarks.landmark[8]
                        thumb_tip = hand_landmarks.landmark[4]
                        frame_buffer.append((index_tip.x, index_tip.y))
                        
                        if len(frame_buffer) >= 2:
                            dx = frame_buffer[-1][0] - frame_buffer[0][0]
                            dy = frame_buffer[-1][1] - frame_buffer[0][1]
                            
                            if abs(dx) > SWIPE_THRESHOLD:
                                gesture = "Swipe Right" if dx > 0 else "Swipe Left"
                            elif abs(dy) > SWIPE_THRESHOLD:
                                gesture = "Swipe Up" if dy < 0 else "Swipe Down"
                            elif thumb_tip.y < index_tip.y:
                                gesture = "Select"
                        
                        instance_buffer.append(gesture)
                else:
                    instance_buffer.append("No Operation")
                
                final_gesture = classify_gesture()
                
                # Update UI
                camera_placeholder.image(image)
                gesture_placeholder.write(f"Detected Gesture: {final_gesture}")
                
                if final_gesture != "No Operation":
                    suggested_action = process_gesture(final_gesture)
                    action_placeholder.write(f"Suggested Action: {suggested_action}")
                
                if stop_camera:
                    break
        
        cap.release()

if __name__ == "__main__":
    main()