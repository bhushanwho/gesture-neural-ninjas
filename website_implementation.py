import streamlit as st
import requests
import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Streamlit UI
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'>Gesture-Based AI Kiosk</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📹 Live Video Feed")
    enable_camera = st.checkbox("Enable Camera")
    stframe = st.empty()

def detect_gesture(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    gesture = ""
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            
            if landmarks[4, 0] < landmarks[3, 0]:
                gesture = "Swipe Left"
            elif landmarks[4, 0] > landmarks[3, 0]:
                gesture = "Swipe Right"
            elif landmarks[8, 1] < landmarks[6, 1]:
                gesture = "Swipe Up"
            elif landmarks[8, 1] > landmarks[6, 1]:
                gesture = "Swipe Down"
            elif np.linalg.norm(landmarks[4] - landmarks[8]) < 0.05:
                gesture = "Select Item"
    
    return gesture, frame

cap = cv2.VideoCapture(0) if enable_camera else None

if cap:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gesture, frame = detect_gesture(frame)
        
        if gesture:
            requests.post("http://127.0.0.1:5000/gesture", json={"gesture": gesture})
            time.sleep(0.5)
        
        stframe.image(frame, channels="BGR")
        st.write(f"Detected Gesture: {gesture}")
    
    cap.release()

with col2:
    st.subheader("🍔 Kiosk-Like Webpage Menu")
    st.components.v1.iframe("http://127.0.0.1:5000", height=600, scrolling=True)