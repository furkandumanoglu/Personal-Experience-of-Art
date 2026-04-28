import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def map_coordinates(input_x, input_y, cam_w, cam_h, paint_w, paint_h):
    # Scale camera coordinates to painting dimensions
    # Kamera koordinatlarını tablo boyutlarına ölçeklendirme
    norm_x = int(input_x * paint_w / cam_w)
    norm_y = int(input_y * paint_h / cam_h)
    return norm_x, norm_y