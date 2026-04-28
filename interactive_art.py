import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

# --- Apple Silicon (M3) Optimization ---
cv2.setUseOptimized(True)

# --- Configuration & Constants ---
IMAGE_PATH = "/Users/furkandumanoglu/Desktop/PEA/babylon.jpg"
PX_PER_CM = 40 

PASCHAL_COORD = (1608, 1755)
GLOBAL_REVEAL_COORD = (1129, 2012)

PASCHAL_RADIUS = 5 * PX_PER_CM  
SMALL_TORCH_RADIUS = 7 * PX_PER_CM 
BIG_TORCH_RADIUS = 12 * PX_PER_CM   # "1" parmağı için daha büyük ışık
MIRACLE_RADIUS = 1 * PX_PER_CM 
MIRACLE_TRIGGER_DIST = 10 * PX_PER_CM 
REVEAL_TRIGGER_DIST = 5 * PX_PER_CM

MIRACLE_COORDS = [
    (1291, 2994), (1241, 2833), (261, 3283), (409, 2647), (1545, 3337),
    (1463, 3640), (1868, 3486), (2282, 2481), (1762, 462), (1170, 863),
    (1178, 1255), (712, 1368), (651, 597), (551, 438), (407, 603)
]

QUADRANT_TEXTS = {
    1: "welcome to heaven",
    2: "Darkness",
    3: "Everyone needs him",
    4: "Even kids"
}

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# --- Load Painting ---
painting = cv2.imread(IMAGE_PATH)
if painting is None:
    print(f"❌ Error: Could not load image at {IMAGE_PATH}")
    exit()

H, W, _ = painting.shape

# --- State Variables ---
active_text = None
text_start_time = 0
text_fade_duration = 2.0
text_display_duration = 10.0

def get_gesture(hand_landmarks):
    """
    Parmak durumunu kontrol eder. 
    'one': Sadece işaret parmağı açık.
    'open': Tüm el açık.
    """
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    status = []
    for t, p in zip(tips, pips):
        status.append(hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y)
    
    thumb_up = abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[3].x) > 0.05
    if status[0] and not any(status[1:]) and not thumb_up:
        return 'one'
    if all(status[:3]):
        return 'open'
    return 'unknown'

def draw_lightning(mask, center, radius):
    """
    Şimşek/Yıldırım efekti oluşturur.
    """
    for _ in range(12):
        end_x = center[0] + random.randint(-radius*2, radius*2)
        end_y = center[1] + random.randint(-radius*2, radius*2)
        cv2.line(mask, center, (end_x, end_y), 255, random.randint(2, 8))
    cv2.circle(mask, center, int(radius * random.uniform(0.8, 1.5)), 255, -1)

def draw_golden_text(img, text, alpha):
    """
    Ekrana altın sarısı, yumuşak geçişli yazı yazar.
    """
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_TRIPLEX
    scale = 4
    thickness = 8
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (W - text_size[0]) // 2
    text_y = (H + text_size[1]) // 2
    
    cv2.putText(overlay, text, (text_x+4, text_y+4), font, scale, (0, 100, 150), thickness + 4) 
    cv2.putText(overlay, text, (text_x, text_y), font, scale, (0, 215, 255), thickness) 
    cv2.putText(overlay, text, (text_x, text_y), font, scale, (150, 255, 255), thickness // 3)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def apply_masking(base_img, torch_pos, torch_radius, miracle_active, reveal_active):
    """
    Karanlık maskesini oluşturur ve efektleri uygular.
    """
    if reveal_active:
        return base_img.copy()

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, PASCHAL_COORD, PASCHAL_RADIUS, 255, -1)

    if torch_pos:
        cv2.circle(mask, torch_pos, torch_radius, 255, -1)

    if miracle_active:
        for coord in MIRACLE_COORDS:
            draw_lightning(mask, coord, MIRACLE_RADIUS)

    mask_blurred = cv2.GaussianBlur(mask, (151, 151), 0)
    mask_3ch = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0
    result = (base_img * mask_3ch).astype(np.uint8)
    return result

# --- Main Loop ---
cap = cv2.VideoCapture(0)

print("🚀 Installation Ready. Master Reveal fix applied.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    torch_pos = None
    torch_radius = SMALL_TORCH_RADIUS
    miracle_active = False
    global_reveal_active = False
    current_quadrant = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tx = int(index_tip.x * W)
            ty = int(index_tip.y * H)
            torch_pos = (tx, ty)

            gesture = get_gesture(hand_landmarks)
            if gesture == 'one':
                torch_radius = BIG_TORCH_RADIUS
                if tx < W // 2:
                    current_quadrant = 1 if ty < H // 2 else 3
                else:
                    current_quadrant = 2 if ty < H // 2 else 4
                
                if current_quadrant and (active_text is None or time.time() - text_start_time > text_display_duration):
                    active_text = QUADRANT_TEXTS[current_quadrant]
                    text_start_time = time.time()
            else:
                torch_radius = SMALL_TORCH_RADIUS

            # Reveal Logic (Updated: Only active while hand is close)
            dist_reveal = math.sqrt((tx - GLOBAL_REVEAL_COORD[0])**2 + (ty - GLOBAL_REVEAL_COORD[1])**2)
            if dist_reveal < REVEAL_TRIGGER_DIST:
                global_reveal_active = True
            
            # Miracle Logic
            dist_paschal = math.sqrt((tx - PASCHAL_COORD[0])**2 + (ty - PASCHAL_COORD[1])**2)
            if dist_paschal < MIRACLE_TRIGGER_DIST:
                miracle_active = True

    final_output = apply_masking(painting, torch_pos, torch_radius, miracle_active, global_reveal_active)

    if active_text:
        elapsed = time.time() - text_start_time
        if elapsed < text_display_duration:
            alpha = min(1.0, elapsed / text_fade_duration)
            draw_golden_text(final_output, active_text, alpha)
        else:
            active_text = None

    preview = cv2.resize(final_output, (W // 4, H // 4))
    cv2.imshow("Interactive Art Installation", preview)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
