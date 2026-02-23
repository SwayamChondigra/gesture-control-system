import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
import screen_brightness_control as sbc

# ================= PERFORMANCE =================
pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()
SMOOTHING = 10
FRAME_DELAY = 0.008

# ================= CAMERA ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ================= MEDIAPIPE ===================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ================= STATE =======================
prev_x, prev_y = 0, 0
last_click = 0
dragging = False
last_vol_time = 0
last_bright_time = 0

# ================= HELPERS =====================
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def smooth(prev, curr, factor):
    return prev + (curr - prev) / factor

# ================= MAIN LOOP ===================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # ---------- MOUSE MOVE ----------
        x = int(lm[8].x * SCREEN_W)
        y = int(lm[8].y * SCREEN_H)

        curr_x = smooth(prev_x, x, SMOOTHING)
        curr_y = smooth(prev_y, y, SMOOTHING)
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        pinch_index = distance(lm[4], lm[8])
        pinch_middle = distance(lm[4], lm[12])

        # ---------- LEFT CLICK ----------
        if pinch_index < 0.035 and time.time() - last_click > 0.6:
            pyautogui.click()
            last_click = time.time()

        # ---------- DRAG ----------
        if pinch_index < 0.025 and not dragging:
            pyautogui.mouseDown()
            dragging = True
        if pinch_index > 0.05 and dragging:
            pyautogui.mouseUp()
            dragging = False

        # ---------- SCROLL ----------
        if lm[8].y < lm[6].y and lm[12].y < lm[10].y:
            diff = lm[12].y - lm[8].y
            pyautogui.scroll(int(-diff * 800))

        # ---------- VOLUME CONTROL (EXE SAFE) ----------
        if pinch_index < 0.04 and pinch_middle > 0.08:
            if time.time() - last_vol_time > 0.15:
                if lm[8].y < 0.45:
                    pyautogui.press("volumeup")
                elif lm[8].y > 0.55:
                    pyautogui.press("volumedown")
                last_vol_time = time.time()

        # ---------- BRIGHTNESS CONTROL ----------
        if pinch_middle < 0.04 and pinch_index > 0.08:
            if time.time() - last_bright_time > 0.2:
                try:
                    bright = int(np.interp(lm[12].y, [0.2, 0.8], [100, 0]))
                    sbc.set_brightness(bright)
                except:
                    pass
                last_bright_time = time.time()

    cv2.imshow("Gesture Control Pro", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(FRAME_DELAY)

cap.release()
cv2.destroyAllWindows()
