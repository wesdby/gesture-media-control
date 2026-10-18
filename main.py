import cv2
import mediapipe as mp
import numpy as np
import math
import time
import subprocess
import threading
from collections import deque
import rumps

# Configuration

# Volume control (circle motion)
MIN_RADIUS = 35          # Min radius in pixels for rotation detection
VOLUME_SENSITIVITY = 0.2 # Volume change speed
VOLUME_PUSH_INTERVAL = 0.05 # Min delay between volume updates (sec)
MAX_TWIST_DELTA_DEG = 25 # Max angle change per frame (prevents spikes)
TWIST_TRAIL_LEN = 20     # History length for rotation tracking
TWIST_SMOOTH_ALPHA = 0.4 # Finger smoothing factor (0-1)

# Track navigation & play/pause (palm swipes)
SWIPE_MIN_DX = 120       # Min horizontal distance (px) for track switch
SWIPE_MIN_DY = 100       # Min vertical distance (px) for play/pause
SWIPE_WINDOW = 0.35      # Time window for swipe (sec)
SWIPE_COOLDOWN = 2.5     # Cooldown between swipes (sec)
SWIPE_LEFT_COOLDOWN = 1.0 # Cooldown for previous track swipe

# Timeline seeking (pinch)
PINCH_ENTER_THRESHOLD = 45 # Distance to start pinching (px)
PINCH_EXIT_THRESHOLD = 60  # Distance to stop pinching (px)
SEEK_SENSITIVITY = 0.4     # Seeking speed multiplier
SEEK_PUSH_INTERVAL = 0.12  # Min delay between seek updates (sec)
MID_X_MAX_JUMP = 90        # Max jump filter for pinch center (px)
SEEK_HARD_CLAMP_SEC = 600  # Max seek limit in seconds

# Tracking & states
FIST_HOLD_TIME = 1.0       # Hold time (sec) to toggle tracking on/off
HAND_SCORE_THRESHOLD = 0.9 # Min detection confidence (0-1)
MAX_JUMP_PX = 130          # Max wrist jump filter (px)
GESTURE_STABLE_FRAMES = 4  # Frames needed to confirm a gesture

# Camera
CAMERA_FAIL_LIMIT = 15     # Failed reads before reconnecting camera
CAMERA_REOPEN_COOLDOWN = 1.0 # Reconnect cooldown (sec)


# ONE EURO FILTER MATHEMATICS
class LowPassFilter:
    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.y = self.s = None

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha should be in (0.0, 1.0]")
        self.alpha = alpha

    def __call__(self, value, alpha=None):
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.y is None:
            s = value
        else:
            s = self.alpha * value + (1.0 - self.alpha) * self.y
        self.y = value
        self.s = s
        return s

class OneEuroFilter:
    def __init__(self, t0, x0, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = x0
        self.t_prev = t0  
        self.x_filter = LowPassFilter(self._alpha(t0, mincutoff))
        self.dx_filter = LowPassFilter(self._alpha(t0, dcutoff))

    def _alpha(self, t, cutoff):
        te = t - self.t_prev
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(te, 1e-5))

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            t_e = 1e-5
        dx = (x - self.x_prev) / t_e
        edx = self.dx_filter(dx, self._alpha(t_e, self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(edx)
        filtered_x = self.x_filter(x, self._alpha(t_e, cutoff))
        self.x_prev = filtered_x
        self.t_prev = t
        return filtered_x

class OneEuroFilter2D:
    def __init__(self, mincutoff=1.2, beta=0.01, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.fx = None
        self.fy = None

    def __call__(self, t, pt):
        if pt is None:
            return None
        if self.fx is None:
            self.fx = OneEuroFilter(t, pt[0], mincutoff=self.mincutoff, beta=self.beta, dcutoff=self.dcutoff)
            self.fy = OneEuroFilter(t, pt[1], mincutoff=self.mincutoff, beta=self.beta, dcutoff=self.dcutoff)
            return pt
        return (self.fx(t, pt[0]), self.fy(t, pt[1]))

# SYSTEM COMMANDS (macOS)
def run_async(cmd):
    threading.Thread(target=lambda: subprocess.run(cmd, capture_output=True), daemon=True).start()

def run_sync(cmd, default=""):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        return r.stdout.strip()
    except Exception:
        return default

def get_system_volume():
    out = run_sync(["osascript", "-e", "output volume of (get volume settings)"], "50")
    try:
        return int(out)
    except Exception:
        return 50

def set_system_volume(level):
    level = int(np.clip(level, 0, 100))
    run_async(["osascript", "-e", f"set volume output volume {level}"])

def _player_guard(body):
    return f'''
    if application "Spotify" is running then
        tell application "Spotify" to {body}
    end if
    if application "Music" is running then
        tell application "Music" to {body}
    end if
    '''

def next_track():
    run_async(["osascript", "-e", _player_guard("next track")])

def prev_track():
    run_async(["osascript", "-e", _player_guard("previous track")])

def playpause():
    run_async(["osascript", "-e", _player_guard("playpause")])

def get_player_position_once():
    script = '''
    if application "Spotify" is running then
        return player position of application "Spotify"
    else if application "Music" is running then
        return player position of application "Music"
    else
        return 0
    end if
    '''
    out = run_sync(["osascript", "-e", script], "0")
    try:
        return float(out)
    except Exception:
        return 0.0

def set_player_position(pos):
    pos = max(0.0, pos)
    run_async(["osascript", "-e", _player_guard(f"set player position to {pos}")])

def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def enhance_image(image):
    # Bilateral Filter removes skin noise without blurring edges
    bilateral = cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
    # CLAHE locally equalizes contrast and lifts shadows
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def fingers_up(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    fingers = []
    if handedness_label == 'Right':
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(lm[tip].y < lm[pip].y)
    return fingers

def classify_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers
    if not any([index, middle, ring, pinky]):
        return 'fist'
    if index and not middle and not ring and not pinky:
        return 'point'
    if all(fingers):
        return 'palm'
    return None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GestureEngine:
    def __init__(self):
        self.running = True
        self.tracking_enabled = True
        self.preview_enabled = False
        self.cap = None
        self.last_camera_open = 0.0
        self.latest_frame = None

        self.volume_level = get_system_volume()
        self.last_volume_push = 0.0
        self.last_pushed_volume = self.volume_level

        self.twist_trail = deque(maxlen=TWIST_TRAIL_LEN)
        self.prev_angle = None
        self.smoothed_index_tip = None

        self.palm_trail = deque(maxlen=15)
        self.last_swipe_time = 0.0
        self.last_left_swipe_time = 0.0
        self.status_text = "starting"

        self.known_position = get_player_position_once()
        self.last_known_update = time.time()
        self.pinch_start_x = None
        self.pinch_start_pos = 0.0
        self.pinch_active_prev = False
        self.prev_mid_x = None
        self.last_seek_push = 0.0
        self.last_pushed_seek_target = None
        self.seek_preview_sec = 0.0

        self.fist_hold_start = None
        self.fist_triggered = False

        self.wrist_filter = OneEuroFilter2D(mincutoff=1.5, beta=0.02)
        self.smoothed_wrist = None
        self.pending_gesture = None
        self.pending_count = 0
        self.stable_gesture = None

        self._open_camera()

    def _open_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(0)
        self.last_camera_open = time.time()

    def toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled

    def stop(self):
        self.running = False

    def run(self):
        consecutive_failures = 0
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                            min_detection_confidence=0.85, min_tracking_confidence=0.7) as hands:
            while self.running:
                if self.cap is None or not self.cap.isOpened():
                    if time.time() - self.last_camera_open > CAMERA_REOPEN_COOLDOWN:
                        self.status_text = "reconnecting camera..."
                        self._open_camera()
                    time.sleep(0.2)
                    continue

                success, image = self.cap.read()
                if not success or image is None:
                    consecutive_failures += 1
                    if consecutive_failures > CAMERA_FAIL_LIMIT and \
                       time.time() - self.last_camera_open > CAMERA_REOPEN_COOLDOWN:
                        self.status_text = "camera stuck, reconnecting..."
                        self._open_camera()
                        consecutive_failures = 0
                    time.sleep(0.03)
                    continue
                consecutive_failures = 0

                try:
                    self._process_frame(image, hands)
                except Exception as e:
                    print(f"[GestureEngine] Frame processing error: {e}")

    def _process_frame(self, image, hands):
        image = cv2.flip(image, 1)
        processed_image = enhance_image(image)
        h, w, _ = processed_image.shape

        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        now = time.time()
        dt = now - self.last_known_update
        self.last_known_update = now
        if self.stable_gesture != 'pinch':
            self.known_position += dt

        raw_gesture = None
        valid_frame = False
        hand_lm = None

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_lm = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0]
            label, score = handedness.label, handedness.score

            raw_wrist = (hand_lm.landmark[0].x * w, hand_lm.landmark[0].y * h)
            
            # Adaptive wrist filtering via One Euro Filter
            self.smoothed_wrist = self.wrist_filter(now, raw_wrist)

            jump_ok = True
            if self.smoothed_wrist is not None:
                pass

            if score >= HAND_SCORE_THRESHOLD:
                valid_frame = True
                thumb_tip = (hand_lm.landmark[4].x * w, hand_lm.landmark[4].y * h)
                index_tip = (hand_lm.landmark[8].x * w, hand_lm.landmark[8].y * h)
                pinch_dist = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
                fingers = fingers_up(hand_lm, label)

                # Hysteresis logic for pinch
                if not self.pinch_active_prev:
                    is_pinching = pinch_dist < PINCH_ENTER_THRESHOLD
                else:
                    is_pinching = pinch_dist < PINCH_EXIT_THRESHOLD

                if is_pinching:
                    raw_gesture = 'pinch'
                else:
                    raw_gesture = classify_gesture(fingers)

        if not valid_frame:
            raw_gesture = None

        if raw_gesture == self.pending_gesture:
            self.pending_count += 1
        else:
            self.pending_gesture = raw_gesture
            self.pending_count = 1
        
        if self.pending_count >= GESTURE_STABLE_FRAMES:
            self.stable_gesture = self.pending_gesture
        elif raw_gesture is None:
            self.stable_gesture = None

        # Gesture Logic (fist, pinch, swipes)
        
        # FIST
        if self.stable_gesture == 'fist':
            if self.fist_hold_start is None:
                self.fist_hold_start = now
            elif not self.fist_triggered and now - self.fist_hold_start >= FIST_HOLD_TIME:
                self.tracking_enabled = not self.tracking_enabled
                self.fist_triggered = True
        else:
            self.fist_hold_start = None
            self.fist_triggered = False

        # PINCH
        entering_pinch = (self.stable_gesture == 'pinch' and not self.pinch_active_prev)
        leaving_pinch = (self.stable_gesture != 'pinch' and self.pinch_active_prev)

        if leaving_pinch:
            self.prev_mid_x = None

        if entering_pinch:
            self.pinch_start_x = None

        if self.tracking_enabled and self.stable_gesture == 'pinch' and valid_frame:
            mid_x = (hand_lm.landmark[4].x * w + hand_lm.landmark[8].x * w) / 2
            if self.prev_mid_x is not None and abs(mid_x - self.prev_mid_x) > MID_X_MAX_JUMP:
                mid_x = self.prev_mid_x
            self.prev_mid_x = mid_x

            if self.pinch_start_x is None:
                self.pinch_start_x = mid_x
                self.known_position = get_player_position_once()
                self.pinch_start_pos = self.known_position

            delta_px = mid_x - self.pinch_start_x
            self.seek_preview_sec = float(np.clip(delta_px * SEEK_SENSITIVITY, -SEEK_HARD_CLAMP_SEC, SEEK_HARD_CLAMP_SEC))
            target_pos = max(0.0, self.pinch_start_pos + self.seek_preview_sec)

            if now - self.last_seek_push > SEEK_PUSH_INTERVAL and \
               (self.last_pushed_seek_target is None or abs(target_pos - self.last_pushed_seek_target) > 0.3):
                set_player_position(target_pos)
                self.last_pushed_seek_target = target_pos
                self.last_seek_push = now
                self.known_position = target_pos

        self.pinch_active_prev = (self.stable_gesture == 'pinch')

        # TWIST (Volume)
        if self.tracking_enabled and self.stable_gesture == 'point' and valid_frame:
            index_tip = (hand_lm.landmark[8].x * w, hand_lm.landmark[8].y * h)
            self.smoothed_index_tip = index_tip if self.smoothed_index_tip is None else (
                self.smoothed_index_tip[0] * (1 - TWIST_SMOOTH_ALPHA) + index_tip[0] * TWIST_SMOOTH_ALPHA,
                self.smoothed_index_tip[1] * (1 - TWIST_SMOOTH_ALPHA) + index_tip[1] * TWIST_SMOOTH_ALPHA,
            )
            self.twist_trail.append(self.smoothed_index_tip)

            if len(self.twist_trail) >= 6:
                pivot_x = sum(p[0] for p in self.twist_trail) / len(self.twist_trail)
                pivot_y = sum(p[1] for p in self.twist_trail) / len(self.twist_trail)
                dx = self.smoothed_index_tip[0] - pivot_x
                dy = self.smoothed_index_tip[1] - pivot_y
                radius = math.hypot(dx, dy)

                if radius > MIN_RADIUS:
                    angle = math.atan2(dy, dx)
                    if self.prev_angle is not None:
                        delta_deg = math.degrees(normalize_angle(angle - self.prev_angle))
                        delta_deg = float(np.clip(delta_deg, -MAX_TWIST_DELTA_DEG, MAX_TWIST_DELTA_DEG))
                        self.volume_level = float(np.clip(self.volume_level + delta_deg * VOLUME_SENSITIVITY, 0, 100))
                    self.prev_angle = angle
                else:
                    self.prev_angle = None
        else:
            self.prev_angle = None
            self.twist_trail.clear()
            self.smoothed_index_tip = None

        # SWIPES
        if self.tracking_enabled and self.stable_gesture == 'palm' and valid_frame and self.smoothed_wrist is not None:
            self.palm_trail.append((now, self.smoothed_wrist[0], self.smoothed_wrist[1]))
        else:
            self.palm_trail.clear()

        if self.tracking_enabled and len(self.palm_trail) >= 2:
            recent = [p for p in self.palm_trail if now - p[0] <= SWIPE_WINDOW]
            if len(recent) >= 2:
                dx = recent[-1][1] - recent[0][1]
                dy = recent[-1][2] - recent[0][2]
                if abs(dx) > SWIPE_MIN_DX and abs(dx) > abs(dy) * 1.5:
                    if dx > 0:
                        if now - self.last_swipe_time > SWIPE_COOLDOWN:
                            next_track()
                            self.last_swipe_time = now
                            self.palm_trail.clear()
                    else:
                        if now - self.last_left_swipe_time > SWIPE_LEFT_COOLDOWN:
                            prev_track()
                            self.last_left_swipe_time = now
                            self.palm_trail.clear()
                elif dy > SWIPE_MIN_DY and abs(dy) > abs(dx) * 1.5:
                    if now - self.last_swipe_time > SWIPE_COOLDOWN:
                        playpause()
                        self.last_swipe_time = now
                        self.palm_trail.clear()

        if self.tracking_enabled and now - self.last_volume_push > VOLUME_PUSH_INTERVAL and \
           abs(self.volume_level - self.last_pushed_volume) > 0.5:
            set_system_volume(self.volume_level)
            self.last_pushed_volume = self.volume_level
            self.last_volume_push = now

        self.status_text = self.stable_gesture or "-"

        if self.preview_enabled:
            display_image = processed_image.copy()
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(display_image, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            status_color = (0, 255, 0) if self.tracking_enabled else (0, 0, 255)
            cv2.putText(display_image, f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
            cv2.putText(display_image, f"Gesture: {self.status_text}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_image, f"Vol: {int(self.volume_level)}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            self.latest_frame = display_image
        else:
            self.latest_frame = None


class MenuBarApp(rumps.App):
    def __init__(self, engine):
        super().__init__("🖐", quit_button=None)
        self.engine = engine

        self.tracking_item = rumps.MenuItem("Tracking: ON", callback=self.toggle_tracking)
        self.preview_item = rumps.MenuItem("Show Preview", callback=self.toggle_preview)
        self.status_item = rumps.MenuItem("Gesture: -")
        self.status_item.set_callback(None)

        self.menu = [
            self.tracking_item,
            self.preview_item,
            None,
            self.status_item,
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

        self.timer = rumps.Timer(self.refresh, 1)
        self.timer.start()

        self.preview_timer = rumps.Timer(self.update_preview, 0.03)
        self.preview_timer.start()

    def toggle_tracking(self, sender):
        self.engine.toggle_tracking()
        self.tracking_item.title = f"Tracking: {'ON' if self.engine.tracking_enabled else 'OFF'}"

    def toggle_preview(self, sender):
        self.engine.preview_enabled = not self.engine.preview_enabled
        sender.state = self.engine.preview_enabled
        if not self.engine.preview_enabled:
            try:
                cv2.destroyWindow("Gesture Control Preview")
            except Exception:
                pass

    def update_preview(self, _):
        if self.engine.preview_enabled and self.engine.latest_frame is not None:
            cv2.imshow("Gesture Control Preview", self.engine.latest_frame)
            cv2.waitKey(1)
        else:
            try:
                cv2.destroyWindow("Gesture Control Preview")
            except Exception:
                pass

    def refresh(self, _):
        self.title = "🖐" if self.engine.tracking_enabled else "🖐💤"
        self.status_item.title = f"Gesture: {self.engine.status_text} | Vol: {int(self.engine.volume_level)}"

    def quit_app(self, _):
        self.engine.stop()
        self.preview_timer.stop()
        time.sleep(0.3)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        rumps.quit_application()


if __name__ == "__main__":
    engine = GestureEngine()
    worker = threading.Thread(target=engine.run, daemon=True)
    worker.start()
    MenuBarApp(engine).run()
