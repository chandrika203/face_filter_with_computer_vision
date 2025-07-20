# Project: Face Filter
# pip install mediapipe #face recognization.venv\Scripts\activate
# pip install opencv-python #webcam access

# Libraries
import cv2
import mediapipe as mp
import numpy as np

# Load the overlay images
mustache_png = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
hat_png = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)

# check if 4 channel  (BGRA) image is loaded or not
for name, img in (("mustache.png", mustache_png ), ("hat.png", hat_png)):
    if img is None or img.shape[2] < 4: 
        raise FileNotFoundError("Missing or Invalid file")

# intialize MediaPipe FaceMesh for landmark detection
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    refine_landmarks = False,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# function to put the overlay on video frame
def overlay_rgba(background, overlay, x, y, w, h):
    overlay = cv2.resize(overlay, (w,h), interpolation=cv2.INTER_AREA)
    b, g, r, a = cv2.split(overlay)

    alpha = a.astype(float)/255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    h_bg, w_bg = background.shape[:2]

    x0, y0 = max(0,x), max(0,y)  # top left corner
    x1, y1 = min(x + w, w_bg), min(y+h, h_bg) # bottom right corner

    overlay_slice = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    background_roi = (slice(y0, y1), slice(x0, x1))

    # Extract ROI (Region of Interest) for overlay and background
    foreground = cv2.merge([b,g,r])[overlay_slice] # this will extract only visible section
    alpha_roi = alpha[overlay_slice] 
    bg_roi = background[background_roi]

    blended = cv2.convertScaleAbs(foreground * alpha_roi + bg_roi * (1 - alpha_roi))
    background[background_roi] = blended
    return background


# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# Main driver code
while True:
    ok , frame = cap.read()
    if not ok:
        print("Empty frame received")
        break

    # Flip frame like a selfie
    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]

    # convert frame to RGB from BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # function to convert landmarks into pixels
        def to_px(idx):
            pt = landmarks[idx]
            return int(pt.x * w_frame), int(pt.y * h_frame)
        
        # finding the place to put mustache
        lib_x1, lib_y1 = to_px(13)  # upper lip center
        lib_x2, lib_y2 = to_px(14) # lower lip center

        # average central lip position
        lib_x = (lib_x1 + lib_x2) // 2
        lib_y = (lib_y1 + lib_y2) // 2

        # finding the place to put hat
        left_temple_x, _ = to_px(127)
        right_temple_x, _ = to_px(356)
        forehead_x, forehead_y = to_px(10)

        face_w = right_temple_x - left_temple_x
        # Add Mushache
        must_w = face_w
        mush_h = int(must_w * 0.3)
        must_x = lib_x - must_w // 2
        must_y = lib_y - int(mush_h * 0.75)
        frame = overlay_rgba(frame, mustache_png,must_x, must_y,must_w, mush_h )

        #  Add Hat
        hat_w = int(face_w * 1.6)
        hat_h = int(hat_w * 0.9)
        hat_x = forehead_x - hat_w // 2
        hat_y = forehead_y - int(hat_h * 0.8)
        frame = overlay_rgba(frame, hat_png, hat_x, hat_y, hat_w, hat_h )
    cv2.imshow("Mustache and Hat", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()