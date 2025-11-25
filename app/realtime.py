import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ================================
# CONFIG
# ================================

# Camera index:
# - 0 = default laptop cam (if you had one)
# - 1,2,... = external / virtual cams like Camo
CAM_INDEX = 0           # <- change if needed (use your working index here)
USE_DSHOW = True        # True is better for Windows sometimes

# Preview scaling (so the window isn't gigantic)
PREVIEW_SCALE = 1.2     # 1.0 = original size, <1.0 = smaller window

# Model input size (must match training)
IMG_SIZE = 64

# Paths
MODEL_PATH = "expression_model.h5"
LABELS_PATH = "expressions_labels.txt"
CASCADE_PATH = "haarcascade_frontalface_default.xml"


# ================================
# LOAD MODEL & LABELS
# ================================

print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

print("[INFO] Loading class labels...")
expressions = []
with open(LABELS_PATH, "r") as f:
    for line in f:
        expr = line.strip()
        if expr:
            expressions.append(expr)

print("[INFO] Loaded expressions:", expressions)

num_classes = len(expressions)
if num_classes == 0:
    print("[ERROR] No class labels found in expressions_labels.txt")
    exit()


# ================================
# LOAD FACE DETECTOR
# ================================

print("[INFO] Loading Haar Cascade face detector...")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("[ERROR] Could not load Haar Cascade. Check path:", CASCADE_PATH)
    exit()


# ================================
# OPEN CAMERA
# ================================

print(f"[INFO] Opening camera at index {CAM_INDEX}...")
if USE_DSHOW:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)  # better for Windows
else:
    cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print(f"[ERROR] Could not open webcam at index {CAM_INDEX}.")
    exit()

# Read one frame to report its shape
ret, frame = cap.read()
if not ret:
    print("[ERROR] Could not read initial frame from camera.")
    cap.release()
    exit()

print("[INFO] Initial frame shape (H, W, C):", frame.shape)


# ================================
# MAIN LOOP
# ================================

print("[INFO] Starting real-time expression recognition. Press 'q' to quit.")

while True:
    # 1) Grab a frame
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Failed to grab frame.")
        break

    # Optional: scale preview down so it doesn't look huge
    if PREVIEW_SCALE != 1.0:
        frame = cv2.resize(
            frame,
            None,
            fx=PREVIEW_SCALE,
            fy=PREVIEW_SCALE,
            interpolation=cv2.INTER_AREA
        )

    # 2) Convert to grayscale for detection & model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3) Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # 4) For each face: crop, preprocess, predict
    for (x, y, w, h) in faces:
        # Crop face region from grayscale frame
        face_roi = gray[y:y+h, x:x+w]

        # Resize to model input size
        face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))

        # Normalize & reshape for model: (1, H, W, 1)
        face_input = face_roi.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

        # Predict expression
        preds = model.predict(face_input, verbose=0)[0]  # e.g. [0.1, 0.7, 0.2, ...]
        label_idx = np.argmax(preds)
        confidence = float(preds[label_idx])

        if label_idx < len(expressions):
            label = expressions[label_idx]
        else:
            label = "Unknown"

        # 5) Draw rectangle & label
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2
        )

        text = f"{label} ({confidence * 100:.1f}%)"
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # 6) Show the result
    cv2.imshow("Real-time Expression Recognition", frame)

    # 7) Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] 'q' pressed. Exiting.")
        break


# ================================
# CLEANUP
# ================================

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera released and windows closed.")
