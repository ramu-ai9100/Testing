
import os, cv2, time, numpy as np
from datetime import datetime
from typing import Tuple, List, Optional
import onnxruntime as ort
import requests
from itertools import product
from math import ceil
from collections import Counter
import warnings

# Suppress JPEG warnings
warnings.filterwarnings('ignore')
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
cv2.setLogLevel(0)

# =========================
# CONFIGURATION
# =========================
FACE_DB_NPZ         = "faces/mobile2/ramu.npz"          # ← NPZ file with embeddings + labels
RETINAFACE_MODEL    = "models/RetinaFace_mobile0.25_640_opset18.onnx"
MOBILEFACENET_MODEL = "models/mobilefacenet.onnx"

VIDEO_SOURCE = 0

INPUT_SIZE    = 640
CONF_THRESH   = 0.7
NMS_THRESH    = 0.4
COSINE_THRESH = 0.43
RESET_TIME    = 10 * 60
MIN_FRAMES    = 5

# API Configuration
API_ENABLED = True
CAMERA_ID   = "CAM002"
API_URL     = "https://smartattendance-cte8.onrender.com/api/attendance/log"
API_KEY     = "sk-dev-4zqMBrU2Ulx5Va4AG7eByIRINJn216WkxqaL5olpAGo"

# =========================
# FRAME ENHANCEMENT SETTINGS
# =========================
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE  = (8, 8)

ENABLE_BRIGHTNESS_CORRECTION = True
ENABLE_CLAHE                  = True
ENABLE_SHARPEN                = True

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)

# =========================
# FRAME ENHANCEMENT FUNCTIONS
# =========================
def fix_brightness(frame: np.ndarray) -> np.ndarray:
    """Auto-correct dark or overexposed frames."""
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 60:
        return cv2.convertScaleAbs(frame, alpha=1.4, beta=30)
    elif brightness > 200:
        return cv2.convertScaleAbs(frame, alpha=0.8, beta=-20)
    return frame


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE on L channel only — improves contrast without color distortion."""
    lab          = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b      = cv2.split(lab)
    l_enhanced   = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def sharpen_frame(frame: np.ndarray) -> np.ndarray:
    """Sharpen only if frame is blurry using unsharp mask."""
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < 80.0:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
        return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    return frame


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Enhance every frame: brightness fix -> CLAHE -> sharpen."""
    if ENABLE_BRIGHTNESS_CORRECTION:
        frame = fix_brightness(frame)
    if ENABLE_CLAHE:
        frame = apply_clahe(frame)
    if ENABLE_SHARPEN:
        frame = sharpen_frame(frame)
    return frame

# =========================
# API FUNCTION
# =========================
def send_api(user_id: str, score: float) -> None:
    """Send attendance log to API."""
    if not API_ENABLED:
        return
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    payload   = {
        "user_id"          : int(user_id),
        "confidence_score" : float(score),
        "camera_id"        : CAMERA_ID,
        "timestamp"        : timestamp
    }
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    try:
        res = requests.post(API_URL, json=payload, headers=headers, timeout=5)
        print(f"[API] {res.status_code} {res.text}")
    except Exception as e:
        print("[API ERROR]", e)

# =========================
# RETINAFACE UTILITIES
# =========================
def generate_anchors(size: int) -> np.ndarray:
    cfg     = {'min_sizes': [[16,32],[64,128],[256,512]], 'steps':[8,16,32]}
    anchors = []
    for k, step in enumerate(cfg['steps']):
        fm = ceil(size / step)
        for i, j in product(range(fm), repeat=2):
            for ms in cfg['min_sizes'][k]:
                anchors.append([(j+0.5)*step/size, (i+0.5)*step/size, ms/size, ms/size])
    return np.array(anchors, dtype=np.float32)


def decode_boxes(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    VAR   = [0.1, 0.2]
    boxes = np.concatenate([
        priors[:,:2] + loc[:,:2]*VAR[0]*priors[:,2:],
        priors[:,2:] * np.exp(loc[:,2:]*VAR[1])
    ], axis=1)
    boxes[:,:2] -= boxes[:,2:] / 2
    boxes[:,2:] += boxes[:,:2]
    return boxes


def calculate_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0],b[0]), max(a[1],b[1])
    x2, y2 = min(a[2],b[2]), min(a[3],b[3])
    inter  = max(0,x2-x1) * max(0,y2-y1)
    areaA  = (a[2]-a[0])*(a[3]-a[1])
    areaB  = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, th: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        remaining = [j for j in idxs[1:] if calculate_iou(boxes[i], boxes[j]) < th]
        idxs = np.array(remaining)
    return keep

# =========================
# FACE RECOGNITION
# =========================
def get_face_embedding(face: np.ndarray, net: cv2.dnn.Net) -> Optional[np.ndarray]:
    if face.size == 0:
        return None
    try:
        face = cv2.resize(face, (112, 112))
        blob = cv2.dnn.blobFromImage(face, 1/127.5, (112,112), (127.5,127.5,127.5), swapRB=False)
        net.setInput(blob)
        emb  = net.forward().flatten()
        norm = np.linalg.norm(emb)
        return None if norm == 0 else emb / norm
    except:
        return None


def find_best_match(query: np.ndarray, db: np.ndarray, ids: List[int], th: float) -> Tuple[Optional[int], float]:
    if len(db) == 0:
        return None, 0.0
    sims       = db @ query
    best_idx   = sims.argmax()
    best_score = float(sims[best_idx])
    return (ids[best_idx], best_score) if best_score >= th else (None, best_score)

# =========================
# MODEL LOADING
# =========================
def load_mobilefacenet(path: str) -> cv2.dnn.Net:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return cv2.dnn.readNetFromONNX(path)


def load_retinaface(path: str) -> Tuple[ort.InferenceSession, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name


# =========================
# NPZ DATABASE LOADING
# Replaces load_face_database() which needed .pkl files + MobileFaceNet at load time.
# NPZ already has pre-computed embeddings — no model needed at startup.
#
# NPZ structure (from generate_embeddings.py):
#   embeddings : np.ndarray  shape (N, 512)  — L2-normalised 512D vectors
#   labels     : np.ndarray  shape (N,)      — labels like "123_face0", "456_face1"
#
# User ID is parsed from label prefix before the first underscore.
# Multiple embeddings per user are averaged into one mean embedding.
# =========================
def load_face_database_npz(npz_path: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load pre-computed embeddings from NPZ.
    
    Strategy:
    1. Try to extract user ID from NPZ filename (e.g., "5.npz" → user_id = 5)
    2. If filename is not numeric, generate a stable hash-based ID from the filename
    3. All embeddings in the file are averaged into one mean embedding for that user
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ database not found: {npz_path}")

    data       = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)   # (N, 512)
    labels     = data["labels"]                          # (N,)

    print(f"[INFO] NPZ loaded: {len(embeddings)} embeddings from '{npz_path}'")

    # Extract user ID from NPZ filename
    npz_filename = os.path.basename(npz_path)
    npz_name     = os.path.splitext(npz_filename)[0]  # "5.npz" → "5", "ramu.npz" → "ramu"
    
    user_id = None
    try:
        user_id = int(npz_name)
        print(f"[INFO] Extracted user ID {user_id} from filename '{npz_filename}'")
    except ValueError:
        # Filename is not numeric — generate a stable hash-based ID
        # Hash the filename to get a consistent numeric ID
        user_id = abs(hash(npz_name)) % 1000000  # limit to 6 digits
        print(f"[INFO] Generated user ID {user_id} from filename '{npz_filename}' (hash-based)")

    # Average all embeddings into one mean embedding
    mean_emb = np.mean(embeddings, axis=0)
    norm     = np.linalg.norm(mean_emb)
    
    if norm == 0:
        raise ValueError(f"Invalid embeddings in {npz_path} — zero norm after averaging")
    
    mean_emb = mean_emb / norm
    print(f"[INFO] Database ready: 1 identity (user {user_id})")
    return np.array([mean_emb], dtype=np.float32), [user_id]

# =========================
# IMAGE UTILITIES
# =========================
def extract_face_roi(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
    x2, y2 = max(0, min(x2, w)),   max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    face = frame[y1:y2, x1:x2]
    return None if face.size == 0 else face

# =========================
# MAIN PIPELINE
# =========================
def run_face_recognition():
    """Main face recognition pipeline."""

    # Load models
    print("[INFO] Loading models...")
    mf_net                 = load_mobilefacenet(MOBILEFACENET_MODEL)
    rf_session, input_name = load_retinaface(RETINAFACE_MODEL)
    anchors                = generate_anchors(INPUT_SIZE)

    # Load NPZ database (no model needed here — embeddings already computed)
    print("[INFO] Loading face database from NPZ...")
    known_embs, known_ids = load_face_database_npz(FACE_DB_NPZ)

    # Initialize video
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cv2.namedWindow("RetinaFace + MobileFaceNet", cv2.WINDOW_NORMAL)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name   = "Webcam" if VIDEO_SOURCE == 0 else str(VIDEO_SOURCE)

    # Initialize tracking
    stats = {
        "processed_frames" : 0,
        "frames_with_det"  : 0,
        "frames_rec_known" : 0,
        "frames_unknown"   : 0
    }
    person_counter = Counter()
    person_scores  = {}
    fps_list       = []
    seen           = {}

    start_time = time.time()
    prev_time  = time.time()

    print("[INFO] Starting video processing...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            stats["processed_frames"] += 1
            h, w = frame.shape[:2]

            # ENHANCE FRAME
            frame = enhance_frame(frame)

            # DETECT FACES
            img  = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
            img -= np.array([104.0, 117.0, 123.0])
            blob = img.transpose(2, 0, 1)[None, ...]

            loc, conf = rf_session.run(None, {input_name: blob})[:2]
            boxes     = decode_boxes(loc[0], anchors)
            scores    = conf[0][:, 1]

            mask          = scores > CONF_THRESH
            boxes, scores = boxes[mask], scores[mask]

            if len(boxes) > 0:
                stats["frames_with_det"] += 1
                keep  = nms(boxes, scores, NMS_THRESH)
                boxes = boxes[keep]
                boxes[:, [0, 2]] *= w
                boxes[:, [1, 3]] *= h

            # RECOGNIZE EACH FACE
            for bbox in boxes:
                face = extract_face_roi(frame, bbox)
                if face is None:
                    continue

                emb = get_face_embedding(face, mf_net)
                if emb is None:
                    continue

                matched_id, sim = find_best_match(emb, known_embs, known_ids, COSINE_THRESH)
                x1, y1, x2, y2 = map(int, bbox)

                if matched_id is not None:
                    name  = str(matched_id)
                    color = (0, 255, 0)
                    stats["frames_rec_known"] += 1
                    person_counter[name] += 1

                    if name not in person_scores:
                        person_scores[name] = []
                    person_scores[name].append(sim)

                    now = time.time()
                    if person_counter[name] >= MIN_FRAMES:
                        if name not in seen or (now - seen[name]) >= RESET_TIME:
                            send_api(name, sim)
                            seen[name] = now
                else:
                    name  = "Unknown"
                    color = (0, 0, 255)
                    stats["frames_unknown"] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ({sim:.2f})", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # FPS DISPLAY
            fps       = 1.0 / (time.time() - prev_time)
            prev_time = time.time()
            fps_list.append(fps)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("RetinaFace + MobileFaceNet", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # REPORT
    total_time = time.time() - start_time
    avg_fps    = sum(fps_list) / len(fps_list) if fps_list else 0.0
    min_fps    = min(fps_list) if fps_list else 0.0
    max_fps    = max(fps_list) if fps_list else 0.0

    per_person_report = "\n".join([
        f"{p}:{person_counter[p]}:{np.mean(person_scores[p]):.2f}"
        for p in person_counter if person_counter[p] >= MIN_FRAMES
    ])

    print(f"""
---------------------------------------------------------
RUN DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
PIPELINE REPORT: RETINAFACE + MOBILEFACENET (512-D)
---------------------------------------------------------

VIDEO DETAILS:
- Video Name : {video_name}
- Resolution : {video_width} x {video_height}

PERFORMANCE:
- Time Taken : {total_time:.2f} sec
- Avg FPS    : {avg_fps:.2f}
- Min FPS    : {min_fps:.2f}
- Max FPS    : {max_fps:.2f}

DETECTION:
- Frames with Face : {stats['frames_with_det']}
- Known Frames     : {stats['frames_rec_known']}
- Unknown Frames   : {stats['frames_unknown']}

PER-PERSON FRAME COUNT (confidence):
{per_person_report}

Cosine Threshold : {COSINE_THRESH}
---------------------------------------------------------
""")
    print("[INFO] Finished")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    try:
        run_face_recognition()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise