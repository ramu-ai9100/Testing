#Face Recognition with NPZ Database - Adaptive Threshold & Async API

import os, cv2, time, numpy as np
from datetime import datetime
from typing import Tuple, List, Optional
import onnxruntime as ort
import requests
from itertools import product
from math import ceil
from collections import Counter
import threading
from queue import Queue

# =========================
# CONFIGURATION
# =========================
FACE_DIR = "faces/mobile2"
RETINAFACE_MODEL = "models/RetinaFace_mobile0.25_640_opset18.onnx"
MOBILEFACENET_MODEL = "models/mobilefacenet.onnx"

VIDEO_SOURCE = "videos/individual.mp4"  #0     #continuous   individual

INPUT_SIZE  = 640
CONF_THRESH = 0.7
NMS_THRESH  = 0.4

# ===== ADAPTIVE THRESHOLD (OPTIONAL) =====
USE_ADAPTIVE_THRESHOLD = True
# Set True to enable
COSINE_THRESH_BRIGHT = 0.46      # Threshold for bright frames
COSINE_THRESH_DARK   = 0.42      # Threshold for dark frames
BRIGHTNESS_CUTOFF    = 80        # < 80 = dark, >= 80 = bright

# Fixed threshold (used when adaptive is OFF)
COSINE_THRESH = 0.46

RESET_TIME = 10 * 60  # seconds
MIN_FRAMES = 5

# API Configuration
API_ENABLED = False
CAMERA_ID   = "CAM002"
API_URL     = "https://smartattendance-cte8.onrender.com/api/attendance/log"
API_KEY     = "sk-dev--2FFGXBuqCAVe6_VMOSOMsVVJPn_RBJbTA9jRL2xF94"

# =========================
# ASYNC API HANDLER
# =========================
class AsyncAPIHandler:
    
    def __init__(self):
        self.queue = Queue()
        self.stop_flag = False
        self.stats = {"total_sent": 0, "success": 0, "failed": 0, "pending": 0}
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("[INFO] Async API handler started")
    
    def send_async(self, user_id: str, score: float):
        """Queue an API request (non-blocking, returns instantly)."""
        if not API_ENABLED:
            return
        
        with self.lock:
            self.stats["pending"] += 1
            self.stats["total_sent"] += 1
        
        self.queue.put({
            "user_id": user_id,
            "score": score,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        })
    
    def _worker(self):
        """Background thread that processes API queue."""
        while not self.stop_flag:
            try:
                request = self.queue.get(timeout=1)
                payload = {
                    "user_id": int(request["user_id"]),
                    "confidence_score": float(request["score"]),
                    "camera_id": CAMERA_ID,
                    "timestamp": request["timestamp"]
                }
                headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
                
                try:
                    res = requests.post(API_URL, json=payload, headers=headers, timeout=5)
                    with self.lock:
                        self.stats["pending"] -= 1
                        if 200 <= res.status_code < 300:
                            self.stats["success"] += 1
                            print(f"[API OK] User {request['user_id']} - {res.status_code} {res.text}")
                        else:
                            self.stats["failed"] += 1
                            print(f"[API X] User {request['user_id']} - {res.status_code}")
                except requests.exceptions.RequestException as e:
                    with self.lock:
                        self.stats["pending"] -= 1
                        self.stats["failed"] += 1
                    print(f"[API ERROR] User {request['user_id']}: {e}")
                
                self.queue.task_done()
            except:
                continue
    
    def stop(self):
        self.stop_flag = True
        self.worker_thread.join(timeout=2)
    
    def get_stats(self):
        with self.lock:
            return self.stats.copy()

# Global API handler
api_handler = AsyncAPIHandler()

# =========================
# ADAPTIVE THRESHOLD
# =========================
def get_adaptive_threshold(face_crop: np.ndarray) -> float:
    """Calculate threshold based on face brightness (optional feature)."""
    if not USE_ADAPTIVE_THRESHOLD:
        return COSINE_THRESH
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return COSINE_THRESH_DARK if brightness < BRIGHTNESS_CUTOFF else COSINE_THRESH_BRIGHT

# =========================
# RETINAFACE UTILITIES
# =========================
def generate_anchors(size: int) -> np.ndarray:
    cfg = {'min_sizes': [[16,32],[64,128],[256,512]], 'steps':[8,16,32]}
    anchors = []
    for k, step in enumerate(cfg['steps']):
        fm = ceil(size / step)
        for i, j in product(range(fm), repeat=2):
            for ms in cfg['min_sizes'][k]:
                anchors.append([(j+0.5)*step/size, (i+0.5)*step/size, ms/size, ms/size])
    return np.array(anchors, dtype=np.float32)

def decode_boxes(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    VAR = [0.1, 0.2]
    boxes = np.concatenate([priors[:,:2] + loc[:,:2]*VAR[0]*priors[:,2:],
                            priors[:,2:] * np.exp(loc[:,2:]*VAR[1])], axis=1)
    boxes[:,:2] -= boxes[:,2:] / 2
    boxes[:,2:] += boxes[:,:2]
    return boxes

def calculate_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0],b[0]), max(a[1],b[1])
    x2, y2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,x2-x1) * max(0,y2-y1)
    areaA, areaB = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

def nms(boxes: np.ndarray, scores: np.ndarray, th: float) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        keep.append(idxs[0])
        if len(idxs) == 1:
            break
        remaining = [j for j in idxs[1:] if calculate_iou(boxes[idxs[0]], boxes[j]) < th]
        idxs = np.array(remaining)
    return keep

# =========================
# FACE RECOGNITION
# =========================
def get_face_embedding(face: np.ndarray, net: cv2.dnn.Net) -> Optional[np.ndarray]:
    """Extract L2-normalized 512D embedding (no enhancement)."""
    if face.size == 0:
        return None
    try:
        face = cv2.resize(face, (112, 112))
        blob = cv2.dnn.blobFromImage(face, 1/127.5, (112,112), (127.5,127.5,127.5), swapRB=False)
        net.setInput(blob)
        emb = net.forward().flatten()
        norm = np.linalg.norm(emb)
        return None if norm == 0 else emb / norm
    except:
        return None

def find_best_match(query: np.ndarray, db: np.ndarray, ids: List[int], th: float) -> Tuple[Optional[int], float]:
    if len(db) == 0:
        return None, 0.0
    sims = db @ query
    best_idx = sims.argmax()
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

def load_face_database(face_dir: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load NPZ database with pre-computed embeddings.
    
    Expected NPZ file format:
    - Filename: {user_id}.npz (e.g., 1.npz, 2.npz, 5.npz)
    - Content: {'embeddings': array of shape (N, 512)}
    
    Returns:
        embeddings: (M, 512) array where M is number of users
        ids: List of user IDs
    """
    if not os.path.exists(face_dir):
        raise FileNotFoundError(f"Directory not found: {face_dir}")
    
    embeddings, ids = [], []
    npz_files = [f for f in os.listdir(face_dir) if f.endswith(".npz")]
    
    if not npz_files:
        raise ValueError(f"No NPZ files found in {face_dir}")
    
    print(f"[INFO] Found {len(npz_files)} NPZ files")
    
    for f in sorted(npz_files):
        try:
            # Extract user ID from filename
            uid = int(os.path.splitext(f)[0])
            
            # Load NPZ file
            filepath = os.path.join(face_dir, f)
            data = np.load(filepath)
            
            # Check if 'embeddings' key exists
            if 'embeddings' not in data:
                print(f"[WARNING] {f} missing 'embeddings' key, skipping")
                continue
            
            stored_embs = data['embeddings']
            
            # Validate shape (should be (N, 512))
            if stored_embs.ndim != 2 or stored_embs.shape[1] != 512:
                print(f"[WARNING] {f} has invalid shape {stored_embs.shape}, expected (N, 512), skipping")
                continue
            
            if len(stored_embs) == 0:
                print(f"[WARNING] {f} has no embeddings, skipping")
                continue
            
            # Average all embeddings for this person
            mean_emb = np.mean(stored_embs, axis=0)
            
            # L2 normalize
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
                embeddings.append(mean_emb)
                ids.append(uid)
                print(f"[LOADED] User {uid}: {len(stored_embs)} embeddings averaged")
            else:
                print(f"[WARNING] {f} has zero-norm embedding, skipping")
                
        except ValueError as e:
            print(f"[WARNING] Invalid filename {f}: {e}, skipping")
        except Exception as e:
            print(f"[ERROR] Failed to load {f}: {e}, skipping")
    
    if len(embeddings) == 0:
        raise ValueError(f"No valid embeddings loaded from {face_dir}")
    
    return np.array(embeddings, dtype=np.float32), ids

# =========================
# IMAGE UTILITIES
# =========================
def extract_face_roi(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, min(x1, w-1)), max(0, min(y1, h-1))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    face = frame[y1:y2, x1:x2]
    return None if face.size == 0 else face

# =========================
# MAIN PIPELINE
# =========================
def run_face_recognition():
    print("[INFO] Loading models...")
    mf_net = load_mobilefacenet(MOBILEFACENET_MODEL)
    rf_session, input_name = load_retinaface(RETINAFACE_MODEL)
    anchors = generate_anchors(INPUT_SIZE)

    print("[INFO] Loading face database from NPZ files...")
    known_embs, known_ids = load_face_database(FACE_DIR)
    print(f"[INFO] Loaded {len(known_ids)} identities")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 800, 600)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = "Webcam" if VIDEO_SOURCE == 0 else str(VIDEO_SOURCE)

    stats = {"processed_frames": 0, "frames_with_det": 0,
             "frames_rec_known": 0, "frames_unknown": 0}
    person_counter = Counter()
    person_scores = {}
    fps_list, seen = [], {}
    threshold_stats = {"bright": 0, "dark": 0}

    start_time = time.time()
    prev_time = time.time()

    adaptive_status = "ENABLED" if USE_ADAPTIVE_THRESHOLD else "DISABLED"
    print(f"[INFO] Adaptive Threshold: {adaptive_status}")
    print(f"[INFO] Database Format: NPZ (pre-computed embeddings)")
    print("[INFO] Starting video processing...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            stats["processed_frames"] += 1
            h, w = frame.shape[:2]

            # DETECT FACES
            img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
            img -= np.array([104.0, 117.0, 123.0])
            blob = img.transpose(2, 0, 1)[None, ...]

            loc, conf = rf_session.run(None, {input_name: blob})[:2]
            boxes = decode_boxes(loc[0], anchors)
            scores = conf[0][:, 1]

            mask = scores > CONF_THRESH
            boxes, scores = boxes[mask], scores[mask]

            if len(boxes) > 0:
                stats["frames_with_det"] += 1
                keep = nms(boxes, scores, NMS_THRESH)
                boxes = boxes[keep]
                boxes[:, [0, 2]] *= w
                boxes[:, [1, 3]] *= h

            # RECOGNIZE EACH FACE
            for bbox in boxes:
                face = extract_face_roi(frame, bbox)
                if face is None:
                    continue

                # Get threshold (adaptive or fixed)
                dynamic_threshold = get_adaptive_threshold(face)
                
                if USE_ADAPTIVE_THRESHOLD:
                    threshold_stats["dark" if dynamic_threshold == COSINE_THRESH_DARK else "bright"] += 1

                emb = get_face_embedding(face, mf_net)
                if emb is None:
                    continue

                matched_id, sim = find_best_match(emb, known_embs, known_ids, dynamic_threshold)
                x1, y1, x2, y2 = map(int, bbox)

                if matched_id is not None:
                    name = str(matched_id)
                    color = (0, 255, 0)
                    stats["frames_rec_known"] += 1
                    person_counter[name] += 1

                    if name not in person_scores:
                        person_scores[name] = []
                    person_scores[name].append(sim)

                    now = time.time()
                    if person_counter[name] >= MIN_FRAMES:
                        if name not in seen or (now - seen[name]) >= RESET_TIME:
                            # ASYNC API - non-blocking
                            api_handler.send_async(name, sim)
                            seen[name] = now
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    stats["frames_unknown"] += 1

                label = f"{name} ({sim:.2f})"
                if USE_ADAPTIVE_THRESHOLD:
                    label += f" T:{dynamic_threshold:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # FPS DISPLAY
            fps = 1.0 / (time.time() - prev_time)
            prev_time = time.time()
            fps_list.append(fps)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            # API STATUS
            api_stats = api_handler.get_stats()
            api_info = f"API: {api_stats['success']}done {api_stats['failed']}Fail {api_stats['pending']}pending"
            cv2.putText(frame, api_info, (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)
            
            if USE_ADAPTIVE_THRESHOLD:
                cv2.putText(frame, "ADAPTIVE MODE", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Stopping async API handler...")
        api_handler.stop()

    # FINAL REPORT
    total_time = time.time() - start_time
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
    min_fps = min(fps_list) if fps_list else 0.0
    max_fps = max(fps_list) if fps_list else 0.0

    per_person_report = "\n".join([
        f"{p}:{person_counter[p]}:{np.mean(person_scores[p]):.2f}"
        for p in person_counter if person_counter[p] >= MIN_FRAMES
    ])

    threshold_report = ""
    if USE_ADAPTIVE_THRESHOLD:
        total = threshold_stats["bright"] + threshold_stats["dark"]
        if total > 0:
            b_pct = (threshold_stats["bright"] / total) * 100
            d_pct = (threshold_stats["dark"] / total) * 100
            threshold_report = f"""
ADAPTIVE THRESHOLD:
- Bright ({COSINE_THRESH_BRIGHT}): {threshold_stats["bright"]} ({b_pct:.1f}%)
- Dark   ({COSINE_THRESH_DARK}):   {threshold_stats["dark"]} ({d_pct:.1f}%)"""

    api_final = api_handler.get_stats()
    api_report = f"""
API STATISTICS:
- Total Sent: {api_final['total_sent']}
- Success   : {api_final['success']} success
- Failed    : {api_final['failed']} failed
- Pending   : {api_final['pending']}"""

    print(f"""
---------------------------------------------------------
RUN DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
PIPELINE: RETINAFACE + MOBILEFACENET (NPZ DATABASE)
---------------------------------------------------------

VIDEO:
- Source     : {video_name}
- Resolution : {video_width} x {video_height}

PERFORMANCE:
- Time       : {total_time:.2f} sec
- Avg FPS    : {avg_fps:.2f}
- Min FPS    : {min_fps:.2f}
- Max FPS    : {max_fps:.2f}

DETECTION:
- Total Frames  : {stats['processed_frames']}
- Frames w/Face : {stats['frames_with_det']}
- Known         : {stats['frames_rec_known']}
- Unknown       : {stats['frames_unknown']}

PER-PERSON (>={MIN_FRAMES} frames):
{per_person_report if per_person_report else 'None'}

Database Format: NPZ (pre-computed embeddings)
Adaptive Threshold: {adaptive_status}{threshold_report}{api_report}
Fixed Threshold   : {COSINE_THRESH if not USE_ADAPTIVE_THRESHOLD else 'N/A'}
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
        api_handler.stop()
    except Exception as e:
        print(f"[ERROR] {e}")
        api_handler.stop()
        raise


