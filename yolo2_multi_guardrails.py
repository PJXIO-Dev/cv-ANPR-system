
import os, re, time, base64
import cv2, torch, numpy as np
from ultralytics import YOLO
from openai import OpenAI
from collections import deque, defaultdict

# -----------------------------
# Config
# -----------------------------
DETECT_TARGET_WIDTH = 1280
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
PADDING = 8
OCR_RESIZE = (256, 128)
DETECT_EVERY_N_FRAMES = 10
HIGHER_CONF_DELTA = 0.02
MAX_AGE = 30
MIN_IOU_MATCH = 0.3
ROI_FROM_CENTER = True
START_ROI_FRAC = 0.5

# Dedup / ownership
DEDUP_IOU_THRESHOLD = 0.2
DRAW_GATING_REFRESHES = 2  

# Car box
CAR_W_FACTOR = 1.8
CAR_H_FACTOR = 1.4
CAR_W_CAP    = 0.22
CAR_H_CAP    = 0.28
CAR_UP_BIAS  = 0.45

# KLT
GFTT_MAX_CORNERS = 100
GFTT_QUALITY     = 0.01
GFTT_MIN_DIST    = 5
LK_WIN_SIZE      = (21, 21)
LK_MAX_LEVEL     = 3
LK_TERM_CRIT     = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
MIN_POINTS_TO_TRACK = 12
RESEED_POINTS_THRESHOLD = 8

# Exit / quality
EDGE_MARGIN_PX = 3
MAX_OFFSCREEN_FRAMES = 6
MAX_WEAK_POINTS_FRAMES = 10
MAX_MISSED_REFRESHES = 3
MIN_VISIBLE_FRAC = 0.25

# -----------------------------
# Device / models / I/O
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model = YOLO("/home/cv-sse/Downloads/yolo11m.pt").to(device)

# OpenAI client (read from env)
client = OpenAI(api_key=" ") 


cap = cv2.VideoCapture("/home/cv-sse/Downloads/car-plates (online-video-cutter.com).mp4")
assert cap.isOpened(), "Error reading video file"
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

out_path = "/home/cv-sse/Downloads/car-plates-MMM.mp4"
video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

# -----------------------------
# OCR helpers
# -----------------------------
prompt = (
    "Can you extract the vehicle number plate text inside the image?\n"
    "If you are not able to extract text, please respond with None.\n"
    "Only output text, please.\n"
    "If any text character is not from the English language, replace it with a dot (.)."
)
UK_PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$')  # GB (since 2001)

def sanitize_plate_text(txt: str) -> str:
    if not txt:
        return ""
    # uppercase, keep only letters, digits, space
    txt = txt.upper()
    txt = re.sub(r'[^A-Z0-9 ]', '', txt)
    # collapse multiple spaces
    txt = re.sub(r'\s+', ' ', txt).strip()
    # common hallucinations: O/0, I/1, S/5 near edges — do NOT fix aggressively; just validate
    return txt

def is_valid_plate(txt: str) -> bool:
    if not txt or txt == "NONE":
        return False
    return bool(UK_PLATE_RE.match(txt))

def extract_text_base64(base64_encoded_data: str) -> str:
    delay = 1.0
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_encoded_data}"}}
                    ]
                }],
            )
            raw = (resp.choices[0].message.content or "").strip()
            s = sanitize_plate_text(raw)
            if is_valid_plate(s):
                return s
            # not valid → treat as None
            return ""
        except Exception as e:
            print(f"OCR error (attempt {attempt+1}): {e}")
            time.sleep(delay); delay *= 1.6
    return ""

# -----------------------------
# Geometry & drawing
# -----------------------------
def resize_for_detection(frame):
    H0, W0 = frame.shape[:2]
    if W0 <= DETECT_TARGET_WIDTH:
        return frame, 1.0
    scale = DETECT_TARGET_WIDTH / float(W0)
    new_w = DETECT_TARGET_WIDTH; new_h = int(H0 * scale)
    im_small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return im_small, scale

def get_detection_roi_small(im_small, scale, H_orig):
    if not ROI_FROM_CENTER:
        return im_small, 0
    y0_small = int((H_orig * START_ROI_FRAC) * scale)
    y0_small = max(0, min(im_small.shape[0] - 1, y0_small))
    roi_small = im_small[y0_small:, :]
    return roi_small, y0_small

def clamp(v, lo, hi): return max(lo, min(hi, v))

def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def visible_fraction(box):
    x1,y1,x2,y2 = box
    bx1 = clamp(x1, 0, W-1); by1 = clamp(y1, 0, H-1)
    bx2 = clamp(x2, 0, W-1); by2 = clamp(y2, 0, H-1)
    vis = max(0, bx2 - bx1) * max(0, by2 - by1)
    area = max(0, x2 - x1) * max(0, y2 - y1) + 1e-6
    return vis / area

def draw_vehicle_box_and_label(im, plate_box, plate_text):
    H_, W_ = im.shape[:2]
    x1p, y1p, x2p, y2p = plate_box
    cx = int((x1p + x2p) * 0.5)
    cy = int((y1p + y2p) * 0.5)
    pw = max(1, x2p - x1p)
    ph = max(1, y2p - y1p)

    # Car-range box
    car_w = int(pw * CAR_W_FACTOR)
    car_h = int(ph * CAR_H_FACTOR)
    car_w = min(car_w, int(W_ * CAR_W_CAP))
    car_h = min(car_h, int(H_ * CAR_H_CAP))

    car_x1 = clamp(cx - car_w // 2, 0, W_ - 1)
    car_y1 = clamp(cy - int(car_h * CAR_UP_BIAS), 0, H_ - 1)
    car_x2 = clamp(car_x1 + car_w, 0, W_ - 1)
    car_y2 = clamp(car_y1 + car_h, 0, H_ - 1)

    cv2.rectangle(im, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 4)

    # Label — move to top-left corner of the bbox
    show_text = plate_text if plate_text else "—"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(1.2, (car_x2 - car_x1) / 700.0))
    (tw, th), baseline = cv2.getTextSize(show_text, font, font_scale, 2)
    pad_x, pad_y = 8, 6

    # Top-left placement
    lbl_x1 = car_x1 + 4
    lbl_y1 = max(0, car_y1 - th - 2 * pad_y)
    lbl_y2 = lbl_y1 + th + 2 * pad_y
    lbl_x2 = lbl_x1 + tw + 2 * pad_x

    # Draw background box and text
    cv2.rectangle(im, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), (255, 255, 255), -1)
    cv2.rectangle(im, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), (0, 0, 0), 2)
    text_x = lbl_x1 + pad_x
    text_y = lbl_y1 + pad_y + th
    cv2.putText(im, show_text, (text_x, text_y), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

# -----------------------------
# KLT tracker
# -----------------------------
class KLTBoxTracker:
    def __init__(self):
        self.prev_gray=None; self.points=None; self.box=None; self._points_count=0
    def _seed_points(self, gray, box):
        x1,y1,x2,y2 = [int(v) for v in box]
        x1 = clamp(x1, 0, gray.shape[1]-1); y1 = clamp(y1, 0, gray.shape[0]-1)
        x2 = clamp(x2, 0, gray.shape[1]-1); y2 = clamp(y2, 0, gray.shape[0]-1)
        if x2<=x1 or y2<=y1: self.points=None; self._points_count=0; return
        roi = gray[y1:y2, x1:x2]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=GFTT_MAX_CORNERS, qualityLevel=GFTT_QUALITY, minDistance=GFTT_MIN_DIST)
        if pts is not None: pts[:,0,0]+=x1; pts[:,0,1]+=y1
        self.points=pts; self._points_count=0 if pts is None else int(pts.shape[0])
    def init(self, frame, box_xyxy):
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray=gray; self.box=tuple(int(v) for v in box_xyxy); self._seed_points(gray, self.box)
    def update(self, frame):
        if self.prev_gray is None or self.box is None: return None
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.points is None or len(self.points)==0:
            self._seed_points(gray, self.box); self.prev_gray=gray; return self.box
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.points, None,
                                               winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_TERM_CRIT)
        if p1 is None or st is None:
            self.prev_gray=gray; self._seed_points(gray, self.box); return self.box
        good_new=p1[st.reshape(-1)==1]; good_old=self.points[st.reshape(-1)==1]
        self._points_count=int(good_new.shape[0])
        if self._points_count < RESEED_POINTS_THRESHOLD:
            self.prev_gray=gray; self._seed_points(gray, self.box); return self.box
        flow=(good_new - good_old).reshape(-1,2); dx=np.median(flow[:,0]); dy=np.median(flow[:,1])
        x1,y1,x2,y2=self.box; w=x2-x1; h=y2-y1
        x1n=clamp(int(x1+dx),0,W-1); y1n=clamp(int(y1+dy),0,H-1)
        x2n=clamp(int(x1n+w),0,W-1); y2n=clamp(int(y1n+h),0,H-1)
        if x2n<=x1n or y2n<=y1n: x1n,y1n,x2n,y2n=x1,y1,x2,y2
        self.box=(x1n,y1n,x2n,y2n); self.prev_gray=gray
        if self._points_count < MIN_POINTS_TO_TRACK: self._seed_points(gray, self.box)
        else: self.points=good_new.reshape(-1,1,2).astype(np.float32)
        return self.box
    def points_count(self): return self._points_count

# -----------------------------
# Tracking objects
# -----------------------------
class Track:
    _next_id=1
    def __init__(self, box, conf, frame, plate_text=""):
        self.id=Track._next_id; Track._next_id+=1
        self.box=tuple(int(v) for v in box); self.best_conf=float(conf)
        self.tracker=KLTBoxTracker(); self.tracker.init(frame, self.box)
        self.plate_text=plate_text; self.age=0; self.time_since_update=0; self.hits=1
        self.history=deque(maxlen=10)
        self.offscreen_frames=0; self.weak_points_frames=0; self.missed_refreshes=0
        self.refresh_age=0   # frames since last matched detection refresh

    def mark_seen_in_detection(self):
        self.missed_refreshes=0; self.refresh_age=0

    def update_with_tracker(self, frame):
        tbox=self.tracker.update(frame); self.age+=1; self.refresh_age+=1
        if tbox is not None:
            self.box=tbox; self.time_since_update=0; self.hits+=1; self.history.append(self.box)
        else:
            self.time_since_update+=1
        pc=self.tracker.points_count()
        if pc < RESEED_POINTS_THRESHOLD: self.weak_points_frames+=1
        else: self.weak_points_frames=0
        x1,y1,x2,y2=self.box
        near_edge=(x1<=EDGE_MARGIN_PX or y1<=EDGE_MARGIN_PX or x2>=W-EDGE_MARGIN_PX or y2>=H-EDGE_MARGIN_PX)
        low_vis=(visible_fraction(self.box) < MIN_VISIBLE_FRAC)
        if near_edge or low_vis: self.offscreen_frames+=1
        else: self.offscreen_frames=0

    def try_upgrade_with_detection(self, det_box, det_conf, frame):
        if iou(self.box, det_box) < MIN_IOU_MATCH: return False
        self.mark_seen_in_detection()
        if float(det_conf) > self.best_conf + HIGHER_CONF_DELTA:
            self.box=tuple(int(v) for v in det_box); self.best_conf=float(det_conf)
            self.tracker=KLTBoxTracker(); self.tracker.init(frame, self.box)
            self.plate_text=""; return True
        return False

    def should_delete(self):
        if self.time_since_update > MAX_AGE: return True
        if self.offscreen_frames >= MAX_OFFSCREEN_FRAMES: return True
        if self.weak_points_frames >= MAX_WEAK_POINTS_FRAMES: return True
        if self.missed_refreshes >= MAX_MISSED_REFRESHES: return True
        x1,y1,x2,y2=self.box
        if (x2-x1) < 5 or (y2-y1) < 5: return True
        return False

# -----------------------------
# Helpers
# -----------------------------
def assign_detections_to_tracks(det_boxes, tracks):
    if len(det_boxes)==0 or len(tracks)==0:
        return [], list(range(len(det_boxes))), list(range(len(tracks)))
    iou_matrix=np.zeros((len(det_boxes), len(tracks)), dtype=np.float32)
    for d, db in enumerate(det_boxes):
        for t, tr in enumerate(tracks):
            iou_matrix[d,t]=iou(tuple(map(int, db)), tr.box)
    matches=[]; det_used=set(); trk_used=set()
    while True:
        d,t = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        if iou_matrix[d,t] < MIN_IOU_MATCH: break
        if d in det_used or t in trk_used:
            iou_matrix[d,:]=-1; iou_matrix[:,t]=-1; continue
        matches.append((d,t)); det_used.add(d); trk_used.add(t)
        iou_matrix[d,:]=-1; iou_matrix[:,t]=-1
    unmatched_dets=[i for i in range(len(det_boxes)) if i not in det_used]
    unmatched_tracks=[i for i in range(len(tracks)) if i not in trk_used]
    return matches, unmatched_dets, unmatched_tracks

def ocr_plate_text(frame, box):
    x1,y1,x2,y2=box
    x1p=max(x1-PADDING,0); y1p=max(y1-PADDING,0)
    x2p=min(x2+PADDING,W); y2p=min(y2+PADDING,H)
    plate_crop=frame[y1p:y2p, x1p:x2p]
    if plate_crop.size==0: return ""
    plate_small=cv2.resize(plate_crop, OCR_RESIZE, interpolation=cv2.INTER_AREA) if OCR_RESIZE else plate_crop
    ok, enc = cv2.imencode(".jpg", plate_small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok: return ""
    b64=base64.b64encode(enc).decode("utf-8")
    return extract_text_base64(b64)

def deduplicate_tracks(tracks):
    keep=[]; used=[False]*len(tracks)
    for i in range(len(tracks)):
        if used[i]: continue
        a=tracks[i]
        for j in range(i+1,len(tracks)):
            if used[j]: continue
            b=tracks[j]
            if a.plate_text and b.plate_text and (a.plate_text==b.plate_text):
                if iou(a.box,b.box) > DEDUP_IOU_THRESHOLD:
                    choose_a=(a.best_conf>b.best_conf) or (abs(a.best_conf-b.best_conf)<1e-6 and a.hits>=b.hits)
                    if choose_a: used[j]=True
                    else: used[i]=True; a=b
        if not used[i]: keep.append(a); used[i]=True
    return keep

def choose_plate_owners(tracks):
    """
    For each plate text, choose ONE track to render:
    priority: recently seen in detection, higher area, higher conf, lower y1 (closer).
    Returns set of allowed track IDs.
    """
    groups=defaultdict(list)
    for tr in tracks:
        if tr.plate_text:
            groups[tr.plate_text].append(tr)
    allowed=set()
    for txt, lst in groups.items():
        # rank
        lst.sort(key=lambda t: (
            t.refresh_age,                      # smaller is better
            -((t.box[2]-t.box[0])*(t.box[3]-t.box[1])),  # larger area preferred
            -t.best_conf,
            t.box[1]                            # smaller y1 means closer to top; we prefer lower? invert if needed
        ))
        allowed.add(lst[0].id)
    return allowed

# -----------------------------
# Main
# -----------------------------
tracks=[]; frame_idx=0
while cap.isOpened():
    ret, im0 = cap.read()
    if not ret: break
    frame_idx += 1

    for tr in tracks:
        tr.update_with_tracker(im0)

    need_refresh = (frame_idx % DETECT_EVERY_N_FRAMES == 0) or (len(tracks) == 0)
    matched_tracks=set()
    if need_refresh:
        im_small, scale = resize_for_detection(im0)
        roi_small, y0_small = get_detection_roi_small(im_small, scale, H)
        results = model.predict(roi_small, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        det_boxes=[]; det_confs=[]
        if len(results)>0:
            r0=results[0].boxes
            xyxy_roi = r0.xyxy.cpu().numpy() if r0 is not None else np.zeros((0,4), dtype=np.float32)
            confs = r0.conf.cpu().numpy() if r0 is not None else np.zeros((0,), dtype=np.float32)
            if xyxy_roi.size>0:
                xyxy_small=xyxy_roi.copy(); xyxy_small[:,[1,3]] += y0_small
                if scale != 1.0:
                    xyxy_small[:,[0,2]]/=scale; xyxy_small[:,[1,3]]/=scale
                for i_det in range(xyxy_small.shape[0]):
                    x1,y1,x2,y2 = [int(v) for v in xyxy_small[i_det][:4]]
                    if x2>x1 and y2>y1:
                        det_boxes.append((x1,y1,x2,y2)); det_confs.append(float(confs[i_det]))
        matches, unmatched_dets, unmatched_tracks = assign_detections_to_tracks(det_boxes, tracks)
        for det_idx, trk_idx in matches:
            db=det_boxes[det_idx]; dc=det_confs[det_idx]
            tracks[trk_idx].try_upgrade_with_detection(db, dc, im0); matched_tracks.add(trk_idx)
        for idx in range(len(tracks)):
            if idx not in matched_tracks:
                tracks[idx].missed_refreshes += 1
        for det_idx in unmatched_dets:
            db=det_boxes[det_idx]; dc=det_confs[det_idx]
            tracks.append(Track(db, dc, im0, plate_text=""))

    # OCR only when missing
    for tr in tracks:
        if tr.plate_text == "":
            txt = ocr_plate_text(im0, tr.box)
            tr.plate_text = txt if is_valid_plate(txt) else ""

    # Merge near-duplicates
    tracks = deduplicate_tracks(tracks)

    # Ownership: only one rendering per plate string
    owners = choose_plate_owners(tracks)

    # Draw with gating:
    for tr in list(tracks):
        if (not tr.plate_text) or (tr.missed_refreshes > DRAW_GATING_REFRESHES):
            continue
        if tr.id not in owners:
            continue
        draw_vehicle_box_and_label(im0, tr.box, tr.plate_text)

    # Prune old tracks
    tracks = [tr for tr in tracks if not tr.should_delete()]

    text = "msabbahi"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 5
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = W - text_w - 20
    y = H - 20
    cv2.putText(im0, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    
    video_writer.write(im0)

cap.release()
video_writer.release()
print(f"Saved: {out_path}")
