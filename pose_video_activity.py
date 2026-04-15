"""
Pose detection + activity classification on video.
Outputs: output_activity.mp4 with per-person skeleton + activity captions.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque, defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "Model/yolo26s-pose.pt"
INPUT_VIDEO  = "group_people.mp4"
OUTPUT_VIDEO = "output_activity.mp4"
CONF_THRESH  = 0.3
KP_CONF      = 0.3           # min keypoint confidence to use
SMOOTH_WIN   = 10            # frames for activity label smoothing

# COCO skeleton pairs
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# Colors
COLORS = [
    (0,255,0),(0,180,255),(255,100,0),(255,0,200),
    (0,255,200),(200,255,0),(100,0,255),(255,200,0),
]

# ── Keypoint indices (COCO) ────────────────────────────────────────────────────
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SHO=5; R_SHO=6; L_ELB=7; R_ELB=8; L_WRI=9; R_WRI=10
L_HIP=11; R_HIP=12; L_KNE=13; R_KNE=14; L_ANK=15; R_ANK=16


# ── Helpers ───────────────────────────────────────────────────────────────────
def angle_at_joint(a, joint, b):
    """Angle (degrees) at `joint` between vectors joint→a and joint→b."""
    v1 = a - joint
    v2 = b - joint
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-4 or n2 < 1e-4:
        return 180.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def get_kp(kps, kps_conf, idx):
    """Return keypoint if confident, else None."""
    if kps_conf[idx] >= KP_CONF:
        return kps[idx].astype(float)
    return None


def mid(a, b):
    if a is not None and b is not None:
        return (a + b) / 2
    return a if a is not None else b


# ── Activity classifier ───────────────────────────────────────────────────────
def classify_activity(kps, kps_conf):
    """
    Rule-based activity classification from 17 COCO keypoints.
    Returns a string label.
    """
    g = lambda i: get_kp(kps, kps_conf, i)

    nose      = g(NOSE)
    l_sho, r_sho = g(L_SHO), g(R_SHO)
    l_elb, r_elb = g(L_ELB), g(R_ELB)
    l_wri, r_wri = g(L_WRI), g(R_WRI)
    l_hip, r_hip = g(L_HIP), g(R_HIP)
    l_kne, r_kne = g(L_KNE), g(R_KNE)
    l_ank, r_ank = g(L_ANK), g(R_ANK)

    sho_mid = mid(l_sho, r_sho)
    hip_mid = mid(l_hip, r_hip)
    kne_mid = mid(l_kne, r_kne)
    ank_mid = mid(l_ank, r_ank)

    # ── body height (pixels, y increases downward) ────────────────────────────
    if sho_mid is not None and ank_mid is not None:
        body_h = max(ank_mid[1] - sho_mid[1], 1)
    elif sho_mid is not None and hip_mid is not None:
        body_h = max((hip_mid[1] - sho_mid[1]) * 2.5, 1)
    else:
        body_h = None

    # ── wrists above shoulders? ───────────────────────────────────────────────
    sho_y = sho_mid[1] if sho_mid is not None else None
    wrists_up = 0
    if sho_y is not None:
        if l_wri is not None and l_wri[1] < sho_y - 10:
            wrists_up += 1
        if r_wri is not None and r_wri[1] < sho_y - 10:
            wrists_up += 1

    # ── knee bend angle ───────────────────────────────────────────────────────
    knee_angles = []
    if l_hip is not None and l_kne is not None and l_ank is not None:
        knee_angles.append(angle_at_joint(l_hip, l_kne, l_ank))
    if r_hip is not None and r_kne is not None and r_ank is not None:
        knee_angles.append(angle_at_joint(r_hip, r_kne, r_ank))
    avg_knee_angle = np.mean(knee_angles) if knee_angles else 180.0

    # ── hip bend (torso lean) ─────────────────────────────────────────────────
    hip_angles = []
    if sho_mid is not None and hip_mid is not None and kne_mid is not None:
        hip_angles.append(angle_at_joint(sho_mid, hip_mid, kne_mid))
    avg_hip_angle = np.mean(hip_angles) if hip_angles else 180.0

    # ── vertical hip position relative to body ────────────────────────────────
    hip_ratio = None
    if sho_mid is not None and hip_mid is not None and body_h is not None:
        hip_ratio = (hip_mid[1] - sho_mid[1]) / body_h  # 0→hips at shoulder, 1→hips at ankle

    # ── ankle asymmetry (walking/running) ────────────────────────────────────
    ankle_asym = 0.0
    if l_ank is not None and r_ank is not None and body_h is not None:
        ankle_asym = abs(l_ank[1] - r_ank[1]) / body_h

    # ── elbow raise (arms extended sideways / overhead) ──────────────────────
    elbow_up = 0
    if sho_y is not None:
        if l_elb is not None and l_elb[1] < sho_y:
            elbow_up += 1
        if r_elb is not None and r_elb[1] < sho_y:
            elbow_up += 1

    # ────────────────────────────────────────────────────────────────────────
    # Decision rules (priority order)
    # ────────────────────────────────────────────────────────────────────────

    # 1. Jumping — both ankles high AND knees bent
    if ank_mid is not None and hip_mid is not None and body_h is not None:
        air_ratio = (hip_mid[1] - ank_mid[1]) / body_h
        if air_ratio < 0.25 and avg_knee_angle < 150:
            return "Jumping"

    # 2. Both hands raised overhead
    if wrists_up == 2:
        return "Hands Raised"

    # 3. One hand raised
    if wrists_up == 1:
        return "Hand Raised"

    # 4. Sitting — knees sharply bent, hips low
    if avg_knee_angle < 115 and (hip_ratio is None or hip_ratio < 0.45):
        return "Sitting"

    # 5. Crouching — significant knee bend, hips lower than normal standing
    if avg_knee_angle < 145 and hip_ratio is not None and hip_ratio < 0.35:
        return "Crouching"

    # 6. Bending forward — hip angle small (torso pitched forward)
    if avg_hip_angle < 140 and avg_knee_angle > 140:
        return "Bending"

    # 7. Walking/Running — asymmetric ankle heights
    if ankle_asym > 0.08:
        if ankle_asym > 0.18:
            return "Running"
        return "Walking"

    # 8. Default
    return "Standing"


# ── Caption drawing ───────────────────────────────────────────────────────────
def draw_caption(img, text, x1, y1, color):
    """Draw a filled-background caption above the bounding box."""
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 4
    tx = max(x1, 0)
    ty = max(y1 - th - pad * 2, th + pad)
    cv2.rectangle(img, (tx - pad, ty - th - pad),
                  (tx + tw + pad, ty + baseline + pad),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {INPUT_VIDEO}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

    # Per-track activity smoothing buffer  {track_id: deque of labels}
    label_history = defaultdict(lambda: deque(maxlen=SMOOTH_WIN))

    frame_idx = 0
    print(f"Processing {total} frames ({W}x{H} @ {fps:.0f} fps)…")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=CONF_THRESH, verbose=False)[0]

        boxes     = results.boxes
        keypoints = results.keypoints

        if boxes is not None and keypoints is not None:
            num_det = len(boxes)
            for i in range(num_det):
                # ── track id for smoothing ─────────────────────────────────
                tid = int(boxes.id[i]) if boxes.id is not None else i
                color = COLORS[tid % len(COLORS)]

                # ── bounding box ──────────────────────────────────────────
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                conf = float(boxes.conf[i])

                # ── keypoints ─────────────────────────────────────────────
                kps      = keypoints.xy[i].cpu().numpy()      # (17,2)
                kps_conf = keypoints.conf[i].cpu().numpy()    # (17,)

                # ── classify & smooth ─────────────────────────────────────
                raw_label = classify_activity(kps, kps_conf)
                label_history[tid].append(raw_label)
                # majority vote over recent window
                hist = label_history[tid]
                activity = max(set(hist), key=hist.count)

                # ── draw skeleton ─────────────────────────────────────────
                for a, b in SKELETON:
                    if kps_conf[a] >= KP_CONF and kps_conf[b] >= KP_CONF:
                        pa = (int(kps[a][0]), int(kps[a][1]))
                        pb = (int(kps[b][0]), int(kps[b][1]))
                        if pa != (0,0) and pb != (0,0):
                            cv2.line(frame, pa, pb, color, 2)

                for j in range(17):
                    if kps_conf[j] >= KP_CONF:
                        px, py = int(kps[j][0]), int(kps[j][1])
                        if (px, py) != (0, 0):
                            cv2.circle(frame, (px, py), 4, (255, 255, 255), -1)
                            cv2.circle(frame, (px, py), 4, color, 1)

                # ── bounding box ──────────────────────────────────────────
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # ── caption: ID + activity ────────────────────────────────
                caption = f"#{tid} {activity}"
                draw_caption(frame, caption, x1, y1, color)

        # ── frame counter overlay ─────────────────────────────────────────
        cv2.putText(frame, f"Frame {frame_idx+1}/{total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total} frames done")

    cap.release()
    out.release()
    print(f"\nDone! Output saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
