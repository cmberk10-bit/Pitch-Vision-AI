"""
PitchGuard AI — Biomechanics Engine
Pose extraction, feature computation, and injury risk assessment.
"""

import numpy as np
import cv2
import mediapipe as mp
from typing import Optional, Tuple, Dict, List, Callable

# ── MediaPipe setup ─────────────────────────────────────────────────────────
_mp_pose    = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils

# ── Landmark indices (MediaPipe 33-point model) ─────────────────────────────
LM = {
    "nose":            0,
    "left_eye":        2,
    "right_eye":       5,
    "left_ear":        7,
    "right_ear":       8,
    "left_shoulder":   11,
    "right_shoulder":  12,
    "left_elbow":      13,
    "right_elbow":     14,
    "left_wrist":      15,
    "right_wrist":     16,
    "left_hip":        23,
    "right_hip":       24,
    "left_knee":       25,
    "right_knee":      26,
    "left_ankle":      27,
    "right_ankle":     28,
}

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("nose",           "left_shoulder"),
    ("nose",           "right_shoulder"),
]

# ── OPTIMAL RANGES (ASMI + Driveline Research) ──────────────────────────────
# Sources:
#   • ASMI Clinical Biomechanics of Pitching (2009, 2018 updates)
#   • Driveline Baseball Research blog
#   • Glenn Fleisig et al., JESS 1999, 2011
#   • F. Escamilla et al., AJSM 1998
OPTIMAL_RANGES: Dict[str, Dict] = {
    "right_elbow_flexion": {
        "min": 85.0, "max": 100.0,
        "description": "Elbow angle at ball release",
        "unit": "degrees",
    },
    "right_shoulder_abduction": {
        "min": 85.0, "max": 100.0,
        "description": "Shoulder abduction at maximum external rotation",
        "unit": "degrees",
    },
    "left_elbow_flexion": {
        "min": 70.0, "max": 110.0,
        "description": "Glove-arm elbow flexion",
        "unit": "degrees",
    },
    "left_knee_flexion": {
        "min": 30.0, "max": 60.0,
        "description": "Stride-leg knee flexion at foot strike",
        "unit": "degrees",
    },
    "right_knee_flexion": {
        "min": 35.0, "max": 65.0,
        "description": "Drive-leg knee flexion at stride initiation",
        "unit": "degrees",
    },
    "hip_shoulder_separation": {
        "min": 40.0, "max": 65.0,
        "description": "Angle between hip and shoulder planes at MER",
        "unit": "degrees",
    },
    "trunk_tilt": {
        "min": 20.0, "max": 42.0,
        "description": "Forward trunk lean at ball release",
        "unit": "degrees",
    },
    "lumbar_spine_angle": {
        "min": 145.0, "max": 175.0,
        "description": "Lumbar spine extension/flexion angle",
        "unit": "degrees",
    },
    "stride_length": {
        "min": 0.75, "max": 0.95,
        "description": "Stride length normalised to height",
        "unit": "ratio",
    },
    "wrist_height_ratio": {
        "min": 0.10, "max": 0.40,
        "description": "Wrist height above shoulder (release point)",
        "unit": "ratio",
    },
    "elbow_height_ratio": {
        "min": -0.05, "max": 0.15,
        "description": "Elbow height relative to shoulder at release",
        "unit": "ratio",
    },
    "head_forward_lean": {
        "min": -0.10, "max": 0.20,
        "description": "Head forward displacement over hip",
        "unit": "ratio",
    },
    "shoulder_height_diff": {
        "min": 0.00, "max": 0.12,
        "description": "Shoulder tilt (height asymmetry)",
        "unit": "ratio",
    },
    "hip_drop_ratio": {
        "min": 0.80, "max": 1.10,
        "description": "Hip-to-shoulder vertical relationship",
        "unit": "ratio",
    },
}
# Add this below your OPTIMAL_RANGES dictionary
RADAR_FEATURES = [
    "right_elbow_flexion",
    "right_shoulder_abduction",
    "hip_shoulder_separation",
    "trunk_tilt",
    "stride_length",
    "left_knee_flexion"
]
# ── MLB Benchmarks (mean values from elite pitchers) ────────────────────────
MLB_BENCHMARKS: Dict[str, float] = {
    "right_elbow_flexion":      92.0,
    "right_shoulder_abduction": 93.0,
    "left_elbow_flexion":       90.0,
    "left_knee_flexion":        45.0,
    "right_knee_flexion":       50.0,
    "hip_shoulder_separation":  48.0,
    "trunk_tilt":               31.0,
    "lumbar_spine_angle":      160.0,
    "stride_length":             0.85,
    "wrist_height_ratio":        0.22,
    "elbow_height_ratio":        0.05,
    "head_forward_lean":         0.08,
    "shoulder_height_diff":      0.06,
    "hip_drop_ratio":            0.95,
}

# ── Body part groupings for risk display ────────────────────────────────────
BODY_PART_LABELS = {
    "Elbow":    ["right_elbow_flexion", "left_elbow_flexion", "elbow_height_ratio"],
    "Shoulder": ["right_shoulder_abduction", "shoulder_height_diff", "wrist_height_ratio"],
    "Knee":     ["left_knee_flexion", "right_knee_flexion"],
    "Hip":      ["hip_shoulder_separation", "hip_drop_ratio"],
    "Spine":    ["trunk_tilt", "lumbar_spine_angle", "head_forward_lean"],
    "Stride":   ["stride_length"],
}

# Body-part risk weights (injury relevance)
BODY_PART_WEIGHTS = {
    "Elbow":    1.4,   # UCL / medial epicondyle — highest injury site
    "Shoulder": 1.3,   # Labrum / RC
    "Knee":     0.9,
    "Hip":      1.0,
    "Spine":    1.1,
    "Stride":   0.8,
}

# Joint → body-part color mapping for overlay
JOINT_TO_BODY_PART = {
    "right_elbow":    "Elbow",
    "left_elbow":     "Elbow",
    "right_shoulder": "Shoulder",
    "left_shoulder":  "Shoulder",
    "left_knee":      "Knee",
    "right_knee":     "Knee",
    "left_hip":       "Hip",
    "right_hip":      "Hip",
    "nose":           "Spine",
    "left_wrist":     "Elbow",
    "right_wrist":    "Elbow",
    "left_ankle":     "Stride",
    "right_ankle":    "Stride",
}


# ── Vector math helpers ──────────────────────────────────────────────────────
def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a


def _angle(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    """Angle at `vertex` formed by points a-vertex-c, in degrees."""
    ba = a - vertex
    bc = c - vertex
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ── Landmark extraction ──────────────────────────────────────────────────────
def _landmarks_to_dict(pose_results, img_w: int, img_h: int) -> Optional[Dict[str, np.ndarray]]:
    """Convert MediaPipe pose results to a dict of {name: np.array([x_px, y_px, z])}."""
    if not pose_results or not pose_results.pose_landmarks:
        return None
    lms = pose_results.pose_landmarks.landmark
    out = {}
    for name, idx in LM.items():
        lm = lms[idx]
        out[name] = np.array([lm.x * img_w, lm.y * img_h, lm.z], dtype=float)
    return out


def extract_landmarks_from_image(
    img_bgr: np.ndarray,
    confidence: float = 0.5,
) -> Tuple[Optional[Dict], Optional[object]]:
    """Run MediaPipe on a single BGR image. Returns (landmark_dict, raw_pose_landmarks)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]

    with _mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=confidence,
    ) as pose:
        results = pose.process(img_rgb)

    lm_dict = _landmarks_to_dict(results, w, h)
    return lm_dict, results.pose_landmarks if results else None


def extract_landmarks_from_video(
    video_path: str,
    sample_rate: int = 3,
    confidence: float = 0.5,
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[Dict], List[np.ndarray], List[int]]:
    """
    Extract landmarks from a video at every `sample_rate` seconds.
    Returns (list_of_landmark_dicts, list_of_frames_bgr, list_of_frame_indices).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * sample_rate))

    all_lm_dicts: List[Dict]       = []
    all_frames:   List[np.ndarray] = []
    frame_indices: List[int]       = []

    pose = _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=confidence,
        min_tracking_confidence=confidence,
    )

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            h, w = frame.shape[:2]
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res  = pose.process(rgb)
            lm   = _landmarks_to_dict(res, w, h)
            if lm:
                all_lm_dicts.append(lm)
                all_frames.append(frame.copy())
                frame_indices.append(frame_num)

            if progress_callback and total_frames > 0:
                progress_callback(min(frame_num / total_frames, 1.0))

        frame_num += 1

    cap.release()
    pose.close()

    if progress_callback:
        progress_callback(1.0)

    return all_lm_dicts, all_frames, frame_indices


# ── Feature computation ──────────────────────────────────────────────────────
def compute_pitching_features(lm: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute 14 biomechanical features from a landmark dict."""
    # 2-D projections (x, y) for joint angles; include z for spine
    def p(name: str) -> np.ndarray:
        return lm[name][:2]   # pixel x, y

    def p3(name: str) -> np.ndarray:
        return lm[name]       # x, y, z

    # Torso length (reference for normalisation)
    torso_len = max(1.0, _dist(p("right_shoulder"), p("right_hip")))

    # Mid-points
    hip_mid = (p("left_hip") + p("right_hip")) / 2.0
    sho_mid = (p("left_shoulder") + p("right_shoulder")) / 2.0

    # ── 14 Features ──
    # 1. Right elbow flexion (shoulder → elbow → wrist)
    feat_right_elbow = _angle(p("right_shoulder"), p("right_elbow"), p("right_wrist"))

    # 2. Right shoulder abduction (right_hip → right_shoulder → right_elbow)
    feat_right_sho_abd = _angle(p("right_hip"), p("right_shoulder"), p("right_elbow"))

    # 3. Left elbow flexion
    feat_left_elbow = _angle(p("left_shoulder"), p("left_elbow"), p("left_wrist"))

    # 4. Left knee flexion (left_hip → left_knee → left_ankle)
    feat_left_knee = _angle(p("left_hip"), p("left_knee"), p("left_ankle"))

    # 5. Right knee flexion
    feat_right_knee = _angle(p("right_hip"), p("right_knee"), p("right_ankle"))

    # 6. Hip-shoulder separation (angle between hip vector and shoulder vector)
    hip_vec = p("right_hip") - p("left_hip")
    sho_vec = p("right_shoulder") - p("left_shoulder")
    dot     = np.dot(hip_vec, sho_vec)
    denom   = (np.linalg.norm(hip_vec) * np.linalg.norm(sho_vec)) + 1e-9
    cos_hs  = np.clip(dot / denom, -1.0, 1.0)
    feat_hip_sho_sep = float(np.degrees(np.arccos(cos_hs)))

    # 7. Trunk tilt (forward lean): arctan(dx/dy) from hip_mid to sho_mid
    dx = float(sho_mid[0] - hip_mid[0])
    dy = float(hip_mid[1] - sho_mid[1]) + 1e-9  # positive = sho above hip
    feat_trunk_tilt = float(np.degrees(np.arctan2(abs(dx), abs(dy))))

    # 8. Lumbar spine angle (left_hip → hip_mid → sho_mid as proxy)
    feat_lumbar = _angle(p("left_hip"), hip_mid, sho_mid)

    # 9. Stride length (ankle-to-ankle / torso)
    stride_px = _dist(p("left_ankle"), p("right_ankle"))
    feat_stride = stride_px / (torso_len + 1e-9)

    # 10. Wrist height ratio (sho_y - wrist_y) / torso
    feat_wrist_h = (p("right_shoulder")[1] - p("right_wrist")[1]) / torso_len

    # 11. Elbow height ratio (sho_y - elbow_y) / torso
    feat_elbow_h = (p("right_shoulder")[1] - p("right_elbow")[1]) / torso_len

    # 12. Head forward lean (nose_x - hip_mid_x) / torso
    feat_head = (p("nose")[0] - hip_mid[0]) / torso_len

    # 13. Shoulder height diff (|r_sho_y - l_sho_y|) / torso
    feat_sho_diff = abs(p("right_shoulder")[1] - p("left_shoulder")[1]) / torso_len

    # 14. Hip drop ratio (hip_mid_y - sho_mid_y) / torso
    feat_hip_drop = (hip_mid[1] - sho_mid[1]) / torso_len

    return {
        "right_elbow_flexion":      round(feat_right_elbow,    2),
        "right_shoulder_abduction": round(feat_right_sho_abd,  2),
        "left_elbow_flexion":       round(feat_left_elbow,     2),
        "left_knee_flexion":        round(feat_left_knee,      2),
        "right_knee_flexion":       round(feat_right_knee,     2),
        "hip_shoulder_separation":  round(feat_hip_sho_sep,    2),
        "trunk_tilt":               round(feat_trunk_tilt,     2),
        "lumbar_spine_angle":       round(feat_lumbar,         2),
        "stride_length":            round(feat_stride,         4),
        "wrist_height_ratio":       round(feat_wrist_h,        4),
        "elbow_height_ratio":       round(feat_elbow_h,        4),
        "head_forward_lean":        round(feat_head,           4),
        "shoulder_height_diff":     round(feat_sho_diff,       4),
        "hip_drop_ratio":           round(feat_hip_drop,       4),
    }


# ── Rule-based injury assessment ────────────────────────────────────────────
def _deviation_score(value: float, lo: float, hi: float) -> float:
    """Return a 0–100 risk score for a feature value outside [lo, hi]."""
    if lo <= value <= hi:
        return 0.0
    optimal_range = hi - lo
    if value < lo:
        dev = lo - value
    else:
        dev = value - hi
    # Scale: 0.5× range → 25 pts, 1× range → 50 pts, 2× range → 80 pts
    ratio = dev / (optimal_range + 1e-9)
    score = min(100.0, 50.0 * ratio + 10.0 * ratio ** 2)
    return float(np.clip(score, 0.0, 100.0))


# Warning message templates
_WARNING_TEMPLATES = {
    "right_elbow_flexion":      "⚠️ Elbow flexion {val:.0f}° (optimal: {lo:.0f}–{hi:.0f}°). Risk of UCL stress / medial epicondyle overload.",
    "right_shoulder_abduction": "⚠️ Shoulder abduction {val:.0f}° (optimal: {lo:.0f}–{hi:.0f}°). Risk of labrum / rotator cuff impingement.",
    "left_elbow_flexion":       "⚠️ Glove-arm elbow {val:.0f}° (optimal: {lo:.0f}–{hi:.0f}°). Imbalanced deceleration, shoulder stress.",
    "left_knee_flexion":        "⚠️ Stride-leg knee {val:.0f}° (optimal: {lo:.0f}–{hi:.0f}°). Reduced energy transfer, knee ligament stress.",
    "right_knee_flexion":       "⚠️ Drive-leg knee {val:.0f}° (optimal: {lo:.0f}–{hi:.0f}°). Reduced power generation, patellofemoral stress.",
    "hip_shoulder_separation":  "⚠️ Hip–shoulder separation {val:.0f}° (optimal: {lo:.0f}–{hi:.0f}°). Reduced velocity; extra arm stress to compensate.",
    "trunk_tilt":               "⚠️ Trunk tilt {val:.0f}° (optimal: {lo:.0f}–{hi:.0f}°). Altered release point, increased lumbar load.",
    "lumbar_spine_angle":       "⚠️ Lumbar angle {val:.0f}° (optimal: {lo:.0f}–{hi:.0f}°). Excessive spinal flexion/extension risk.",
    "stride_length":            "⚠️ Stride length ratio {val:.2f} (optimal: {lo:.2f}–{hi:.2f}). Affects control and lower-body energy transfer.",
    "wrist_height_ratio":       "⚠️ Wrist height ratio {val:.2f} (optimal: {lo:.2f}–{hi:.2f}). Release point inconsistency detected.",
    "elbow_height_ratio":       "⚠️ Elbow height ratio {val:.2f} (optimal: {lo:.2f}–{hi:.2f}). Risk of dropped elbow, UCL stress.",
    "head_forward_lean":        "⚠️ Head lean ratio {val:.2f} (optimal: {lo:.2f}–{hi:.2f}). Balance and control issue.",
    "shoulder_height_diff":     "⚠️ Shoulder height diff {val:.2f} (optimal: {lo:.2f}–{hi:.2f}). Lateral tilt may stress shoulder capsule.",
    "hip_drop_ratio":           "⚠️ Hip drop ratio {val:.2f} (optimal: {lo:.2f}–{hi:.2f}). Glute/hip weakness indicator.",
}

_RISK_COLORS = {
    "ok":      "#00c853",   # green
    "warn":    "#ffab00",   # amber
    "danger":  "#ff5252",   # red
}


def rule_based_injury_assessment(features: Dict[str, float]) -> Dict:
    """
    Assess injury risk from features.
    Returns dict with overall_risk, risk_index, warnings, body_part_risks, joint_colors.
    """
    warnings:        List[str]         = []
    feature_scores:  Dict[str, float]  = {}
    body_part_scores: Dict[str, List[float]] = {bp: [] for bp in BODY_PART_LABELS}

    for feat, val in features.items():
        opt = OPTIMAL_RANGES.get(feat)
        if opt is None:
            continue
        lo, hi = opt["min"], opt["max"]
        score  = _deviation_score(val, lo, hi)
        feature_scores[feat] = score

        # Warnings for significant deviations
        if score >= 15.0:
            tmpl = _WARNING_TEMPLATES.get(feat, "")
            if tmpl:
                warnings.append(tmpl.format(val=val, lo=lo, hi=hi))

        # Assign to body parts
        for bp, feats in BODY_PART_LABELS.items():
            if feat in feats:
                body_part_scores[bp].append(score)

    # Body part risk (weighted mean)
    body_part_risks: Dict[str, float] = {}
    for bp, scores in body_part_scores.items():
        if scores:
            raw  = float(np.mean(scores))
            body_part_risks[bp] = min(96.0, raw * BODY_PART_WEIGHTS.get(bp, 1.0))
        else:
            body_part_risks[bp] = 0.0

    # Overall risk index (weighted mean across body parts)
    if body_part_risks:
        total_w = sum(BODY_PART_WEIGHTS.get(bp, 1.0) for bp in body_part_risks)
        risk_index = sum(
            body_part_risks[bp] * BODY_PART_WEIGHTS.get(bp, 1.0)
            for bp in body_part_risks
        ) / total_w
        risk_index = float(np.clip(risk_index, 0, 94))
    else:
        risk_index = 0.0

    # Overall risk level
    if risk_index < 33:
        overall_risk = "Low"
    elif risk_index < 66:
        overall_risk = "Medium"
    else:
        overall_risk = "High"

    # Joint colors for overlay
    joint_colors: Dict[str, str] = {}
    for joint_name in LM.keys():
        bp = JOINT_TO_BODY_PART.get(joint_name, "Spine")
        bp_score = body_part_risks.get(bp, 0.0)
        if bp_score < 25:
            joint_colors[joint_name] = _RISK_COLORS["ok"]
        elif bp_score < 55:
            joint_colors[joint_name] = _RISK_COLORS["warn"]
        else:
            joint_colors[joint_name] = _RISK_COLORS["danger"]

    return {
        "overall_risk":    overall_risk,
        "risk_index":      int(risk_index),
        "warnings":        warnings,
        "body_part_risks": body_part_risks,
        "joint_colors":    joint_colors,
        "feature_scores":  feature_scores,
    }


# ── Pose overlay drawing ─────────────────────────────────────────────────────
def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert #rrggbb to (B, G, R)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_pose_on_image(
    img_bgr: np.ndarray,
    pose_landmarks,      # MediaPipe pose_landmarks (or None)
    joint_colors: Dict[str, str],
) -> np.ndarray:
    """
    Draw skeleton connections and joint circles on the image.
    joint_colors: {joint_name: hex_color}
    """
    h, w = img_bgr.shape[:2]

    # If we have pose_landmarks from MediaPipe, use pixel coords
    pts: Dict[str, Tuple[int, int]] = {}
    if pose_landmarks:
        lms = pose_landmarks.landmark
        for name, idx in LM.items():
            lm = lms[idx]
            pts[name] = (int(lm.x * w), int(lm.y * h))

    if not pts:
        return img_bgr

    # Draw connections (semi-transparent lines)
    overlay = img_bgr.copy()
    for (a, b) in SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            # Pick line color based on worst joint
            col_a = joint_colors.get(a, "#00c853")
            col_b = joint_colors.get(b, "#00c853")
            # Use higher-risk color
            score_a = _color_to_score(col_a)
            score_b = _color_to_score(col_b)
            line_col = col_a if score_a >= score_b else col_b
            cv2.line(overlay, pts[a], pts[b], _hex_to_bgr(line_col), 3, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.7, img_bgr, 0.3, 0, img_bgr)

    # Draw joint dots
    for name, (px, py) in pts.items():
        color_hex = joint_colors.get(name, "#00c853")
        color_bgr = _hex_to_bgr(color_hex)
        cv2.circle(img_bgr, (px, py), 7, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(img_bgr, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)  # white outline

    return img_bgr


def _color_to_score(hex_color: str) -> int:
    """Return 0 / 1 / 2 for green / amber / red."""
    mapping = {"#00c853": 0, "#ffab00": 1, "#ff5252": 2}
    return mapping.get(hex_color, 0)
