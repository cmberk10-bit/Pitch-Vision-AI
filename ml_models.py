"""
PitchGuard AI — ML Models Module
Synthetic data generation, XGBoost training, model persistence, and inference.
"""

import os
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ── Constants ────────────────────────────────────────────────────────────────
MODELS_DIR   = Path("models")
OUTCOME_PATH = MODELS_DIR / "outcome_model.pkl"
INJURY_PATH  = MODELS_DIR / "injury_model.pkl"

FEATURE_NAMES = [
    "right_elbow_flexion",
    "right_shoulder_abduction",
    "left_elbow_flexion",
    "left_knee_flexion",
    "right_knee_flexion",
    "hip_shoulder_separation",
    "trunk_tilt",
    "lumbar_spine_angle",
    "stride_length",
    "wrist_height_ratio",
    "elbow_height_ratio",
    "head_forward_lean",
    "shoulder_height_diff",
    "hip_drop_ratio",
]

# ── Realistic feature distributions per delivery class ───────────────────────
_EFFICIENT_PARAMS: Dict[str, Tuple[float, float]] = {
    "right_elbow_flexion":      (92.0,  4.5),
    "right_shoulder_abduction": (93.0,  4.0),
    "left_elbow_flexion":       (88.0,  8.0),
    "left_knee_flexion":        (45.0,  6.0),
    "right_knee_flexion":       (50.0,  6.5),
    "hip_shoulder_separation":  (48.0,  6.0),
    "trunk_tilt":               (31.0,  5.0),
    "lumbar_spine_angle":       (160.0, 8.0),
    "stride_length":            (0.85,  0.04),
    "wrist_height_ratio":       (0.22,  0.05),
    "elbow_height_ratio":       (0.05,  0.04),
    "head_forward_lean":        (0.08,  0.04),
    "shoulder_height_diff":     (0.06,  0.02),
    "hip_drop_ratio":           (0.95,  0.08),
}

_MECHANICAL_LEAK_PARAMS: Dict[str, Tuple[float, float]] = {
    "right_elbow_flexion":      (110.0, 8.0),
    "right_shoulder_abduction": (110.0, 7.0),
    "left_elbow_flexion":       (120.0, 12.0),
    "left_knee_flexion":        (25.0,  8.0),
    "right_knee_flexion":       (25.0,  8.0),
    "hip_shoulder_separation":  (25.0,  8.0),
    "trunk_tilt":               (48.0,  8.0),
    "lumbar_spine_angle":       (140.0, 10.0),
    "stride_length":            (0.65,  0.06),
    "wrist_height_ratio":       (0.05,  0.06),
    "elbow_height_ratio":       (0.22,  0.06),
    "head_forward_lean":        (0.28,  0.06),
    "shoulder_height_diff":     (0.18,  0.04),
    "hip_drop_ratio":           (1.20,  0.10),
}

_HIGH_RISK_PARAMS: Dict[str, Tuple[float, float]] = {
    "right_elbow_flexion":      (125.0, 10.0),
    "right_shoulder_abduction": (118.0, 9.0),
    "left_elbow_flexion":       (140.0, 15.0),
    "left_knee_flexion":        (15.0,  8.0),
    "right_knee_flexion":       (15.0,  8.0),
    "hip_shoulder_separation":  (12.0,  7.0),
    "trunk_tilt":               (58.0,  9.0),
    "lumbar_spine_angle":       (125.0, 12.0),
    "stride_length":            (0.50,  0.07),
    "wrist_height_ratio":       (-0.08, 0.07),
    "elbow_height_ratio":       (0.35,  0.07),
    "head_forward_lean":        (0.40,  0.07),
    "shoulder_height_diff":     (0.28,  0.05),
    "hip_drop_ratio":           (1.40,  0.12),
}

_CLASS_PARAMS = {
    "Efficient":           (_EFFICIENT_PARAMS,       0.60),
    "Mechanical_Leak":     (_MECHANICAL_LEAK_PARAMS, 0.30),
    "High_Risk_Mechanics": (_HIGH_RISK_PARAMS,       0.10),
}


def _compute_risk_index(sample: Dict[str, float]) -> float:
    """Replicate rule-based risk index without importing biomechanics at module level."""
    from biomechanics import OPTIMAL_RANGES, BODY_PART_LABELS, BODY_PART_WEIGHTS

    body_part_scores = {bp: [] for bp in BODY_PART_LABELS}

    for feat, val in sample.items():
        opt = OPTIMAL_RANGES.get(feat)
        if opt is None:
            continue
        lo, hi = opt["min"], opt["max"]
        if lo <= val <= hi:
            score = 0.0
        else:
            dev   = (lo - val) if val < lo else (val - hi)
            span  = (hi - lo) or 1.0
            ratio = dev / span
            score = min(100.0, 50.0 * ratio + 10.0 * ratio ** 2)

        for bp, feats in BODY_PART_LABELS.items():
            if feat in feats:
                body_part_scores[bp].append(score)

    bp_risks = {}
    for bp, scores in body_part_scores.items():
        if scores:
            bp_risks[bp] = min(96.0, float(np.mean(scores)) * BODY_PART_WEIGHTS.get(bp, 1.0))
        else:
            bp_risks[bp] = 0.0

    if not bp_risks:
        return 0.0

    total_w = sum(BODY_PART_WEIGHTS.get(bp, 1.0) for bp in bp_risks)
    risk    = sum(bp_risks[bp] * BODY_PART_WEIGHTS.get(bp, 1.0) for bp in bp_risks) / total_w
    return float(np.clip(risk, 0, 94))


def generate_synthetic_data(n_samples: int = 4000, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic pitcher biomechanics training data.
    Returns: X (n_samples, 14), y_out (outcome labels), y_risk (risk level labels).
    """
    rng = np.random.default_rng(random_seed)

    X_rows:   List[List[float]] = []
    y_out:    List[str]         = []
    y_risk:   List[str]         = []

    for class_label, (params, proportion) in _CLASS_PARAMS.items():
        n = int(n_samples * proportion)
        for _ in range(n):
            row = []
            sample_dict = {}
            for feat in FEATURE_NAMES:
                mean, std = params[feat]
                val = float(rng.normal(mean, std))
                val = _clamp_feature(feat, val)
                row.append(val)
                sample_dict[feat] = val
            X_rows.append(row)
            y_out.append(class_label)

            risk_idx = _compute_risk_index(sample_dict)
            if risk_idx < 33:
                y_risk.append("Low")
            elif risk_idx < 66:
                y_risk.append("Medium")
            else:
                y_risk.append("High")

    idx   = rng.permutation(len(X_rows))
    X     = np.array(X_rows, dtype=np.float32)[idx]
    y_out = np.array(y_out)[idx]
    y_risk= np.array(y_risk)[idx]

    return X, y_out, y_risk


def _clamp_feature(feat: str, val: float) -> float:
    _bounds = {
        "right_elbow_flexion":      (40.0, 180.0),
        "right_shoulder_abduction": (40.0, 180.0),
        "left_elbow_flexion":       (30.0, 180.0),
        "left_knee_flexion":        (0.0,  180.0),
        "right_knee_flexion":       (0.0,  180.0),
        "hip_shoulder_separation":  (0.0,  90.0),
        "trunk_tilt":               (0.0,  90.0),
        "lumbar_spine_angle":       (60.0, 180.0),
        "stride_length":            (0.2,  1.4),
        "wrist_height_ratio":       (-0.5, 0.8),
        "elbow_height_ratio":       (-0.3, 0.6),
        "head_forward_lean":        (-0.4, 0.6),
        "shoulder_height_diff":     (0.0,  0.5),
        "hip_drop_ratio":           (0.3,  1.8),
    }
    lo, hi = _bounds.get(feat, (-99, 99))
    return float(np.clip(val, lo, hi))


def _make_xgb_classifier(**kwargs):
    """
    Instantiate XGBClassifier, stripping params removed in XGBoost 2.0+.
    - use_label_encoder was removed in XGBoost 2.0
    """
    from xgboost import XGBClassifier
    # Remove deprecated/removed params so the code works across XGBoost 1.x and 2.x
    kwargs.pop("use_label_encoder", None)
    return XGBClassifier(**kwargs)


def train_models(n_samples: int = 4000, verbose: bool = True) -> None:
    """
    Train XGBoost outcome + injury risk classifiers and save to disk.
    Creates models/ directory if needed.
    """
    try:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError as e:
        raise ImportError(f"Training requires xgboost + scikit-learn: {e}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Generating synthetic training data…")
    X, y_out, y_risk = generate_synthetic_data(n_samples)

    # ── Outcome model ─────────────────────────────────────────────────────
    if verbose:
        print("Training outcome classifier…")
    le_out = LabelEncoder()
    y_out_enc = le_out.fit_transform(y_out)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_out_enc, test_size=0.2, random_state=42, stratify=y_out_enc
    )
    # FIX: use_label_encoder removed from XGBoost 2.0 — use _make_xgb_classifier
    clf_out = _make_xgb_classifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
    )
    clf_out.fit(X_tr, y_tr)
    acc_out = accuracy_score(y_te, clf_out.predict(X_te))
    if verbose:
        print(f"  Outcome model accuracy: {acc_out*100:.1f}%")

    joblib.dump(
        {"model": clf_out, "encoder": le_out, "features": FEATURE_NAMES},
        OUTCOME_PATH,
    )
    if verbose:
        print(f"  Saved → {OUTCOME_PATH}")

    # ── Injury risk model ─────────────────────────────────────────────────
    if verbose:
        print("Training injury risk classifier…")
    le_risk = LabelEncoder()
    y_risk_enc = le_risk.fit_transform(y_risk)

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X, y_risk_enc, test_size=0.2, random_state=42, stratify=y_risk_enc
    )
    clf_risk = _make_xgb_classifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
    )
    clf_risk.fit(X_tr2, y_tr2)
    acc_risk = accuracy_score(y_te2, clf_risk.predict(X_te2))
    if verbose:
        print(f"  Injury risk model accuracy: {acc_risk*100:.1f}%")

    joblib.dump(
        {"model": clf_risk, "encoder": le_risk, "features": FEATURE_NAMES},
        INJURY_PATH,
    )
    if verbose:
        print(f"  Saved → {INJURY_PATH}")
        print("✅ Training complete.")


# ── Model loading ─────────────────────────────────────────────────────────────
_model_cache: Optional[Dict] = None


def load_models() -> Optional[Dict]:
    """Load trained models from disk (with in-process caching). Returns None if unavailable."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not OUTCOME_PATH.exists() or not INJURY_PATH.exists():
        try:
            train_models(n_samples=3000, verbose=False)
        except Exception:
            return None

    try:
        outcome_bundle = joblib.load(OUTCOME_PATH)
        injury_bundle  = joblib.load(INJURY_PATH)
        _model_cache = {
            "outcome": outcome_bundle,
            "injury":  injury_bundle,
        }
        return _model_cache
    except Exception:
        return None


def predict_outcome(features: Dict[str, float], models: Optional[Dict]) -> Optional[Dict]:
    """Run XGBoost prediction and return a rich result dict."""
    if models is None:
        return None

    bundle = models.get("outcome")
    if bundle is None:
        return None

    clf        = bundle["model"]
    le         = bundle["encoder"]
    feat_names: List[str] = bundle["features"]

    try:
        x_vec = np.array([[features.get(f, 0.0) for f in feat_names]], dtype=np.float32)
    except Exception:
        return None

    proba  = clf.predict_proba(x_vec)[0]
    pred   = int(np.argmax(proba))
    label  = le.inverse_transform([pred])[0]

    class_probs = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}

    try:
        importances = clf.feature_importances_
        feat_imp = {feat_names[i]: float(importances[i]) for i in range(len(feat_names))}
    except Exception:
        feat_imp = {}

    return {
        "label":               label,
        "confidence":          float(proba[pred]),
        "class_probabilities": class_probs,
        "feature_importance":  feat_imp,
    }


def predict_injury_risk(features: Dict[str, float], models: Optional[Dict]) -> Optional[Dict]:
    """Run XGBoost injury risk prediction."""
    if models is None:
        return None

    bundle = models.get("injury")
    if bundle is None:
        return None

    clf        = bundle["model"]
    le         = bundle["encoder"]
    feat_names = bundle["features"]

    x_vec = np.array([[features.get(f, 0.0) for f in feat_names]], dtype=np.float32)
    proba = clf.predict_proba(x_vec)[0]
    pred  = int(np.argmax(proba))
    label = le.inverse_transform([pred])[0]

    class_probs = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}

    return {
        "label":               label,
        "confidence":          float(proba[pred]),
        "class_probabilities": class_probs,
    }