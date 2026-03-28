"""
PitchGuard AI — Model Training Script
Run this once to generate and save the XGBoost models before launching the app.

Usage:
    python train_models.py
    python train_models.py --samples 6000
    python train_models.py --samples 4000 --seed 99
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PitchGuard AI XGBoost biomechanics classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--quiet",   action="store_true")
    return parser.parse_args()


def check_dependencies() -> bool:
    missing = []
    for pkg, import_name in [
        ("xgboost",      "xgboost"),
        ("scikit-learn", "sklearn"),
        ("numpy",        "numpy"),
        ("joblib",       "joblib"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        return False
    return True


def print_banner():
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║        ⚾  PitchGuard AI                  ║")
    print("  ║   Biomechanics Model Training Script     ║")
    print("  ╚══════════════════════════════════════════╝")
    print()


def print_dataset_summary(X, y_out, y_risk):
    import numpy as np
    print("  ┌─────────────────────────────────────────┐")
    print("  │  Dataset Summary                        │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Total samples  : {len(X):>6}                  │")
    print(f"  │  Features       : {X.shape[1]:>6}                  │")
    print("  ├─────────────────────────────────────────┤")
    print("  │  Delivery class distribution:           │")
    for label in np.unique(y_out):
        count = int((y_out == label).sum())
        pct   = count / len(y_out) * 100
        bar   = "█" * int(pct / 5)
        print(f"  │  {label:<22} {count:>4} ({pct:5.1f}%) {bar:<10}│")
    print("  ├─────────────────────────────────────────┤")
    print("  │  Injury risk distribution:              │")
    for label in ["Low", "Medium", "High"]:
        count = int((y_risk == label).sum())
        pct   = count / len(y_risk) * 100
        bar   = "█" * int(pct / 5)
        print(f"  │  {label:<22} {count:>4} ({pct:5.1f}%) {bar:<10}│")
    print("  └─────────────────────────────────────────┘")
    print()


def _make_xgb_classifier(**kwargs):
    """
    FIX: XGBoost 2.0 removed the use_label_encoder parameter.
    This helper strips it so training works across both XGBoost 1.x and 2.x.
    """
    from xgboost import XGBClassifier
    kwargs.pop("use_label_encoder", None)
    return XGBClassifier(**kwargs)


def train(args: argparse.Namespace) -> int:
    verbose = not args.quiet

    if verbose:
        print_banner()

    if not check_dependencies():
        return 1

    import ml_models
    from pathlib import Path as _Path
    ml_models.MODELS_DIR   = _Path(args.out_dir)
    ml_models.OUTCOME_PATH = ml_models.MODELS_DIR / "outcome_model.pkl"
    ml_models.INJURY_PATH  = ml_models.MODELS_DIR / "injury_model.pkl"

    from ml_models import generate_synthetic_data, FEATURE_NAMES
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import numpy as np
    import joblib

    out_dir = _Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"⚙️  Generating {args.samples} synthetic samples (seed={args.seed})…")
    t0 = time.time()
    X, y_out, y_risk = generate_synthetic_data(n_samples=args.samples, random_seed=args.seed)
    if verbose:
        print(f"   Done in {time.time()-t0:.1f}s")
        print_dataset_summary(X, y_out, y_risk)

    # ── Outcome model ─────────────────────────────────────────────────────
    if verbose:
        print("🏋️  Training outcome classifier (Efficient / Mechanical_Leak / High_Risk_Mechanics)…")
    t1 = time.time()

    le_out    = LabelEncoder()
    y_out_enc = le_out.fit_transform(y_out)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_out_enc, test_size=0.2, random_state=42, stratify=y_out_enc
    )
    # FIX: use_label_encoder removed in XGBoost 2.0 — use _make_xgb_classifier
    clf_out = _make_xgb_classifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric="mlogloss",
    )
    clf_out.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred_out = clf_out.predict(X_te)
    acc_out    = accuracy_score(y_te, y_pred_out)

    if verbose:
        print(f"   Accuracy : {acc_out*100:.2f}%  ({time.time()-t1:.1f}s)")
        print()
        print("   Classification Report:")
        report = classification_report(y_te, y_pred_out, target_names=le_out.classes_, digits=3)
        for line in report.split("\n"):
            print(f"   {line}")
        print()

    joblib.dump({"model": clf_out, "encoder": le_out, "features": FEATURE_NAMES},
                out_dir / "outcome_model.pkl")
    if verbose:
        print(f"   💾 Saved → {out_dir / 'outcome_model.pkl'}")

    # ── Injury risk model ─────────────────────────────────────────────────
    if verbose:
        print("🏋️  Training injury risk classifier (Low / Medium / High)…")
    t2 = time.time()

    le_risk    = LabelEncoder()
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
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric="mlogloss",
    )
    clf_risk.fit(X_tr2, y_tr2, eval_set=[(X_te2, y_te2)], verbose=False)
    y_pred_risk = clf_risk.predict(X_te2)
    acc_risk    = accuracy_score(y_te2, y_pred_risk)

    if verbose:
        print(f"   Accuracy : {acc_risk*100:.2f}%  ({time.time()-t2:.1f}s)")
        print()
        print("   Classification Report:")
        report2 = classification_report(y_te2, y_pred_risk, target_names=le_risk.classes_, digits=3)
        for line in report2.split("\n"):
            print(f"   {line}")
        print()

    joblib.dump({"model": clf_risk, "encoder": le_risk, "features": FEATURE_NAMES},
                out_dir / "injury_model.pkl")
    if verbose:
        print(f"   💾 Saved → {out_dir / 'injury_model.pkl'}")

    # ── Feature importance summary ────────────────────────────────────────
    if verbose:
        print()
        print("  ┌─────────────────────────────────────────────┐")
        print("  │  Top-5 Feature Importances (Outcome Model)  │")
        print("  ├─────────────────────────────────────────────┤")
        imp_pairs = sorted(
            zip(FEATURE_NAMES, clf_out.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:5]
        for feat, imp in imp_pairs:
            bar = "█" * int(imp * 200)
            print(f"  │  {feat:<32} {imp:.4f}  {bar:<12}│")
        print("  └─────────────────────────────────────────────┘")
        print()

    total_time = time.time() - t0
    if verbose:
        print("  ╔══════════════════════════════════════════╗")
        print("  ║           ✅  Training Complete           ║")
        print("  ╠══════════════════════════════════════════╣")
        print(f"  ║  Outcome model accuracy   : {acc_out*100:6.2f}%     ║")
        print(f"  ║  Injury risk accuracy     : {acc_risk*100:6.2f}%      ║")
        print(f"  ║  Total time               : {total_time:6.1f}s      ║")
        print(f"  ║  Models saved to          : {str(out_dir):<16}║")
        print("  ╚══════════════════════════════════════════╝")
        print()
        print("  ▶ Now run:  streamlit run app.py")
        print()

    return 0


if __name__ == "__main__":
    args   = parse_args()
    result = train(args)
    sys.exit(result)