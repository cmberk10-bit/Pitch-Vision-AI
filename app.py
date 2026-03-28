"""
PitchGuard AI — Baseball Pitcher Biomechanics Analyzer
Main Streamlit Application
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import json
import time
from PIL import Image
import io
import base64
from pathlib import Path

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Pitch Vision AI",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ────────────────────────────────────────────────────────────
from biomechanics import (
    extract_landmarks_from_image,
    extract_landmarks_from_video,
    compute_pitching_features,
    rule_based_injury_assessment,
    draw_pose_on_image,
    OPTIMAL_RANGES,
    MLB_BENCHMARKS,
    BODY_PART_LABELS,
)
from visualizations import (
    create_risk_gauge,
    create_body_part_risk_chart,
    create_time_series_chart,
    create_feature_radar,
    create_per_frame_risk_trend,
    create_feature_importance_chart,
)
from ml_models import load_models, predict_outcome
from coaching import generate_coaching_plan

# ── CSS injection ────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif !important;
        }

        .stApp { background-color: #0d0d1a !important; }
        section[data-testid="stSidebar"] {
            background-color: #13131f !important;
            border-right: 1px solid #2a2a3e;
        }

        .pg-hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
        .pg-hero h1 {
            font-size: 3.2rem; font-weight: 700;
            background: linear-gradient(135deg, #7c4dff 0%, #00e5ff 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; letter-spacing: -1px; margin-bottom: 0.4rem;
        }
        .pg-hero p { color: #888888; font-size: 1.05rem; margin-top: 0; }

        .pg-card {
            background: #13131f; border: 1px solid #2a2a3e;
            border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem;
        }
        .pg-metric-card {
            background: #13131f; border: 1px solid #2a2a3e;
            border-radius: 10px; padding: 1.1rem 1.2rem; text-align: center;
        }
        .pg-metric-card .metric-label {
            font-size: 0.75rem; text-transform: uppercase;
            letter-spacing: 1.5px; color: #888888; margin-bottom: 0.3rem;
        }
        .pg-metric-card .metric-value {
            font-size: 2rem; font-weight: 700; color: #e0e0e0; line-height: 1;
        }
        .pg-metric-card .metric-sub { font-size: 0.8rem; color: #7c4dff; margin-top: 0.25rem; }

        .pg-section-header {
            font-size: 1rem; font-weight: 600; color: #e0e0e0;
            text-transform: uppercase; letter-spacing: 2px;
            border-bottom: 1px solid #2a2a3e;
            padding-bottom: 0.5rem; margin: 1.5rem 0 1rem;
        }

        .badge-low {
            display: inline-block; background: rgba(0,200,83,0.15);
            color: #00c853; border: 1px solid #00c853;
            border-radius: 20px; padding: 0.2rem 0.8rem;
            font-size: 0.8rem; font-weight: 600;
        }
        .badge-medium {
            display: inline-block; background: rgba(255,171,0,0.15);
            color: #ffab00; border: 1px solid #ffab00;
            border-radius: 20px; padding: 0.2rem 0.8rem;
            font-size: 0.8rem; font-weight: 600;
        }
        .badge-high {
            display: inline-block; background: rgba(213,0,0,0.15);
            color: #ff5252; border: 1px solid #ff5252;
            border-radius: 20px; padding: 0.2rem 0.8rem;
            font-size: 0.8rem; font-weight: 600;
        }

        .coach-report {
            background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
            border: 1px solid #7c4dff44; border-radius: 12px;
            padding: 1.8rem 2rem; color: #e0e0e0; line-height: 1.7;
        }
        .coach-report h3 {
            color: #7c4dff; font-size: 1rem; text-transform: uppercase;
            letter-spacing: 1.5px; margin-top: 1.5rem; margin-bottom: 0.5rem;
            border-left: 3px solid #7c4dff; padding-left: 0.6rem;
        }
        .coach-report ul { padding-left: 1.4rem; }
        .coach-report li { margin-bottom: 0.4rem; color: #c0c0d0; }
        .coach-report strong { color: #00e5ff; }

        .frame-caption {
            font-size: 0.72rem; font-family: 'JetBrains Mono', monospace;
            text-align: center; padding: 0.25rem 0.5rem;
            border-radius: 4px; margin-top: 4px;
        }
        .frame-cap-low  { color: #00c853; background: rgba(0,200,83,0.12); }
        .frame-cap-med  { color: #ffab00; background: rgba(255,171,0,0.12); }
        .frame-cap-high { color: #ff5252; background: rgba(213,0,0,0.12); }

        .pg-warning {
            background: rgba(255,171,0,0.08); border-left: 3px solid #ffab00;
            border-radius: 0 6px 6px 0; padding: 0.5rem 0.8rem;
            margin-bottom: 0.4rem; color: #e0c060; font-size: 0.88rem;
        }

        .sidebar-logo {
            font-size: 1.6rem; font-weight: 700;
            background: linear-gradient(135deg, #7c4dff 0%, #00e5ff 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 0.2rem;
        }
        .sidebar-tagline {
            font-size: 0.72rem; color: #555577;
            text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1.2rem;
        }

        .stButton > button {
            background: linear-gradient(135deg, #7c4dff 0%, #4527a0 100%) !important;
            color: white !important; border: none !important;
            border-radius: 8px !important; font-weight: 600 !important;
            letter-spacing: 0.5px !important; transition: opacity 0.2s !important;
        }
        .stButton > button:hover { opacity: 0.88 !important; }

        .stDataFrame { border-radius: 8px; overflow: hidden; }

        .stTabs [data-baseweb="tab"] { color: #888888 !important; font-weight: 500; }
        .stTabs [aria-selected="true"] {
            color: #7c4dff !important; border-bottom-color: #7c4dff !important;
        }

        .pitcher-compare {
            background: #13131f; border: 1px solid #2a2a3e;
            border-radius: 10px; padding: 1rem; text-align: center;
        }
        .pitcher-compare .p-name { font-size: 0.95rem; font-weight: 600; color: #00e5ff; }
        .pitcher-compare .p-stat { font-size: 0.78rem; color: #888888; margin-top: 0.2rem; }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0d0d1a; }
        ::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #7c4dff66; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">⚾ Pitch Vision AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">Biomechanics · Injury Prevention</div>', unsafe_allow_html=True)
        st.divider()

        default_key = os.environ.get("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "🔑 Gemini API Key",
            value=default_key,
            type="password",
            help="Get a free key at aistudio.google.com",
            placeholder="AIza...",
        )

        st.divider()

        mode = st.radio(
            "📊 Analysis Mode",
            ["🖼️ Image Upload", "🎬 Video Upload", "📷 Live Webcam"],
            index=0,
        )

        st.divider()

        sample_rate = st.slider("🎞️ Frame Sampling Rate (Video)", 1, 10, 3,
                                help="Extract 1 frame every N seconds")

        with st.expander("⚙️ Advanced Options"):
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
            max_frames_display   = st.slider("Max Frames to Display", 4, 24, 12)
            show_radar           = st.checkbox("Show Radar Chart", True)
            show_feature_imp     = st.checkbox("Show Feature Importance", True)

        st.divider()

        with st.expander("ℹ️ How risk is calculated"):
            st.markdown(
                """
                <small style='color:#888'>
                Risk is computed from 14 biomechanical features using ASMI & Driveline
                Research validated ranges. Each feature deviation is scored 0–100 and
                weighted by injury relevance. The overall risk index is a weighted mean.
                </small>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        models = load_models()
        if models is None:
            st.warning("⚠️ Models not trained. Run `train_models.py` first.")
        else:
            st.success("✅ ML Models loaded")

    mode_clean = mode.split(" ", 1)[1].strip()
    return mode_clean, api_key, sample_rate, confidence_threshold, max_frames_display, show_radar, show_feature_imp


# ── Helpers ──────────────────────────────────────────────────────────────────
def _badge(level: str) -> str:
    cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(level, "badge-low")
    return f'<span class="{cls}">{level} Risk</span>'


def _risk_color(score: float) -> str:
    if score < 33:
        return "#00c853"
    elif score < 66:
        return "#ffab00"
    return "#ff5252"


def render_metrics_row(features: dict, injury: dict, outcome_result: dict):
    c1, c2, c3 = st.columns(3)
    risk_idx = injury.get("risk_index", 0)
    risk_lvl = injury.get("overall_risk", "Low")
    delivery  = outcome_result.get("label", "Unknown") if outcome_result else "N/A"
    conf      = outcome_result.get("confidence", 0.0) if outcome_result else 0.0

    color = _risk_color(risk_idx)
    with c1:
        st.markdown(
            f"""<div class="pg-metric-card">
                <div class="metric-label">Delivery</div>
                <div class="metric-value" style="font-size:1.3rem">{delivery.replace('_',' ')}</div>
                <div class="metric-sub">{conf*100:.0f}% confidence</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="pg-metric-card">
                <div class="metric-label">Injury Risk Index</div>
                <div class="metric-value" style="color:{color}">{risk_idx}</div>
                <div class="metric-sub">{_badge(risk_lvl)}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        hs = features.get("hip_shoulder_separation", 0)
        st.markdown(
            f"""<div class="pg-metric-card">
                <div class="metric-label">Hip-Shoulder Sep.</div>
                <div class="metric-value">{hs:.1f}°</div>
                <div class="metric-sub">Key velocity driver</div>
            </div>""",
            unsafe_allow_html=True,
        )


def render_warnings(warnings: list):
    if not warnings:
        st.success("✅ No significant biomechanical warnings detected.")
        return
    st.markdown('<div class="pg-section-header">⚠️ Active Warnings</div>', unsafe_allow_html=True)
    for w in warnings:
        st.markdown(f'<div class="pg-warning">{w}</div>', unsafe_allow_html=True)


def render_biomechanics_table(features: dict):
    import pandas as pd
    rows = []
    for feat, val in features.items():
        opt = OPTIMAL_RANGES.get(feat, {})
        mlb = MLB_BENCHMARKS.get(feat, "—")
        lo, hi = opt.get("min", None), opt.get("max", None)
        if lo is not None and hi is not None:
            in_range = lo <= val <= hi
            status = "✅ OK" if in_range else "⚠️ Out of Range"
            target = f"{lo}–{hi}"
        else:
            status = "—"
            target = "—"
        rows.append({
            "Feature":   feat.replace("_", " ").title(),
            "Value":     f"{val:.2f}",
            "Target":    target,
            "MLB Avg":   str(mlb),
            "Status":    status,
        })
    df = __import__("pandas").DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


MLB_ELITE = {
    "Gerrit Cole": {
        "hip_shoulder_separation": 52, "right_elbow_flexion": 92,
        "trunk_tilt": 30, "stride_length": 0.88, "era": "2.63", "k9": "13.1",
    },
    "Jacob deGrom": {
        "hip_shoulder_separation": 55, "right_elbow_flexion": 90,
        "trunk_tilt": 32, "stride_length": 0.90, "era": "2.52", "k9": "14.3",
    },
    "Max Scherzer": {
        "hip_shoulder_separation": 50, "right_elbow_flexion": 88,
        "trunk_tilt": 28, "stride_length": 0.85, "era": "2.90", "k9": "11.2",
    },
}


def render_elite_comparison(features: dict):
    st.markdown('<div class="pg-section-header">🏆 Elite Pitcher Comparison</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    key_features = ["hip_shoulder_separation", "right_elbow_flexion", "trunk_tilt", "stride_length"]
    for i, (name, stats) in enumerate(MLB_ELITE.items()):
        with cols[i]:
            diffs = []
            for kf in key_features:
                pitcher_val = features.get(kf, 0)
                elite_val   = stats.get(kf, 0)
                diffs.append(pitcher_val - elite_val)
            avg_diff = np.mean([abs(d) for d in diffs])
            similarity = max(0, 100 - avg_diff * 2)

            st.markdown(
                f"""<div class="pitcher-compare">
                    <div class="p-name">{name}</div>
                    <div class="p-stat">ERA {stats['era']} · K/9 {stats['k9']}</div>
                    <div style="font-size:1.4rem; font-weight:700;
                        color:{'#00c853' if similarity>70 else '#ffab00' if similarity>40 else '#ff5252'}">
                        {similarity:.0f}%
                    </div>
                    <div class="p-stat">Mechanics Similarity</div>
                </div>""",
                unsafe_allow_html=True,
            )


def render_coaching_section(features, injury, outcome_result, api_key):
    st.markdown('<div class="pg-section-header">🤖 AI Coaching Report</div>', unsafe_allow_html=True)
    if st.button("✨ Generate AI Coaching Plan", use_container_width=True):
        if not api_key:
            st.error("❌ Please enter your Gemini API key in the sidebar.")
            return
        with st.spinner("Generating personalized coaching plan..."):
            # FIX: wrap in try/except so Gemini API errors surface as clean st.error
            try:
                report_html = generate_coaching_plan(features, injury, outcome_result, api_key)
            except ValueError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"❌ Unexpected error generating coaching plan: {e}")
                return
        if report_html:
            st.markdown(f'<div class="coach-report">{report_html}</div>', unsafe_allow_html=True)


# ── Image Mode ───────────────────────────────────────────────────────────────
def image_mode(api_key, show_radar, show_feature_imp, confidence_threshold):
    st.markdown('<div class="pg-section-header">🖼️ Upload Pitcher Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a JPG/PNG of a pitcher at ball-release", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("📤 Upload an image to begin biomechanical analysis.")
        return

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("❌ Could not decode image.")
        return

    with st.spinner("🔬 Detecting pose and computing biomechanics..."):
        landmarks, pose_lms = extract_landmarks_from_image(img_bgr, confidence_threshold)

    if landmarks is None:
        st.error("❌ No pitcher pose detected. Try a clearer image with the full body visible.")
        return

    features = compute_pitching_features(landmarks)
    injury   = rule_based_injury_assessment(features)
    models   = load_models()
    outcome_result = predict_outcome(features, models) if models else None

    annotated = draw_pose_on_image(img_bgr.copy(), pose_lms, injury.get("joint_colors", {}))
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    col_img, col_charts = st.columns([1.4, 1])
    with col_img:
        # FIX: use_column_width deprecated in Streamlit 1.35 → use_container_width
        st.image(annotated_rgb, caption="Pose Overlay — joints colored by risk", use_container_width=True)
    with col_charts:
        fig_gauge = create_risk_gauge(injury["risk_index"])
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        fig_bar = create_body_part_risk_chart(injury["body_part_risks"])
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    render_metrics_row(features, injury, outcome_result)
    st.markdown("<br>", unsafe_allow_html=True)
    render_warnings(injury["warnings"])
    render_elite_comparison(features)

    if show_radar:
        st.markdown('<div class="pg-section-header">📡 Mechanics Radar vs MLB Benchmarks</div>', unsafe_allow_html=True)
        fig_radar = create_feature_radar(features, MLB_BENCHMARKS)
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    with st.expander("📊 Detailed Biomechanics Table"):
        render_biomechanics_table(features)

    if show_feature_imp and outcome_result:
        with st.expander("📈 Feature Importance"):
            fig_imp = create_feature_importance_chart(outcome_result.get("feature_importance", {}))
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("🔧 Raw Model Outputs"):
        raw = {
            "features":       features,
            "injury_result":  {k: v for k, v in injury.items() if k != "joint_colors"},
            "outcome_result": outcome_result,
        }
        st.json(raw)

    render_coaching_section(features, injury, outcome_result, api_key)
    _save_to_history(annotated_rgb, injury, outcome_result)
    _render_history()


# ── Video Mode ───────────────────────────────────────────────────────────────
def video_mode(api_key, sample_rate, max_frames_display, show_radar, show_feature_imp, confidence_threshold):
    st.markdown('<div class="pg-section-header">🎬 Upload Pitcher Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop an MP4/MOV/AVI video of a pitch", type=["mp4", "mov", "avi"])

    if uploaded is None:
        st.info("📤 Upload a video to begin frame-by-frame analysis.")
        return

    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    progress_bar = st.progress(0, text="Extracting frames…")

    try:
        with st.spinner("🎞️ Processing video frames…"):
            all_landmarks, all_frames, frame_indices = extract_landmarks_from_video(
                tmp_path, sample_rate, confidence_threshold,
                progress_callback=lambda p: progress_bar.progress(
                    int(p * 100), text=f"Processing frame {int(p*100)}%"
                )
            )
    finally:
        progress_bar.empty()

    if not all_landmarks:
        st.error("❌ No poses detected in the video. Ensure the pitcher is clearly visible.")
        os.unlink(tmp_path)
        return

    per_frame_features = [compute_pitching_features(lm) for lm in all_landmarks]
    per_frame_injuries = [rule_based_injury_assessment(f) for f in per_frame_features]

    agg_features = {}
    for key in per_frame_features[0].keys():
        vals = [f[key] for f in per_frame_features]
        agg_features[key] = float(np.mean(vals))

    agg_injury = rule_based_injury_assessment(agg_features)
    models = load_models()
    outcome_result = predict_outcome(agg_features, models) if models else None

    col_vid, col_charts = st.columns([1.4, 1])
    with col_vid:
        st.video(tmp_path)
    with col_charts:
        fig_gauge = create_risk_gauge(agg_injury["risk_index"])
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        fig_bar = create_body_part_risk_chart(agg_injury["body_part_risks"])
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    render_metrics_row(agg_features, agg_injury, outcome_result)
    st.markdown("<br>", unsafe_allow_html=True)
    render_warnings(agg_injury["warnings"])

    st.markdown('<div class="pg-section-header">🎞️ Analyzed Frames</div>', unsafe_allow_html=True)
    display_count = min(max_frames_display, len(all_frames))
    cols_per_row  = 4
    for row_start in range(0, display_count, cols_per_row):
        row_cols = st.columns(cols_per_row)
        for ci, fi in enumerate(range(row_start, min(row_start + cols_per_row, display_count))):
            frame_bgr     = all_frames[fi]
            inj           = per_frame_injuries[fi]
            joint_colors  = inj.get("joint_colors", {})
            annotated     = draw_pose_on_image(frame_bgr.copy(), None, joint_colors)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            risk_val      = inj["risk_index"]
            risk_lvl      = inj["overall_risk"]
            cap_cls = {"Low": "frame-cap-low", "Medium": "frame-cap-med", "High": "frame-cap-high"}.get(risk_lvl, "frame-cap-low")
            with row_cols[ci]:
                # FIX: use_container_width replaces deprecated use_column_width
                st.image(annotated_rgb, use_container_width=True)
                actual_frame_num = frame_indices[fi] if fi < len(frame_indices) else fi
                st.markdown(
                    f'<div class="frame-caption {cap_cls}">Frame {actual_frame_num} · Risk {risk_val}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="pg-section-header">📉 Temporal Analysis</div>', unsafe_allow_html=True)
    col_ts, col_trend = st.columns(2)
    feature_keys = list(per_frame_features[0].keys())
    with col_ts:
        selected_metrics = st.multiselect(
            "Select metrics to plot",
            feature_keys,
            default=["right_elbow_flexion", "hip_shoulder_separation", "trunk_tilt"],
        )
        if selected_metrics:
            fig_ts = create_time_series_chart(per_frame_features, selected_metrics)
            st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False})
    with col_trend:
        fig_trend = create_per_frame_risk_trend(per_frame_injuries)
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    if show_radar:
        st.markdown('<div class="pg-section-header">📡 Mechanics Radar vs MLB Benchmarks</div>', unsafe_allow_html=True)
        fig_radar = create_feature_radar(agg_features, MLB_BENCHMARKS)
        st.plotly_chart(fig_radar, use_container_width=True)

    with st.expander("📊 Detailed Biomechanics Table (Aggregated)"):
        render_biomechanics_table(agg_features)

    if show_feature_imp and outcome_result:
        with st.expander("📈 Feature Importance"):
            fig_imp = create_feature_importance_chart(outcome_result.get("feature_importance", {}))
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True)

    with st.expander("🔧 Raw Model Outputs"):
        st.json({
            "aggregated_features": agg_features,
            "injury_result":       {k: v for k, v in agg_injury.items() if k != "joint_colors"},
            "outcome_result":      outcome_result,
            "frame_count":         len(all_frames),
        })

    render_coaching_section(agg_features, agg_injury, outcome_result, api_key)

    # FIX: Save to history from video mode too (use first annotated frame)
    if all_frames:
        first_annotated = cv2.cvtColor(
            draw_pose_on_image(all_frames[0].copy(), None, agg_injury.get("joint_colors", {})),
            cv2.COLOR_BGR2RGB,
        )
        _save_to_history(first_annotated, agg_injury, outcome_result)
    _render_history()

    try:
        os.unlink(tmp_path)
    except Exception:
        pass


# ── Webcam Mode ──────────────────────────────────────────────────────────────
def webcam_mode(api_key, confidence_threshold, show_radar):
    st.markdown('<div class="pg-section-header">📷 Live Webcam Capture</div>', unsafe_allow_html=True)
    st.info("💡 Get into your pitching stance and capture a frame for real-time analysis.")

    captured = st.camera_input("📸 Capture frame for analysis")

    if captured is None:
        return

    file_bytes = np.frombuffer(captured.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("❌ Could not decode webcam image.")
        return

    with st.spinner("🔬 Real-time biomechanical analysis…"):
        landmarks, pose_lms = extract_landmarks_from_image(img_bgr, confidence_threshold)

    if landmarks is None:
        st.error("❌ No pose detected. Ensure your full body is visible in frame.")
        return

    features = compute_pitching_features(landmarks)
    injury   = rule_based_injury_assessment(features)
    models   = load_models()
    outcome_result = predict_outcome(features, models) if models else None

    annotated     = draw_pose_on_image(img_bgr.copy(), pose_lms, injury.get("joint_colors", {}))
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        # FIX: use_container_width replaces deprecated use_column_width
        st.image(annotated_rgb, caption="Live Pose Overlay", use_container_width=True)
    with col2:
        fig_gauge = create_risk_gauge(injury["risk_index"])
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    render_metrics_row(features, injury, outcome_result)
    st.markdown("<br>", unsafe_allow_html=True)
    render_warnings(injury["warnings"])

    if show_radar:
        fig_radar = create_feature_radar(features, MLB_BENCHMARKS)
        st.plotly_chart(fig_radar, use_container_width=True)

    with st.expander("📊 Biomechanics Table"):
        render_biomechanics_table(features)

    render_coaching_section(features, injury, outcome_result, api_key)

    # FIX: Save to history from webcam mode too
    _save_to_history(annotated_rgb, injury, outcome_result)
    _render_history()


# ── Session History ──────────────────────────────────────────────────────────
def _save_to_history(img_rgb, injury, outcome_result):
    if "history" not in st.session_state:
        st.session_state.history = []
    entry = {
        "thumbnail": img_rgb,
        "risk":      injury.get("overall_risk", "Low"),
        "risk_idx":  injury.get("risk_index", 0),
        "delivery":  outcome_result.get("label", "—") if outcome_result else "—",
        "ts":        time.strftime("%H:%M:%S"),
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:5]


def _render_history():
    hist = st.session_state.get("history", [])
    if not hist:
        return
    st.markdown('<div class="pg-section-header">🕑 Recent Analyses</div>', unsafe_allow_html=True)
    cols = st.columns(len(hist))
    for i, entry in enumerate(hist):
        with cols[i]:
            # FIX: use_container_width replaces deprecated use_column_width
            st.image(entry["thumbnail"], use_container_width=True)
            risk_color = _risk_color(entry["risk_idx"])
            st.markdown(
                f"""<div style="text-align:center;font-size:0.75rem;color:{risk_color}">
                    {entry['delivery']} · {entry['risk']} · {entry['ts']}
                </div>""",
                unsafe_allow_html=True,
            )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    st.markdown(
        """
        <div class="pg-hero">
            <h1>⚾ Pitch Vision AI</h1>
            <p>Elite biomechanics analysis · Injury prevention · AI-powered coaching</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode, api_key, sample_rate, confidence_threshold, max_frames_display, show_radar, show_feature_imp = render_sidebar()

    if mode == "Image Upload":
        image_mode(api_key, show_radar, show_feature_imp, confidence_threshold)
    elif mode == "Video Upload":
        video_mode(api_key, sample_rate, max_frames_display, show_radar, show_feature_imp, confidence_threshold)
    elif mode == "Live Webcam":
        webcam_mode(api_key, confidence_threshold, show_radar)


if __name__ == "__main__":
    main()
