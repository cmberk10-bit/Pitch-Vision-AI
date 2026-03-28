"""
PitchGuard AI — Visualizations Module
All Plotly charts used across the application.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional

# ── Dark theme constants ─────────────────────────────────────────────────────
PAPER_BG  = "#13131f"
PLOT_BG   = "#1e1e2e"
FONT_COL  = "#e0e0e0"
GRID_COL  = "#2a2a3e"
PURPLE    = "#7c4dff"
CYAN      = "#00e5ff"
GREEN     = "#00c853"
AMBER     = "#ffab00"
RED       = "#ff5252"

_BASE_LAYOUT = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=FONT_COL, family="Space Grotesk, sans-serif"),
    margin=dict(l=16, r=16, t=32, b=16),
)


def _bar_color(score: float) -> str:
    if score < 33:
        return GREEN
    elif score < 66:
        return AMBER
    return RED

#########################
def create_risk_gauge(risk_index: int) -> go.Figure:
    """
    Plotly indicator gauge 0–100.
    Zones: 0-33 green, 33-66 amber, 66-100 red.
    """
    risk_index = int(np.clip(risk_index, 0, 100))

    if risk_index < 33:
        needle_color = GREEN
    elif risk_index < 66:
        needle_color = AMBER
    else:
        needle_color = RED

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_index,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'color': needle_color, 'family': "Space Grotesk"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': FONT_COL},
            'bar': {'color': needle_color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': GRID_COL,
            'steps': [
                {'range': [0, 33], 'color': 'rgba(0, 200, 83, 0.1)'},
                {'range': [33, 66], 'color': 'rgba(255, 171, 0, 0.1)'},
                {'range': [66, 100], 'color': 'rgba(213, 0, 0, 0.1)'}
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FONT_COL, 'family': "Space Grotesk"},
        # Only one margin definition here:
        #margin1=dict(l=20, r=20, t=40, b=20),
        #height=220
    )
    
    return fig
#########################

# ── 1. Risk Gauge ────────────────────────────────────────────────────────────

###########################
def create_body_part_risk_chart(body_part_risks: Dict[str, float]) -> go.Figure:
    """
    Horizontal bar chart showing risk per body segment.
    """
    # Sort risks so highest risk is at the top
    sorted_risks = sorted(body_part_risks.items(), key=lambda x: x[1])
    labels = [k.replace("_", " ").title() for k, v in sorted_risks]
    values = [v for k, v in sorted_risks]
    colors = [_bar_color(v) for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{v}%" for v in values],
        textposition='auto',
    ))

    fig.update_layout(
        title=dict(
            text="Risk Distribution by Segment",
            font=dict(size=14, color=FONT_COL)
        ),
        xaxis=dict(
            title="Risk Score (%)",
            range=[0, 105],
            gridcolor=GRID_COL,
            zeroline=False
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': FONT_COL, 'family': "Space Grotesk"},
        # FIXED: Only one margin definition
        #margin=dict(l=10, r=20, t=40, b=10),
        #height=300,
        #showlegend=False
    )

    return fig
###########################
# ── 2. Body-Part Risk Bar Chart ──────────────────────────────────────────────


# ── 3. Time-Series Chart ─────────────────────────────────────────────────────
def create_time_series_chart(
    per_frame_features: List[Dict[str, float]],
    keys: List[str],
) -> go.Figure:
    """
    Multi-line chart of selected biomechanical metrics across frames.
    Height 300px.
    """
    palette = [PURPLE, CYAN, GREEN, AMBER, RED, "#ff80ab", "#b388ff", "#80d8ff"]
    frames  = list(range(len(per_frame_features)))

    fig = go.Figure()
    for i, key in enumerate(keys):
        values = [f.get(key, 0) for f in per_frame_features]
        col    = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=frames,
            y=values,
            name=key.replace("_", " ").title(),
            mode="lines+markers",
            line=dict(color=col, width=2.5),
            marker=dict(size=5, color=col),
            hovertemplate=f"<b>{key}</b><br>Frame %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Metric Trends Over Frames", font=dict(size=13, color=FONT_COL), x=0),
        height=300,
        xaxis=dict(
            title="Frame",
            showgrid=True,
            gridcolor=GRID_COL,
            zeroline=False,
            tickfont=dict(color="#666688", size=10),
        ),
        yaxis=dict(
            title="Value",
            showgrid=True,
            gridcolor=GRID_COL,
            zeroline=False,
            tickfont=dict(color="#666688", size=10),
        ),
        legend=dict(
            bgcolor="rgba(19,19,31,0.8)",
            bordercolor=GRID_COL,
            borderwidth=1,
            font=dict(size=10, color=FONT_COL),
        ),
        hovermode="x unified",
    )
    return fig

###################

###################
# ── 4. Mechanics Radar Chart ─────────────────────────────────────────────────
def create_feature_radar(features: Dict[str, float], benchmarks: Dict[str, float]) -> go.Figure:
    """
    Radar chart comparing current pitcher features against MLB Benchmarks
    using biomechanical normalization.
    """
    # 1. Import local dependency 
    from biomechanics import OPTIMAL_RANGES, RADAR_FEATURES

    # 2. Prepare labels and normalized values
    labels = [f.replace("_", " ").title() for f in RADAR_FEATURES]
    pitcher_vals = []
    mlb_vals = []

    for feat in RADAR_FEATURES:
        opt = OPTIMAL_RANGES.get(feat, {})
        lo = opt.get("min", 0.0)
        hi = opt.get("max", 1.0)
        span = (hi - lo) or 1.0

        raw_p = features.get(feat, (lo + hi) / 2)
        raw_m = benchmarks.get(feat, (lo + hi) / 2)

        # Normalise: clamp to [lo-span, hi+span] then map to 0–1
        norm_p = float(np.clip((raw_p - lo) / span + 0.5, 0.05, 1.0))
        norm_m = float(np.clip((raw_m - lo) / span + 0.5, 0.05, 1.0))
        pitcher_vals.append(norm_p)
        mlb_vals.append(norm_m)

    # 3. Close the polygon
    labels_closed = labels + [labels[0]]
    pitcher_vals_closed = pitcher_vals + [pitcher_vals[0]]
    mlb_vals_closed = mlb_vals + [mlb_vals[0]]

    # 4. Create the figure
    fig = go.Figure()

    # MLB trace
    fig.add_trace(go.Scatterpolar(
        r=mlb_vals_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(0,229,255,0.07)",
        line=dict(color=CYAN, width=2, dash="dash"),
        name="MLB Elite Avg",
        hovertemplate="<b>%{theta}</b><br>MLB: %{r:.2f}<extra></extra>",
    ))

    # Pitcher trace
    fig.add_trace(go.Scatterpolar(
        r=pitcher_vals_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(124,77,255,0.18)",
        line=dict(color=PURPLE, width=2.5),
        name="Your Pitcher",
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>",
    ))

    # 5. Apply the SAFE layout (fixing the TypeError)
    layout_update = _BASE_LAYOUT.copy()
    layout_update.update(dict(
        title=dict(text="Mechanics Radar vs MLB Benchmarks", font=dict(size=13, color=FONT_COL), x=0.5),
        height=380,
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(visible=True, range=[0, 1.15], showticklabels=False, gridcolor=GRID_COL),
            angularaxis=dict(tickfont=dict(size=10, color=FONT_COL), gridcolor=GRID_COL),
        ),
        legend=dict(
            bgcolor="rgba(19,19,31,0.85)", bordercolor=GRID_COL, borderwidth=1,
            orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5
        ),
        # FIXED: Only one margin definition passed via the update() method
        #margin=dict(l=40, r=40, t=50, b=60)
    ))
    
    fig.update_layout(**layout_update)
    return fig

# ── 5. Per-Frame Risk Trend ──────────────────────────────────────────────────
def create_per_frame_risk_trend(per_frame_injuries: List[Dict]) -> go.Figure:
    """
    Line chart of risk_index per frame, filled below. Height 280px.
    """
    frames  = list(range(len(per_frame_injuries)))
    risks   = [float(inj.get("risk_index", 0)) for inj in per_frame_injuries]

    # Color segments by zone
    fig = go.Figure()

    # Filled area (full)
    fig.add_trace(go.Scatter(
        x=frames,
        y=risks,
        mode="none",
        fill="tozeroy",
        fillcolor="rgba(124,77,255,0.10)",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main line colored by level
    fig.add_trace(go.Scatter(
        x=frames,
        y=risks,
        mode="lines+markers",
        line=dict(
            color=PURPLE,
            width=2.5,
            shape="spline",
            smoothing=0.8,
        ),
        marker=dict(
            size=7,
            color=risks,
            colorscale=[[0, GREEN], [0.33, GREEN], [0.33, AMBER], [0.66, AMBER], [0.66, RED], [1.0, RED]],
            cmin=0,
            cmax=100,
            line=dict(color="white", width=1),
        ),
        name="Risk Index",
        hovertemplate="Frame %{x}<br>Risk Index: %{y:.0f}<extra></extra>",
    ))

    # Reference lines
    for lvl, col, lbl in [(33, AMBER, "Medium"), (66, RED, "High")]:
        fig.add_hline(
            y=lvl,
            line=dict(color=col, width=1, dash="dot"),
            annotation_text=lbl,
            annotation_font=dict(color=col, size=10),
            annotation_position="right",
        )

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Per-Frame Risk Trend", font=dict(size=13, color=FONT_COL), x=0),
        height=280,
        xaxis=dict(
            title="Frame",
            showgrid=True,
            gridcolor=GRID_COL,
            zeroline=False,
            tickfont=dict(color="#666688", size=10),
        ),
        yaxis=dict(
            title="Risk Index",
            range=[0, 105],
            showgrid=True,
            gridcolor=GRID_COL,
            zeroline=False,
            tickfont=dict(color="#666688", size=10),
        ),
        showlegend=False,
        hovermode="x",
    )
    return fig


# ── 6. Feature Importance Chart ──────────────────────────────────────────────
def create_feature_importance_chart(
    feature_importance: Dict[str, float],
) -> Optional[go.Figure]:
    """
    Horizontal bar chart of XGBoost feature importances.
    """
    if not feature_importance:
        return None

    items  = sorted(feature_importance.items(), key=lambda x: x[1])
    labels = [k.replace("_", " ").title() for k, _ in items]
    values = [v for _, v in items]

    # Gradient color: low importance = dim purple, high = bright cyan
    max_v  = max(values) or 1.0
    colors = [
        f"rgba({int(124 + (0 - 124) * v / max_v)},{int(77 + (229 - 77) * v / max_v)},{int(255 + (255 - 255) * v / max_v)},0.85)"
        for v in values
    ]
    colors = [PURPLE if v < max_v * 0.5 else CYAN for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.06)", width=1)),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color=FONT_COL, size=10),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="XGBoost Feature Importance", font=dict(size=13, color=FONT_COL), x=0),
        height=360,
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID_COL,
            zeroline=False,
            tickfont=dict(color="#666688", size=10),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color=FONT_COL, size=10),
        ),
        bargap=0.22,
        #margin=dict(l=10, r=60, t=40, b=10),
    )
    return fig