from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "ipl_win_probability_model.joblib"


@st.cache_resource
def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        st.error("Model artifact is missing. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


def balls_from_overs(overs: float) -> int:
    completed_overs = int(overs)
    balls_in_current_over = int(round((overs - completed_overs) * 10))
    balls_in_current_over = min(max(balls_in_current_over, 0), 5)
    return completed_overs * 6 + balls_in_current_over


def probability_for_class(model, frame: pd.DataFrame, class_label: int) -> float:
    probabilities = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(class_label)])


artifact = load_artifact()
options = artifact["options"]
defaults = artifact["defaults"]

st.set_page_config(page_title="IPL Win Probability Predictor", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --ink: #14212e;
      --muted: #687486;
      --line: #dce4ee;
      --panel: rgba(255, 255, 255, 0.92);
      --green: #1f9f78;
      --blue: #3159a6;
      --gold: #d99728;
    }
    .stApp {
      background:
        radial-gradient(circle at top left, rgba(49, 89, 166, 0.16), transparent 32rem),
        linear-gradient(135deg, #f4f7fb 0%, #eef6f3 55%, #fff8ed 100%);
    }
    .block-container {
      max-width: 1180px;
      padding-top: 2.1rem;
      padding-bottom: 3rem;
    }
    h1, h2, h3, p {
      letter-spacing: 0;
    }
    [data-testid="stMetric"] {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 42px rgba(31, 45, 61, 0.09);
      padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] p {
      color: var(--muted);
      font-weight: 800;
    }
    [data-testid="stMetricValue"] {
      color: var(--ink);
      font-weight: 900;
    }
    [data-testid="stSelectbox"], [data-testid="stNumberInput"], [data-testid="stSlider"] {
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px 12px;
    }
    .hero-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 20px 52px rgba(38, 57, 77, 0.11);
      margin-bottom: 22px;
      overflow: hidden;
      padding: 24px 26px;
      position: relative;
    }
    .hero-panel:before {
      background: linear-gradient(90deg, var(--blue), var(--green), var(--gold));
      content: "";
      height: 5px;
      left: 0;
      position: absolute;
      right: 0;
      top: 0;
    }
    .kicker {
      color: var(--blue);
      font-size: 0.78rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      margin: 0 0 8px;
      text-transform: uppercase;
    }
    .hero-title {
      color: var(--ink);
      font-size: clamp(2.3rem, 5vw, 4.4rem);
      font-weight: 900;
      line-height: 0.95;
      margin: 0;
    }
    .hero-copy {
      color: var(--muted);
      font-size: 1.04rem;
      line-height: 1.55;
      margin: 14px 0 0;
      max-width: 720px;
    }
    .score-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 20px 52px rgba(38, 57, 77, 0.11);
      padding: 22px;
    }
    .versus-row {
      align-items: center;
      display: grid;
      gap: 18px;
      grid-template-columns: 1fr auto 1fr;
      margin-bottom: 18px;
    }
    .team-name {
      color: var(--ink);
      font-size: 1.28rem;
      font-weight: 900;
    }
    .team-name:last-child {
      text-align: right;
    }
    .versus-pill {
      background: #14212e;
      border-radius: 999px;
      color: #fff;
      font-size: 0.76rem;
      font-weight: 900;
      padding: 7px 10px;
    }
    .winbar {
      background: #dfe7ef;
      border-radius: 999px;
      height: 18px;
      overflow: hidden;
    }
    .winbar span {
      background: linear-gradient(90deg, var(--green), var(--blue));
      display: block;
      height: 100%;
      width: var(--batting);
    }
    .prob-row {
      color: var(--muted);
      display: flex;
      font-size: 0.92rem;
      font-weight: 800;
      justify-content: space-between;
      margin-top: 10px;
    }
    .state-card {
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 14px 16px;
    }
    .state-card strong {
      color: var(--ink);
      display: block;
      font-size: 1.45rem;
      line-height: 1.2;
      margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero-panel">
      <p class="kicker">Sports Analytics</p>
      <h1 class="hero-title">IPL Match Win Predictor</h1>
      <p class="hero-copy">Second innings probability view for target chases, wickets, run rate pressure, venue, and team matchups.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")
with left:
    batting_team = st.selectbox(
        "Batting team",
        options["batting_team"],
        index=options["batting_team"].index(defaults["batting_team"]),
    )
    city = st.selectbox(
        "City",
        options["city"],
        index=options["city"].index(defaults["city"]),
    )
    target = st.number_input("Target score", min_value=1, max_value=300, value=defaults["target"])
    overs = st.number_input(
        "Overs completed",
        min_value=0.0,
        max_value=19.5,
        value=float(defaults["overs"]),
        step=0.1,
        format="%.1f",
    )
with right:
    bowling_options = [team for team in options["bowling_team"] if team != batting_team]
    default_bowling = defaults["bowling_team"]
    if default_bowling == batting_team:
        default_bowling = bowling_options[0]
    bowling_team = st.selectbox(
        "Bowling team",
        bowling_options,
        index=bowling_options.index(default_bowling),
    )
    current_score = st.number_input(
        "Current score", min_value=0, max_value=300, value=defaults["current_score"]
    )
    wickets_fallen = st.slider("Wickets fallen", min_value=0, max_value=10, value=defaults["wickets_fallen"])

balls_bowled = balls_from_overs(overs)
balls_left = max(120 - balls_bowled, 0)
wickets_remaining = max(10 - wickets_fallen, 0)
runs_left = max(target - current_score, 0)
current_run_rate = (current_score * 6 / balls_bowled) if balls_bowled else 0
required_run_rate = (runs_left * 6 / balls_left) if balls_left else 0

if current_score >= target:
    batting_probability = 1.0
elif wickets_fallen >= 10 or balls_left == 0:
    batting_probability = 0.0
else:
    input_frame = pd.DataFrame(
        [
            {
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "city": city,
                "runs_left": runs_left,
                "balls_left": balls_left,
                "wickets": wickets_remaining,
                "total_runs_x": target,
                "cur_run_rate": current_run_rate,
                "req_run_rate": required_run_rate,
            }
        ],
        columns=artifact["features"],
    )
    batting_probability = probability_for_class(artifact["model"], input_frame, 1)

bowling_probability = 1 - batting_probability
batting_percent = batting_probability * 100
bowling_percent = bowling_probability * 100

st.markdown(
    f"""
    <section class="score-panel">
      <div class="versus-row">
        <div class="team-name">{batting_team}</div>
        <div class="versus-pill">VS</div>
        <div class="team-name">{bowling_team}</div>
      </div>
      <div class="winbar" style="--batting: {batting_percent:.1f}%;">
        <span></span>
      </div>
      <div class="prob-row">
        <span>{batting_percent:.1f}% batting</span>
        <span>{bowling_percent:.1f}% bowling</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Match State")
state_columns = st.columns(4)
state_columns[0].markdown(f'<div class="state-card">Runs left<strong>{runs_left}</strong></div>', unsafe_allow_html=True)
state_columns[1].markdown(f'<div class="state-card">Balls left<strong>{balls_left}</strong></div>', unsafe_allow_html=True)
state_columns[2].markdown(f'<div class="state-card">Wickets in hand<strong>{wickets_remaining}</strong></div>', unsafe_allow_html=True)
state_columns[3].markdown(f'<div class="state-card">Required RR<strong>{required_run_rate:.2f}</strong></div>', unsafe_allow_html=True)

with st.expander("Model details"):
    metrics = artifact["metrics"]
    st.write(
        {
            "accuracy": round(metrics.get("accuracy", 0), 3),
            "f1": round(metrics.get("f1", 0), 3),
            "roc_auc": round(metrics.get("roc_auc", 0), 3),
        }
    )
