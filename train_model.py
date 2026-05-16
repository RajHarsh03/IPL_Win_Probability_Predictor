from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Dataset"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "ipl_win_probability_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "ipl_win_probability_model.metrics.json"

CURRENT_TEAMS = [
    "Sunrisers Hyderabad",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Kings XI Punjab",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Delhi Capitals",
]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def normalize_team_names(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = frame[column].str.replace("Delhi Daredevils", "Delhi Capitals")
        frame[column] = frame[column].str.replace("Deccan Chargers", "Sunrisers Hyderabad")
    return frame


def build_training_frame() -> pd.DataFrame:
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    deliveries = pd.read_csv(DATA_DIR / "deliveries.csv")

    first_innings_total = (
        deliveries.groupby(["match_id", "inning"])["total_runs"]
        .sum()
        .reset_index()
    )
    first_innings_total = first_innings_total[first_innings_total["inning"] == 1]
    first_innings_total["total_runs"] = first_innings_total["total_runs"] + 1

    match_df = matches.merge(
        first_innings_total[["match_id", "total_runs"]],
        left_on="id",
        right_on="match_id",
    )
    match_df = normalize_team_names(match_df, ["team1", "team2"])
    match_df = match_df[
        match_df["team1"].isin(CURRENT_TEAMS) & match_df["team2"].isin(CURRENT_TEAMS)
    ]
    match_df = match_df[match_df["dl_applied"] == 0]
    match_df = match_df[["match_id", "city", "winner", "total_runs"]]

    delivery_df = match_df.merge(deliveries, on="match_id")
    delivery_df = normalize_team_names(delivery_df, ["batting_team", "bowling_team"])
    delivery_df = delivery_df[delivery_df["inning"] == 2].copy()

    delivery_df["current_score"] = delivery_df.groupby("match_id")["total_runs_y"].cumsum()
    delivery_df["runs_left"] = delivery_df["total_runs_x"] - delivery_df["current_score"]
    delivery_df["balls_left"] = 126 - (delivery_df["over"] * 6 + delivery_df["ball"])

    delivery_df["player_dismissed"] = delivery_df["player_dismissed"].fillna("0")
    delivery_df["player_dismissed"] = delivery_df["player_dismissed"].apply(
        lambda value: 0 if value == "0" else 1
    )
    wickets_fallen = delivery_df.groupby("match_id")["player_dismissed"].cumsum()
    delivery_df["wickets"] = 10 - wickets_fallen

    balls_bowled = 120 - delivery_df["balls_left"]
    delivery_df["cur_run_rate"] = (delivery_df["current_score"] * 6) / balls_bowled
    delivery_df["req_run_rate"] = (delivery_df["runs_left"] * 6) / delivery_df["balls_left"]
    delivery_df["result"] = (
        delivery_df["batting_team"] == delivery_df["winner"]
    ).astype(int)

    final_df = delivery_df[
        [
            "batting_team",
            "bowling_team",
            "city",
            "runs_left",
            "balls_left",
            "wickets",
            "total_runs_x",
            "cur_run_rate",
            "req_run_rate",
            "result",
        ]
    ]
    final_df = final_df.dropna()
    final_df = final_df[final_df["balls_left"] > 0]
    final_df = final_df[final_df["runs_left"] >= 0]
    return final_df


def train() -> dict:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    data = build_training_frame()

    x = data.drop(columns=["result"])
    y = data["result"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                make_one_hot_encoder(),
                ["batting_team", "bowling_team", "city"],
            )
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(solver="liblinear", max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    artifact = {
        "project": "IPL Match Win Predictor",
        "model": model,
        "features": x.columns.tolist(),
        "metrics": metrics,
        "options": {
            "batting_team": sorted(x["batting_team"].unique().tolist()),
            "bowling_team": sorted(x["bowling_team"].unique().tolist()),
            "city": sorted(x["city"].unique().tolist()),
        },
        "defaults": {
            "batting_team": "Delhi Capitals",
            "bowling_team": "Sunrisers Hyderabad",
            "city": "Mumbai",
            "target": 180,
            "current_score": 100,
            "overs": 10.0,
            "wickets_fallen": 4,
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    trained_artifact = train()
    print(f"Saved IPL win probability model to {MODEL_PATH}")
    print(json.dumps(trained_artifact["metrics"], indent=2))
