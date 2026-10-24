"""Train the model and refresh the recommendations file."""

import json
import logging
from datetime import UTC, datetime

import joblib
import pandas as pd

from animerec.config import Settings
from animerec.mal import MALClient
from animerec.model import FEATURES, build_pipeline, evaluate, label
from animerec.ratings import load_candidates, load_ratings

log = logging.getLogger(__name__)

MIN_LABELS_PER_CLASS = 5


class PipelineError(Exception):
    """Something the user has to fix (missing model, too few labels...)."""


def training_set(user_list: pd.DataFrame, ratings: pd.DataFrame, like_threshold: int):
    """Scores from your MAL list plus the /rate page ratings, which win on conflicts."""
    from_list = user_list.assign(y=label(user_list, like_threshold)).dropna(subset=["y"])
    from_list = from_list[~from_list["anime_id"].isin(ratings["anime_id"])]
    data = pd.concat([from_list[[*FEATURES, "y"]], ratings[[*FEATURES, "y"]]], ignore_index=True)
    return data[FEATURES], data["y"].astype(int)


def train(settings: Settings, client: MALClient) -> None:
    """Fit the model on your MAL list scores and your /rate page ratings."""
    X, y = training_set(
        client.user_list(settings.mal_username), load_ratings(settings), settings.like_threshold
    )

    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < MIN_LABELS_PER_CLASS:
        raise PipelineError(
            f"Need at least {MIN_LABELS_PER_CLASS} liked and {MIN_LABELS_PER_CLASS} disliked anime "
            f"to train, got {counts.to_dict()}. Rate more titles on the /rate page (`animerec serve`)."
        )
    log.info("Training on %d anime (%d liked)", len(y), counts.get(1, 0))

    pipeline = build_pipeline()
    log.info("Cross-validated average precision: %.3f (baseline %.3f)", evaluate(pipeline, X, y), y.mean())
    pipeline.fit(X[FEATURES], y)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.model_path)
    log.info("Model saved to %s", settings.model_path)


def recommend(
    candidates: pd.DataFrame,
    user_list: pd.DataFrame,
    ratings: pd.DataFrame,
    pipeline,
    settings: Settings,
) -> pd.DataFrame:
    """Score candidates, skipping what is on your list (except plan-to-watch) or already rated."""
    on_list = user_list.loc[user_list["list_status"] != "plan_to_watch", "anime_id"]
    seen = pd.concat([on_list, ratings["anime_id"]])
    candidates = candidates[
        candidates["media_type"].isin(settings.media_types) & ~candidates["anime_id"].isin(seen)
    ].copy()
    candidates["score"] = pipeline.predict_proba(candidates[FEATURES])[:, 1]
    return candidates.sort_values("score", ascending=False).head(settings.top_n)


def update(settings: Settings, client: MALClient) -> None:
    """Write the top recommendations to the JSON file served by the app."""
    if not settings.model_path.exists():
        raise PipelineError(f"No model at {settings.model_path}, run `animerec train` first.")
    pipeline = joblib.load(settings.model_path)

    top = recommend(
        load_candidates(settings, client, refresh=True),
        client.user_list(settings.mal_username),
        load_ratings(settings),
        pipeline,
        settings,
    )
    items = [
        {
            "anime_id": int(row.anime_id),
            "title": row.title,
            "genres": row.genres.split("|") if row.genres else [],
            "num_episodes": None if pd.isna(row.num_episodes) else int(row.num_episodes),
            "mean_score": None if pd.isna(row.mean_score) else float(row.mean_score),
            "media_type": row.media_type,
            "picture": row.picture,
            "score": round(float(row.score), 4),
            "url": f"https://myanimelist.net/anime/{row.anime_id}",
        }
        for row in top.itertuples()
    ]
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "username": settings.mal_username,
        "items": items,
    }
    settings.data_path.parent.mkdir(parents=True, exist_ok=True)
    settings.data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %d recommendations to %s", len(items), settings.data_path)


def load_recommendations(settings: Settings) -> dict:
    if not settings.data_path.exists():
        return {"generated_at": None, "username": settings.mal_username, "items": []}
    return json.loads(settings.data_path.read_text(encoding="utf-8"))
