"""Likes and dislikes given on the /rate page, plus the queue of titles to rate."""

import json
import time

import joblib
import pandas as pd

from animerec.config import Settings
from animerec.mal import COLUMNS, MALClient
from animerec.model import FEATURES

CANDIDATES_MAX_AGE = 7 * 24 * 3600


def load_ratings(settings: Settings) -> pd.DataFrame:
    """One row per rated anime: its features plus `y` (1 liked, 0 disliked)."""
    if not settings.ratings_path.exists():
        return pd.DataFrame(columns=[*COLUMNS, "y"])
    rows = json.loads(settings.ratings_path.read_text(encoding="utf-8"))
    return pd.DataFrame(rows, columns=[*COLUMNS, "y"])


def save_rating(settings: Settings, anime: dict, liked: bool) -> None:
    ratings = load_ratings(settings)
    ratings = ratings[ratings["anime_id"] != anime["anime_id"]]
    row = {column: anime.get(column) for column in COLUMNS} | {"y": int(liked)}
    ratings = pd.concat([ratings, pd.DataFrame([row])], ignore_index=True)
    settings.ratings_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ratings_path.write_text(ratings.to_json(orient="records", force_ascii=False, indent=1), "utf-8")


def load_candidates(settings: Settings, client: MALClient, refresh: bool = False) -> pd.DataFrame:
    """The MAL ranking, cached on disk for a week."""
    path = settings.candidates_path
    if not refresh and path.exists() and time.time() - path.stat().st_mtime < CANDIDATES_MAX_AGE:
        return pd.read_json(path, orient="records", dtype={"genres": str, "studios": str})
    candidates = client.ranking(settings.num_candidates)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_json(path, orient="records", force_ascii=False)
    return candidates


def rating_queue(
    candidates: pd.DataFrame, user_list: pd.DataFrame, ratings: pd.DataFrame, settings: Settings, n: int
) -> pd.DataFrame:
    """Next titles to rate: the ones the model is least sure about, or the most popular without a model."""
    on_list = user_list.loc[user_list["list_status"] != "plan_to_watch", "anime_id"]
    pool = candidates[
        candidates["media_type"].isin(settings.media_types)
        & ~candidates["anime_id"].isin(on_list)
        & ~candidates["anime_id"].isin(ratings["anime_id"])
    ].copy()
    if settings.model_path.exists():
        probability = joblib.load(settings.model_path).predict_proba(pool[FEATURES])[:, 1]
        pool["uncertainty"] = -abs(probability - 0.5)
        return pool.sort_values("uncertainty", ascending=False).head(n)
    return pool.sort_values("popularity").head(n)
