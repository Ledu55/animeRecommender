import json

import numpy as np
import pandas as pd
import pytest

from animerec.config import Settings
from animerec.mal import MALClient, parse_node, to_frame
from animerec.model import FEATURES, build_pipeline, label
from animerec.pipeline import recommend, training_set, update
from animerec.ratings import load_candidates, load_ratings, rating_queue, save_rating
from animerec.web import app, build_site

NODE = {
    "id": 1,
    "title": "Cowboy Bebop",
    "main_picture": {"medium": "https://cdn.myanimelist.net/1.jpg"},
    "mean": 8.75,
    "genres": [{"id": 1, "name": "Action"}, {"id": 24, "name": "Sci-Fi"}],
    "studios": [{"id": 14, "name": "Sunrise"}],
    "media_type": "tv",
    "num_episodes": 26,
    "start_date": "1998-04-03",
    "synopsis": "Bounty hunters in space.",
    "popularity": 40,
}


def fake_anime(n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic anime where "Action" titles score high, so the model has something to learn."""
    rng = np.random.default_rng(seed)
    genres = rng.choice(["Action|Sci-Fi", "Romance|Drama", "Comedy"], n)
    nodes = [
        {
            "id": 1000 * seed + i,
            "title": f"{genre.split('|')[0]} show {i}",
            "mean": round(float(rng.uniform(6, 9)), 2),
            "genres": [{"name": g} for g in genre.split("|")],
            "studios": [{"name": str(rng.choice(["Sunrise", "Madhouse", "Bones"]))}],
            "media_type": "tv",
            "num_episodes": int(rng.integers(1, 50)),
            "start_date": f"{rng.integers(1990, 2025)}-01-01",
            "synopsis": f"A story about {genre.lower()}.",
            "popularity": int(rng.integers(1, 5000)),
        }
        for i, genre in enumerate(genres)
    ]
    return to_frame(nodes)


def fake_user_list(n: int = 60) -> pd.DataFrame:
    df = fake_anime(n, seed=1)
    df["list_status"] = "completed"
    df["list_score"] = np.where(df["genres"].str.contains("Action"), 9, 4)
    return df


def test_parse_node_flattens_fields():
    row = parse_node(NODE)
    assert row["genres"] == "Action|Sci-Fi"
    assert row["studios"] == "Sunrise"
    assert row["start_year"] == 1998
    assert row["picture"].endswith("1.jpg")


def test_parse_node_handles_missing_fields():
    row = parse_node({"id": 2, "title": "Mystery", "num_episodes": 0})
    assert row["genres"] == "" and row["start_year"] is None
    assert row["num_episodes"] is None  # MAL uses 0 for unknown
    assert row["media_type"] == "unknown"


def test_label():
    df = pd.DataFrame(
        {
            "list_score": [9, 7, 6, 0, 0],
            "list_status": ["completed", "completed", "completed", "dropped", "watching"],
        }
    )
    assert label(df, like_threshold=7).tolist()[:4] == [1, 1, 0, 0]
    assert np.isnan(label(df, like_threshold=7).iloc[4])


def test_pipeline_learns_preference():
    train = fake_user_list()
    y = (train["list_score"] >= 7).astype(int)
    pipeline = build_pipeline().fit(train[FEATURES], y)

    candidates = fake_anime(30, seed=2)
    scores = pipeline.predict_proba(candidates[FEATURES])[:, 1]
    is_action = candidates["genres"].str.contains("Action")
    assert scores[is_action].mean() > scores[~is_action].mean()


def test_recommend_skips_seen_and_keeps_plan_to_watch(tmp_path):
    user_list = fake_user_list(20)
    user_list.loc[0, "list_status"] = "plan_to_watch"
    pipeline = build_pipeline().fit(user_list[FEATURES], (user_list["list_score"] >= 7).astype(int))
    candidates = pd.concat([user_list[FEATURES + ["anime_id"]], fake_anime(10, seed=3)])

    top = recommend(
        candidates,
        user_list,
        load_ratings(Settings(ratings_path=tmp_path / "r.json")),
        pipeline,
        Settings(top_n=100),
    )
    seen = set(user_list["anime_id"][1:])
    assert not seen & set(top["anime_id"])
    assert user_list.loc[0, "anime_id"] in set(top["anime_id"])
    assert top["score"].is_monotonic_decreasing


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    """Serves two pages to check that pagination follows the `next` link."""

    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if len(self.calls) == 1:
            return FakeResponse({"data": [{"node": NODE}], "paging": {"next": "https://next"}})
        return FakeResponse({"data": [{"node": {**NODE, "id": 2}}], "paging": {}})


def test_client_paginates():
    session = FakeSession()
    df = MALClient("id", session=session, delay=0).ranking(limit=10)
    assert df["anime_id"].tolist() == [1, 2]
    assert session.calls[1] == "https://next"
    assert session.headers["X-MAL-CLIENT-ID"] == "id"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        mal_username="tester",
        model_path=tmp_path / "model.joblib",
        data_path=tmp_path / "recommendations.json",
        site_dir=tmp_path / "site",
        ratings_path=tmp_path / "ratings.json",
        candidates_path=tmp_path / "candidates.json",
        top_n=5,
    )


def test_update_and_render(settings, monkeypatch):
    import joblib

    user_list = fake_user_list()
    joblib.dump(
        build_pipeline().fit(user_list[FEATURES], (user_list["list_score"] >= 7).astype(int)),
        settings.model_path,
    )

    class FakeClient:
        def ranking(self, limit):
            return fake_anime(20, seed=4)

        def user_list(self, username):
            return user_list

    update(settings, FakeClient())
    payload = json.loads(settings.data_path.read_text(encoding="utf-8"))
    assert len(payload["items"]) == 5
    assert payload["items"][0]["url"].startswith("https://myanimelist.net/anime/")

    html = build_site(settings).read_text(encoding="utf-8")
    assert payload["items"][0]["title"] in html

    monkeypatch.setenv("DATA_PATH", str(settings.data_path))
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert payload["items"][0]["title"].encode() in response.data


class FakeClient:
    def __init__(self):
        self.ranking_calls = 0

    def ranking(self, limit):
        self.ranking_calls += 1
        return fake_anime(40, seed=5)

    def user_list(self, username):
        return fake_user_list(10)


def test_ratings_override_list_scores(settings):
    user_list = fake_user_list(10)
    first = user_list.iloc[0].to_dict()
    save_rating(settings, first, liked=False)
    save_rating(settings, first, liked=True)  # re-rating replaces the old rating
    ratings = load_ratings(settings)
    assert len(ratings) == 1 and ratings["y"].tolist() == [1]

    X, y = training_set(user_list, ratings, like_threshold=7)
    assert len(X) == 10 and y.iloc[-1] == 1


def test_candidates_are_cached(settings):
    fake = FakeClient()
    load_candidates(settings, fake)
    cached = load_candidates(settings, fake)
    assert fake.ranking_calls == 1
    assert cached["genres"].str.contains("|", regex=False).any()
    load_candidates(settings, fake, refresh=True)
    assert fake.ranking_calls == 2


def test_rating_queue_prefers_uncertain(settings):
    import joblib

    candidates = fake_anime(40, seed=5)
    user_list = fake_user_list(10)
    save_rating(settings, candidates.iloc[0].to_dict(), liked=True)
    ratings = load_ratings(settings)

    queue = rating_queue(candidates, user_list, ratings, settings, n=5)
    assert candidates.iloc[0]["anime_id"] not in set(queue["anime_id"])
    assert queue["popularity"].is_monotonic_increasing  # no model yet: most popular first

    train = fake_user_list()
    joblib.dump(
        build_pipeline().fit(train[FEATURES], (train["list_score"] >= 7).astype(int)), settings.model_path
    )
    queue = rating_queue(candidates, user_list, ratings, settings, n=5)
    assert queue["uncertainty"].is_monotonic_decreasing


def test_rate_page_flow(settings, monkeypatch):
    import animerec.web as web

    monkeypatch.setattr(web, "Settings", lambda: settings)
    monkeypatch.setattr(web, "client", lambda s: FakeClient())
    http = app.test_client()

    page = http.get("/rate")
    assert page.status_code == 200 and b"Popular" not in page.data
    anime_id = int(load_candidates(settings, FakeClient())["anime_id"].iloc[0])

    response = http.post(
        "/rate", data={"anime_id": anime_id, "choice": "like"}, headers={"Accept": "application/json"}
    )
    assert response.get_json() == {"ok": True}
    assert http.post("/rate", data={"anime_id": 1, "choice": "skip"}).status_code == 302
    assert load_ratings(settings)["anime_id"].tolist() == [anime_id]

    # 5 liked x 6 disliked is just enough to train
    response = http.post("/train", follow_redirects=True)
    assert b"Model retrained" in response.data
    assert settings.model_path.exists() and settings.data_path.exists()

    # With too few labels the error comes back as a message instead of a crash
    settings.ratings_path.unlink()
    response = http.post("/train", follow_redirects=True)
    assert b"Need at least" in response.data
