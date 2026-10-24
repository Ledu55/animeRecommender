"""Minimal client for the MyAnimeList API v2.

Docs: https://myanimelist.net/apiconfig/references/api/v2
"""

import time
from collections.abc import Iterator
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.myanimelist.net/v2"
PAGE_SIZE = 500
ANIME_FIELDS = (
    "id,title,main_picture,mean,genres,media_type,status,num_episodes,"
    "rating,start_date,synopsis,studios,popularity"
)

COLUMNS = [
    "anime_id",
    "title",
    "synopsis",
    "genres",
    "studios",
    "mean_score",
    "num_episodes",
    "start_year",
    "popularity",
    "media_type",
    "status",
    "rating",
    "picture",
]


class MALClient:
    def __init__(self, client_id: str, session: requests.Session | None = None, delay: float = 1.0):
        self.session = session or requests.Session()
        self.session.headers["X-MAL-CLIENT-ID"] = client_id
        self.delay = delay

    def _paginate(self, url: str, params: dict[str, Any], limit: int | None = None) -> Iterator[dict]:
        fetched = 0
        while url and (limit is None or fetched < limit):
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            for item in payload["data"]:
                yield item
                fetched += 1
                if limit is not None and fetched >= limit:
                    return
            # The "next" link already carries every query parameter
            url, params = payload.get("paging", {}).get("next"), {}
            if url:
                time.sleep(self.delay)

    def ranking(self, limit: int) -> pd.DataFrame:
        """Top `limit` anime of the overall MAL ranking."""
        items = self._paginate(
            f"{BASE_URL}/anime/ranking",
            {"ranking_type": "all", "limit": PAGE_SIZE, "fields": ANIME_FIELDS, "nsfw": "false"},
            limit,
        )
        return to_frame(item["node"] for item in items)

    def user_list(self, username: str) -> pd.DataFrame:
        """Every anime on a user's public list, with `list_status` and `list_score` columns."""
        items = list(
            self._paginate(
                f"{BASE_URL}/users/{username}/animelist",
                {"limit": 1000, "fields": f"list_status,{ANIME_FIELDS}", "nsfw": "true"},
            )
        )
        df = to_frame(item["node"] for item in items)
        df["list_status"] = [item["list_status"]["status"] for item in items]
        df["list_score"] = [item["list_status"].get("score", 0) for item in items]
        return df


def parse_node(node: dict) -> dict:
    """Flatten one anime node of the API into a row."""
    start_date = node.get("start_date") or ""
    return {
        "anime_id": node["id"],
        "title": node["title"],
        "synopsis": node.get("synopsis") or "",
        "genres": "|".join(g["name"] for g in node.get("genres", [])),
        "studios": "|".join(s["name"] for s in node.get("studios", [])),
        "mean_score": node.get("mean"),
        "num_episodes": node.get("num_episodes") or None,  # 0 means "unknown"
        "start_year": int(start_date[:4]) if start_date[:4].isdigit() else None,
        "popularity": node.get("popularity"),
        "media_type": node.get("media_type") or "unknown",
        "status": node.get("status") or "",
        "rating": node.get("rating") or "",
        "picture": (node.get("main_picture") or {}).get("medium", ""),
    }


def to_frame(nodes) -> pd.DataFrame:
    return pd.DataFrame([parse_node(node) for node in nodes], columns=COLUMNS)
