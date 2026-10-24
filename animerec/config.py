"""Settings read from environment variables (and an optional .env file)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")


def _list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    mal_client_id: str = field(default_factory=lambda: os.getenv("MAL_CLIENT_ID", ""))
    mal_username: str = field(default_factory=lambda: os.getenv("MAL_USERNAME", ""))
    # How many titles of the MAL ranking are scored as candidates
    num_candidates: int = field(default_factory=lambda: int(os.getenv("NUM_CANDIDATES", "2000")))
    media_types: list[str] = field(default_factory=lambda: _list(os.getenv("MEDIA_TYPES", "tv,ova,ona")))
    # A score on your list at or above this value counts as "I liked it"
    like_threshold: int = field(default_factory=lambda: int(os.getenv("LIKE_THRESHOLD", "7")))
    top_n: int = field(default_factory=lambda: int(os.getenv("TOP_N", "50")))
    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
    )
    data_path: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_PATH", ROOT / "data" / "recommendations.json"))
    )
    site_dir: Path = field(default_factory=lambda: Path(os.getenv("SITE_DIR", ROOT / "site")))
    # Likes/dislikes given on the /rate page; commit it so CI trains with them too
    ratings_path: Path = field(
        default_factory=lambda: Path(os.getenv("RATINGS_PATH", ROOT / "data" / "ratings.json"))
    )
    # Local copy of the MAL ranking, so the /rate page doesn't hit the API on every request
    candidates_path: Path = field(
        default_factory=lambda: Path(os.getenv("CANDIDATES_PATH", ROOT / "data" / "candidates.json"))
    )

    def require_mal(self) -> None:
        missing = [
            name
            for name, value in (("MAL_CLIENT_ID", self.mal_client_id), ("MAL_USERNAME", self.mal_username))
            if not value
        ]
        if missing:
            raise SystemExit(f"Missing environment variables: {', '.join(missing)} (see .env.example)")
