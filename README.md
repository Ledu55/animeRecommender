# Anime Recommender

Suggests anime based on your own [MyAnimeList](https://myanimelist.net) history.

A model learns your taste from the scores on your MAL list, then ranks the most
popular titles on MAL that you haven't watched yet. The picks are published as a
small web page.

## How it works

1. **Labels** come from two places:
   - **Your MAL list.** A score of `LIKE_THRESHOLD` (default 7) or higher means
     *liked*. A lower score, or a dropped title, means *disliked*. Unscored
     titles are ignored.
   - **The `/rate` page.** It shows anime you haven't watched and lets you answer
     👍 or 👎. Once a model exists, it asks about the titles the model is least
     sure about (active learning). Your answers are saved to `data/ratings.json`
     and override list scores. This is useful when your list has few low scores.
2. **Features**: TF-IDF of the title and synopsis, genres, studios, media type,
   plus MAL mean score, episode count, start year and popularity. They are
   scaled with `MaxAbsScaler`.
3. **Model**: a soft-voting ensemble of Logistic Regression (30%) and LightGBM
   (70%). Preprocessing and model live in a single scikit-learn `Pipeline`.
   Training logs the cross-validated average precision.
4. **Recommendations**: the top `NUM_CANDIDATES` titles of the MAL ranking are
   scored. Everything already on your list is skipped, except plan-to-watch. The
   best `TOP_N` are written to `data/recommendations.json`.

## Running locally

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env             # then fill in MAL_CLIENT_ID and MAL_USERNAME
```

Create the client ID at <https://myanimelist.net/apiconfig>. Your anime list must
be public.

```bash
animerec serve        # http://localhost:5000; rate anime at /rate, then click "Retrain"
animerec train        # same as the button: fit the model -> models/model.joblib
animerec update       # score the MAL ranking           -> data/recommendations.json
animerec build-site   # static page                     -> site/index.html
```

Commit `data/ratings.json` so that the scheduled GitHub Actions run trains with
your ratings too.

## Deploying

**GitHub Pages (no server).** The workflow
[`.github/workflows/pages.yml`](.github/workflows/pages.yml) retrains the model,
refreshes the picks and publishes the static page every Monday. To set it up:

1. Add the repository secrets `MAL_CLIENT_ID` and `MAL_USERNAME`.
2. Under *Settings → Pages*, set the source to *GitHub Actions*.
3. Run the workflow once by hand from the *Actions* tab.

**Docker** (Render, Fly.io, Railway, ...). Run `animerec update` first, so that
`data/recommendations.json` is baked into the image:

```bash
docker build -t animerec .
docker run -p 8000:8000 animerec
```

## Development

```bash
pytest
ruff check . && ruff format .
nbstripout --install   # strip notebook outputs on commit
```

The [`notebooks/`](notebooks) folder holds the original 2022 exploration: the
MySQL ETL, model comparison and active learning. They are kept for reference and
are not used by the package.

## Next steps

- Multi-user mode: train one model per MAL username on demand.
- Active learning again: surface the titles the model is least sure about so you
  can rate them.

## References

- MyAnimeList API v2: <https://myanimelist.net/apiconfig/references/api/v2>
- What is anime: <https://en.wikipedia.org/wiki/Anime>
