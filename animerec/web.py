"""Web front-end: a Flask app and a static-site export sharing the same templates."""

from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for
from jinja2 import Environment, FileSystemLoader, select_autoescape

from animerec import pipeline
from animerec.config import Settings
from animerec.mal import MALClient
from animerec.ratings import load_candidates, load_ratings, rating_queue, save_rating

TEMPLATES = Path(__file__).parent / "templates"
QUEUE_SIZE = 12

app = Flask(__name__)


def client(settings: Settings) -> MALClient:
    settings.require_mal()
    return MALClient(settings.mal_client_id)


@app.route("/")
def index():
    return render_template("index.html", interactive=True, **pipeline.load_recommendations(Settings()))


@app.get("/rate")
def rate_page():
    settings = Settings()
    mal = client(settings)
    ratings = load_ratings(settings)
    queue = rating_queue(
        load_candidates(settings, mal), mal.user_list(settings.mal_username), ratings, settings, QUEUE_SIZE
    )
    return render_template(
        "rate.html",
        interactive=True,
        queue=queue.to_dict("records"),
        rated=len(ratings),
        has_model=settings.model_path.exists(),
        message=request.args.get("message"),
    )


@app.post("/rate")
def rate():
    settings = Settings()
    choice = request.form["choice"]
    if choice in ("like", "dislike"):
        candidates = load_candidates(settings, client(settings))
        match = candidates[candidates["anime_id"] == int(request.form["anime_id"])]
        if match.empty:
            return jsonify(error="unknown anime"), 404
        save_rating(settings, match.iloc[0].to_dict(), liked=choice == "like")
    if request.accept_mimetypes.best == "application/json":
        return jsonify(ok=True)
    return redirect(url_for("rate_page"))


@app.post("/train")
def train():
    settings = Settings()
    mal = client(settings)
    try:
        pipeline.train(settings, mal)
        pipeline.update(settings, mal)
        message = "Model retrained and recommendations refreshed."
    except pipeline.PipelineError as error:
        message = str(error)
    return redirect(url_for("rate_page", message=message))


def build_site(settings: Settings) -> Path:
    """Render the recommendations to a standalone index.html (for GitHub Pages or any static host)."""
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape())
    html = env.get_template("index.html").render(interactive=False, **pipeline.load_recommendations(settings))
    settings.site_dir.mkdir(parents=True, exist_ok=True)
    out = settings.site_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out
