"""Command line: `animerec train | update | build-site | serve`."""

import argparse
import logging

from animerec.config import Settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="animerec", description="Anime recommender based on your MAL list")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="train the model on your MAL list")
    commands.add_parser("update", help="score the MAL ranking and write the recommendations")
    commands.add_parser("build-site", help="render the recommendations to a static index.html")
    serve = commands.add_parser("serve", help="run the development web server")
    serve.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings()

    if args.command in ("train", "update"):
        from animerec import pipeline
        from animerec.mal import MALClient

        settings.require_mal()
        try:
            getattr(pipeline, args.command)(settings, MALClient(settings.mal_client_id))
        except pipeline.PipelineError as error:
            raise SystemExit(str(error)) from None
    elif args.command == "build-site":
        from animerec.web import build_site

        logging.info("Site written to %s", build_site(settings))
    else:
        from animerec.web import app

        app.run(debug=True, port=args.port)


if __name__ == "__main__":
    main()
