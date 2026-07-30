"""
app.py — Flask application factory.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'website' package resolves correctly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import datetime, timedelta
from flask import Flask, jsonify
from flask_caching import Cache

cache = Cache()

_REFRESH_LIMIT = 5
_REFRESH_WINDOW = timedelta(hours=1)
_refresh_timestamps: list[datetime] = []


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["CACHE_TYPE"] = "SimpleCache"
    app.config["CACHE_DEFAULT_TIMEOUT"] = 300

    cache.init_app(app)

    # Inject cache into loader so it can memoize
    import website.data.loader as loader
    loader.cache = cache

    from website.routes.season         import bp as season_bp
    from website.routes.coach          import bp as coach_bp
    from website.routes.records        import bp as records_bp
    from website.routes.power_rankings import bp as pr_bp
    from website.routes.draft          import bp as draft_bp
    from website.routes.acquisition    import bp as acq_bp

    app.register_blueprint(season_bp)
    app.register_blueprint(coach_bp,   url_prefix="/coach")
    app.register_blueprint(records_bp, url_prefix="/records")
    app.register_blueprint(pr_bp,      url_prefix="/power-rankings")
    app.register_blueprint(draft_bp,   url_prefix="/draft-heat-map")
    app.register_blueprint(acq_bp,     url_prefix="/acquisition")

    def _current_round() -> int:
        import pandas as pd
        from website.config import DATA_LIVE_DIR
        try:
            return int(pd.read_csv(DATA_LIVE_DIR / "ladder.csv", usecols=["round"])["round"].max())
        except Exception:
            return 1

    @app.get("/api/round-status")
    def round_status():
        from waiver.fixture_schedule import fetch_fixture
        round_num = _current_round()
        df = fetch_fixture()
        if df.empty:
            return jsonify({"round": round_num, "complete": True, "last_game": None})
        games = df[df["round_num"] == round_num]
        if games.empty:
            return jsonify({"round": round_num, "complete": True, "last_game": None})
        last_game = games["game_dt_aest"].max()
        from datetime import timezone
        now = datetime.now(tz=timezone.utc).astimezone(last_game.tzinfo)
        complete = now > last_game
        return jsonify({
            "round": round_num,
            "complete": complete,
            "last_game": last_game.strftime("%a %d %b %H:%M AEST"),
        })

    @app.post("/api/refresh")
    def refresh_cache():
        from waiver.fixture_schedule import round_is_complete
        round_num = _current_round()
        if not round_is_complete(round_num):
            return jsonify({"status": "round_in_progress", "message": f"Round {round_num} still in progress"}), 403
        now = datetime.now()
        cutoff = now - _REFRESH_WINDOW
        _refresh_timestamps[:] = [t for t in _refresh_timestamps if t > cutoff]
        if len(_refresh_timestamps) >= _REFRESH_LIMIT:
            return jsonify({"status": "rate_limited"}), 429
        _refresh_timestamps.append(now)
        from website.data.api_ingest import refresh_2026_data
        refresh_2026_data()
        cache.clear()
        return jsonify({"status": "ok"})

    @app.get("/api/data-freshness")
    def data_freshness():
        import os, re
        from datetime import datetime
        import pandas as pd
        from website.config import DATA_LIVE_DIR, PROCESSED_DATA_DIR

        def mtime(path):
            try:
                ts = os.path.getmtime(path)
                return datetime.fromtimestamp(ts).strftime("%d %b %Y %H:%M")
            except OSError:
                return None

        # SuperCoach: ladder.csv written on every api_ingest refresh
        ladder_path = DATA_LIVE_DIR / "ladder.csv"
        sc_time = mtime(ladder_path)
        sc_round = None
        try:
            sc_round = int(pd.read_csv(ladder_path, usecols=["round"])["round"].max())
        except Exception:
            pass

        # Fanfooty: most recently modified processed fanfooty CSV
        ff_files = sorted(
            PROCESSED_DATA_DIR.glob("*_round_*_fanfooty_data.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        ff_time = None
        ff_round = None
        if ff_files:
            ff_time = mtime(ff_files[0])
            m = re.search(r"_round_(\d+)_", ff_files[0].name)
            if m:
                ff_round = int(m.group(1))

        return jsonify({
            "supercoach": sc_time, "supercoach_round": sc_round,
            "fanfooty":   ff_time, "fanfooty_round":   ff_round,
        })

    @app.post("/api/cache-clear")
    def clear_cache():
        cache.clear()
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=5000)
