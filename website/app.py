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
from flask import Flask, jsonify, request, url_for
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

    # build_season_awards() runs an expensive Monte Carlo sim + draft heat-map
    # ranking per year — every Season Summary page calls it once, and the
    # Hall of Fame page calls it once per completed season. Caching here
    # benefits both.
    import website.data.awards as awards_module
    awards_module.cache = cache

    # build_round_value_baseline() loops every historical season's draft data
    # to compute the Worst Draft Pick awards' historical baseline — cache it
    # the same way.
    import website.data.draft_data as draft_data_module
    draft_data_module.cache = cache

    # build_notable_season_records() also runs a Monte Carlo sim per year
    # (for the Unluckiest Spoon Season leaderboard) — cache it too.
    import website.data.hall_of_fame as hall_of_fame_module
    hall_of_fame_module.cache = cache

    from website.routes.season         import bp as season_bp
    from website.routes.coach          import bp as coach_bp
    from website.routes.records        import bp as records_bp
    from website.routes.power_rankings import bp as pr_bp
    from website.routes.draft          import bp as draft_bp
    from website.routes.acquisition    import bp as acq_bp
    from website.routes.head_to_head   import bp as h2h_bp

    app.register_blueprint(season_bp)
    app.register_blueprint(coach_bp,   url_prefix="/coach")
    app.register_blueprint(records_bp, url_prefix="/records")
    app.register_blueprint(pr_bp,      url_prefix="/power-rankings")
    app.register_blueprint(draft_bp,   url_prefix="/draft-heat-map")
    app.register_blueprint(acq_bp,     url_prefix="/acquisition")
    app.register_blueprint(h2h_bp,     url_prefix="/head-to-head")

    @app.context_processor
    def inject_season_nav():
        from website.config import SEASON_YEAR
        from website.data.loader import list_available_seasons, list_season_coaches

        selected_year = (request.view_args or {}).get("year", SEASON_YEAR)

        # Whether the CURRENT page's own route accepts a year — all-time
        # pages like the Hall of Fame (/records) don't, so the nav's year
        # switcher would otherwise look clickable but always just reload the
        # same page with no year arg attached, which reads as broken rather
        # than as "this page covers every season".
        from flask import current_app
        _rules = [r for r in current_app.url_map.iter_rules() if r.endpoint == request.endpoint]
        current_route_has_year = any("year" in r.arguments for r in _rules)

        def year_href(endpoint, year, **extra_args):
            # The current season is exported at the flat root paths (not
            # /season/<year>/...) by build_static.py, so drop the year arg
            # to resolve to that same bare route on both the live app and
            # the static export. Historical years get the year arg only if
            # the target endpoint actually accepts one.
            from flask import current_app
            args = dict(extra_args)
            if year != SEASON_YEAR:
                rules = [r for r in current_app.url_map.iter_rules() if r.endpoint == endpoint]
                takes_year = any("year" in r.arguments for r in rules)
                if takes_year:
                    args["year"] = year
            try:
                return url_for(endpoint, **args)
            except Exception:
                return "#"

        def year_url(year):
            # Switch to a different year on the SAME page (used by the nav's
            # year-switcher dropdown).
            args = {k: v for k, v in (request.view_args or {}).items() if k != "year"}
            return year_href(request.endpoint, year, **args)

        def season_url(endpoint, **extra_args):
            # Link to a DIFFERENT page (e.g. the Draft Heat Map nav item)
            # that preserves whichever season is currently selected, instead
            # of always resetting to the current season.
            return year_href(endpoint, selected_year, **extra_args)

        return dict(
            available_seasons=list_available_seasons(),
            year_url=year_url,
            season_url=season_url,
            year_href=year_href,
            nav_coaches=list_season_coaches(selected_year),
            current_route_has_year=current_route_has_year,
        )

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
