import sys, os, re, shutil
sys.path.insert(0, os.path.dirname(__file__))

from website.app import create_app
from website.config import SEASON_YEAR
from website.data.loader import list_available_seasons, list_season_coaches

OUTPUT_DIR = "docs"
BASE_PATH = "/asl-hub"

# Current season keeps today's flat URL structure (docs/, docs/coach/<name>/,
# etc.) so existing published links don't change. Historical seasons snapshot
# the trailing-year route variants into a parallel docs/season/<year>/ tree.
CURRENT_SEASON_ROUTES = [
    ("/",               f"{OUTPUT_DIR}/index.html"),
    ("/records",        f"{OUTPUT_DIR}/records/index.html"),
    ("/power-rankings", f"{OUTPUT_DIR}/power-rankings/index.html"),
    ("/draft-heat-map", f"{OUTPUT_DIR}/draft-heat-map/index.html"),
    ("/acquisition",    f"{OUTPUT_DIR}/acquisition/index.html"),
]


def historical_season_routes(year: int) -> list[tuple[str, str]]:
    season_dir = f"{OUTPUT_DIR}/season/{year}"
    return [
        (f"/season/{year}",               f"{season_dir}/index.html"),
        (f"/power-rankings/{year}",       f"{season_dir}/power-rankings/index.html"),
        (f"/draft-heat-map/{year}",       f"{season_dir}/draft-heat-map/index.html"),
        (f"/acquisition/{year}",          f"{season_dir}/acquisition/index.html"),
    ]


def rewrite_paths(html):
    # Rewrite absolute internal paths (href="/ and src="/) to include BASE_PATH.
    # Excludes protocol-relative URLs (href="//...) and keeps external https:// untouched.
    html = re.sub(r'(href|src)="(/(?!/))', rf'\1="{BASE_PATH}\2', html)
    # Rewrite /static/ in JS backtick template literals (not caught by the HTML-attribute regex).
    html = html.replace('`/static/', f'`{BASE_PATH}/static/')
    return html

def render(client, route, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resp = client.get(route, follow_redirects=True)
    if resp.status_code != 200:
        print(f"  WARNING: {route} returned {resp.status_code}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rewrite_paths(resp.data.decode("utf-8")))

app = create_app()
app.jinja_env.globals['static_mode'] = True

if os.path.exists(OUTPUT_DIR):
    for item in os.listdir(OUTPUT_DIR):
        if item == ".git":
            continue
        p = os.path.join(OUTPUT_DIR, item)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
else:
    os.makedirs(OUTPUT_DIR)
shutil.copytree("website/static", f"{OUTPUT_DIR}/static")

page_count = 0

with app.test_client() as client:
    for route, out_path in CURRENT_SEASON_ROUTES:
        print(f"  {route} -> {out_path}")
        render(client, route, out_path)
        page_count += 1

    for name in list_season_coaches(SEASON_YEAR):
        out_path = f"{OUTPUT_DIR}/coach/{name}/index.html"
        print(f"  /coach/{name} -> {out_path}")
        render(client, f"/coach/{name}", out_path)
        page_count += 1

    for year in list_available_seasons():
        if year == SEASON_YEAR:
            continue
        for route, out_path in historical_season_routes(year):
            print(f"  {route} -> {out_path}")
            render(client, route, out_path)
            page_count += 1
        for name in list_season_coaches(year):
            out_path = f"{OUTPUT_DIR}/season/{year}/coach/{name}/index.html"
            print(f"  /coach/{name}/{year} -> {out_path}")
            render(client, f"/coach/{name}/{year}", out_path)
            page_count += 1

print(f"\nStatic site built -> {OUTPUT_DIR}/  ({page_count} pages)")
