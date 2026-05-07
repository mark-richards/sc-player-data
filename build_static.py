import sys, os, re, shutil
sys.path.insert(0, os.path.dirname(__file__))

from website.app import create_app

OUTPUT_DIR = "docs"
BASE_PATH = "/asl-hub"

COACH_NAMES = ['Anthony', 'James', 'Jordan', 'Lester', 'Luke', 'Mark', 'Paul', 'Simon']

ROUTES = [
    ("/",                    f"{OUTPUT_DIR}/index.html"),
    ("/records",             f"{OUTPUT_DIR}/records/index.html"),
    ("/power-rankings",      f"{OUTPUT_DIR}/power-rankings/index.html"),
    ("/draft-heat-map",      f"{OUTPUT_DIR}/draft-heat-map/index.html"),
    ("/acquisition",         f"{OUTPUT_DIR}/acquisition/index.html"),
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

with app.test_client() as client:
    for route, out_path in ROUTES:
        print(f"  {route} -> {out_path}")
        render(client, route, out_path)

    for name in COACH_NAMES:
        out_path = f"{OUTPUT_DIR}/coach/{name}/index.html"
        print(f"  /coach/{name} -> {out_path}")
        render(client, f"/coach/{name}", out_path)

print(f"\nStatic site built -> {OUTPUT_DIR}/  ({len(ROUTES) + len(COACH_NAMES)} pages)")
