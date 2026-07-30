"""
emailer.py — Sends the waiver newsletter as a styled HTML email via Gmail SMTP.

Configuration (.env)
---------------------
EMAIL_SENDER     Your Gmail address (e.g. you@gmail.com).
EMAIL_PASSWORD   Gmail app password (NOT your regular Gmail password).
                 To create one: myaccount.google.com → Security →
                 2-Step Verification → App passwords → Create → Mail.
EMAIL_RECIPIENT  Address to deliver to. Defaults to richardsmark@outlook.com.
"""

import logging
import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587

SENDER: str = os.getenv("EMAIL_SENDER", "")
RECIPIENT: str = os.getenv("EMAIL_RECIPIENT", "richardsmark@outlook.com")
PASSWORD: str = os.getenv("EMAIL_PASSWORD", "").replace(" ", "")

# ── SuperCoach colour palette (matched from website) ─────────────────────────
_BG       = "#f2f4f7"   # page background (light grey)
_NAV      = "#0f1117"   # top nav / header bar (near-black)
_CARD     = "#ffffff"   # card / table background
_TBL_HDR  = "#292932"   # table header row (dark charcoal)
_GREEN    = "#7bc943"   # SC lime green (active tabs, highlights)
_ORANGE   = "#e8751a"   # round badge / CTA orange
_RED      = "#dc2626"   # alerts / action-required
_TEXT     = "#1a1a2e"   # primary text (dark navy)
_MUTED    = "#5a5f7d"   # secondary text
_BORDER   = "#e2e8f0"   # subtle borders
_WHITE    = "#ffffff"
_BLUE     = "#3b82f6"   # ownership dot / link colour

# ── HTML email template (SuperCoach-matched styling) ─────────────────────────
# Uses %%PLACEHOLDER%% so CSS curly braces don't conflict with Python f-strings.
_HTML_TEMPLATE = (
    "<!DOCTYPE html>"
    "<html lang='en'><head>"
    "<meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>%%SUBJECT%%</title>"
    "<style>"
    # Reset + base
    f"body,table,td,p,a,h1,h2,h3,h4,ul,li{{margin:0;padding:0;}}"
    f"body{{background:{_BG};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;-webkit-text-size-adjust:100%;}}"
    f"a{{color:{_GREEN};text-decoration:none;}}"
    # Section headings — SC dark bar style matching the table headers
    f"h2{{background:{_TBL_HDR};color:{_WHITE};padding:10px 16px;"
    f"font-size:13px;font-weight:700;margin:24px 0 0;letter-spacing:0.5px;text-transform:uppercase;}}"
    f"h3{{color:{_TEXT};font-size:14px;font-weight:700;margin:16px 0 4px;padding:0;border-bottom:2px solid {_GREEN};padding-bottom:4px;}}"
    # Body copy
    f"p{{color:{_MUTED};font-size:13px;line-height:1.6;margin:6px 0;}}"
    f"em{{color:#4b5563;font-size:12px;font-style:italic;}}"
    f"strong{{color:{_TEXT};font-weight:700;}}"
    f"hr{{border:none;border-top:1px solid {_BORDER};margin:20px 0;}}"
    f"ul,ol{{color:{_MUTED};font-size:13px;padding-left:20px;margin:8px 0;}}"
    f"li{{margin-bottom:4px;}}"
    # Blockquotes — scoring rule, info banners
    f"blockquote{{margin:0 0 14px;padding:10px 14px;background:#f0f9e8;"
    f"border-left:4px solid {_GREEN};border-radius:0 5px 5px 0;}}"
    f"blockquote p{{margin:0;color:{_TEXT};font-size:13px;}}"
    f"blockquote h2{{background:none;color:{_TEXT};padding:0;border-radius:0;font-size:13px;"
    f"margin:0 0 6px;text-transform:none;letter-spacing:0;font-weight:700;}}"
    # Data tables — matches SC player list table
    f"table.data-table{{border-collapse:collapse;width:100%;font-size:12px;margin:0;"
    f"border:1px solid {_BORDER};border-top:none;}}"
    f"table.data-table thead tr{{background:{_TBL_HDR};}}"
    f"table.data-table th{{color:#ffffff;padding:8px 11px;"
    f"text-align:left;font-weight:600;white-space:nowrap;font-size:11px;letter-spacing:0.4px;text-transform:uppercase;}}"
    f"table.data-table td{{padding:9px 11px;border-bottom:1px solid {_BORDER};"
    f"color:{_TEXT};font-size:12px;vertical-align:middle;background:{_CARD};}}"
    f"table.data-table tr:last-child td{{border-bottom:none;}}"
    f"table.data-table tr:nth-child(even) td{{background:#f7f9fc;}}"
    f"table.data-table td strong{{color:{_TEXT};font-weight:700;}}"
    # Pickup badges — colour-coded like SC position tags
    f".badge-start{{display:inline-block;background:{_GREEN};color:{_WHITE};"
    f"padding:2px 9px;border-radius:3px;font-size:10px;font-weight:700;letter-spacing:0.3px;}}"
    f".badge-ceiling{{display:inline-block;background:{_BLUE};color:{_WHITE};"
    f"padding:2px 9px;border-radius:3px;font-size:10px;font-weight:700;letter-spacing:0.3px;}}"
    f".badge-depth{{display:inline-block;background:#e8eaf0;color:{_MUTED};"
    f"padding:2px 9px;border-radius:3px;font-size:10px;font-weight:600;}}"
    f".badge-longterm{{display:inline-block;background:#fff0d6;color:#9a5a00;"
    f"padding:2px 9px;border-radius:3px;font-size:10px;font-weight:600;}}"
    # Alert / action-required box
    f".action-required{{background:#fff0ed;border-left:4px solid {_ORANGE};"
    f"border-radius:0 6px 6px 0;padding:14px 16px;margin-bottom:18px;}}"
    "</style></head><body>"

    # ── Outer wrapper ──────────────────────────────────────────────────────
    f"<table width='100%' cellpadding='0' cellspacing='0' style='background:{_BG};padding:24px 0;'>"
    "<tr><td align='center'>"
    f"<table width='620' cellpadding='0' cellspacing='0' style='max-width:620px;width:100%;background:{_CARD};"
    "border-radius:8px;overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.12);'>"

    # ── SC-style top header bar ────────────────────────────────────────────
    f"<tr><td style='background:{_NAV};padding:0 24px;'>"
    # Green accent strip (mimics SC nav active indicator)
    f"<table width='100%' cellpadding='0' cellspacing='0'><tr>"
    f"<td style='padding:18px 0 14px;border-bottom:3px solid {_GREEN};'>"
    f"<div style='font-size:11px;font-weight:700;color:{_GREEN};letter-spacing:1.2px;"
    "text-transform:uppercase;margin-bottom:4px;'>SuperCoach Draft</div>"
    f"<div style='font-size:22px;font-weight:800;color:{_WHITE};line-height:1.2;letter-spacing:-0.3px;'>"
    "🏈 Waiver Wire Newsletter</div>"
    "</td>"
    # Round badge (SC orange pill)
    f"<td align='right' style='padding:18px 0 14px;border-bottom:3px solid {_GREEN};vertical-align:bottom;'>"
    f"<span style='display:inline-block;background:{_ORANGE};color:{_WHITE};"
    "font-size:11px;font-weight:800;padding:5px 14px;border-radius:4px;letter-spacing:0.5px;'>"
    "%%ROUND_BADGE%%</span>"
    "</td>"
    "</tr></table>"
    "</td></tr>"

    # ── Meta strip ────────────────────────────────────────────────────────
    f"<tr><td style='background:#1c1c2a;padding:10px 24px;'>"
    f"<table width='100%' cellpadding='0' cellspacing='0'><tr>"
    f"<td style='font-size:11px;color:#8892a4;'>Generated %%RUN_TIME%%</td>"
    f"<td align='right'><span style='font-size:11px;font-weight:700;color:{_ORANGE};'>"
    "⏰ Waiver closes Tue 4:00 AM AEST</span></td>"
    "</tr></table>"
    "</td></tr>"

    # ── Body ──────────────────────────────────────────────────────────────
    f"<tr><td style='padding:20px 24px 24px;background:{_CARD};'>%%BODY%%</td></tr>"

    # ── Footer ────────────────────────────────────────────────────────────
    f"<tr><td style='background:{_TBL_HDR};padding:14px 24px;'>"
    f"<p style='font-size:11px;color:#6b7280;margin:0;text-align:center;'>"
    f"SuperCoach Draft League 2026 &nbsp;·&nbsp; Automated Waiver Wire Newsletter &nbsp;·&nbsp; %%SENDER%%"
    "</p></td></tr>"

    "</table></td></tr></table>"
    "</body></html>"
)


# ── Markdown → HTML conversion ────────────────────────────────────────────────

_TH_STYLE = (
    f'style="background:{_TBL_HDR};color:#ffffff;padding:8px 11px;'
    'text-align:left;font-weight:600;white-space:nowrap;'
    'font-size:11px;letter-spacing:0.4px;text-transform:uppercase;"'
)
_TD_STYLE = (
    f'style="padding:9px 11px;border-bottom:1px solid {_BORDER};'
    f'color:{_TEXT};font-size:12px;vertical-align:middle;background:{_CARD};"'
)


def _inline_table_styles(html: str) -> str:
    """Inlines all table styles so email clients can't override them."""
    html = html.replace("<table>", '<table class="data-table" style="border-collapse:collapse;width:100%;font-size:12px;margin:0;">')
    html = html.replace("<th>", f"<th {_TH_STYLE}>")
    html = html.replace("<td>", f"<td {_TD_STYLE}>")
    # Strip <strong> tags inside <th> — markdown bolds headers but strong overrides white colour
    html = re.sub(r"(<th [^>]+>)<strong>(.*?)</strong>", r"\1\2", html)
    return html


def _badge(text: str) -> str:
    """Wraps pickup-label text in a styled badge span."""
    cls = {
        "Start now":    "badge-start",
        "Ceiling pick": "badge-ceiling",
        "Depth add":    "badge-depth",
        "Long-term add":"badge-longterm",
    }.get(text.strip(), "badge-depth")
    return f'<span class="{cls}">{text.strip()}</span>'


def _style_pickup_labels(html: str) -> str:
    """Turns <td>Start now</td> etc. into badged cells."""
    labels = ["Start now", "Ceiling pick", "Depth add", "Long-term add"]
    for label in labels:
        html = html.replace(f"<td>{label}</td>", f"<td>{_badge(label)}</td>")
    return html


def _extract_header_meta(md_text: str) -> tuple[str, str]:
    """
    Pulls round_label and run_time from the newsletter header lines.
    Falls back gracefully if the format changes.
    """
    round_label = "Waiver Wire Newsletter"
    run_time = ""

    # Line 1: # 🏈 Waiver Wire Newsletter — Round 2, 2026
    m = re.search(r"#\s+.*?Newsletter\s+[—–-]+\s+(.+)", md_text)
    if m:
        round_label = m.group(1).strip()

    # Line 2: *Generated: Monday, 24 March 2026 at ...*
    m = re.search(r"Generated:\s*(.+?)\s*\|", md_text)
    if m:
        run_time = m.group(1).strip()

    return round_label, run_time


def _extract_bat_notice(md_text: str) -> tuple[str, str]:
    """
    Strips the bat-file action-required block from the top of the markdown
    (it contains a raw file path, better rendered as a styled HTML alert).
    Returns (cleaned_md, alert_html).
    """
    alert_html = ""
    # The block starts with a blockquote h2 and ends before the first real H1
    pattern = re.compile(r"(^>\s*##\s*🔔 ACTION REQUIRED.*?)(?=\n# )", re.DOTALL | re.MULTILINE)
    m = pattern.search(md_text)
    if m:
        alert_html = (
            '<div class="action-required">'
            '<strong style="color:#dc2626;font-size:15px;">🔔 ACTION REQUIRED — Task Schedule Updated</strong>'
            '<p style="margin:8px 0 0;font-size:13px;color:#7f1d1d;">'
            'The AFL fixture has changed. Re-run <strong>waiver_schedule.bat</strong> as '
            'Administrator to update your scheduled tasks. The file is at:<br>'
            '<code style="font-size:12px;background:#fee2e2;padding:2px 6px;border-radius:3px;">'
            'waiver_schedule.bat</code> in your project folder.</p></div>'
        )
        md_text = md_text[: m.start()] + md_text[m.end() :]

    return md_text.strip(), alert_html


def _strip_h1_and_meta(md_text: str) -> str:
    """
    Removes the H1 title line and the *Generated: …* meta line — these are
    rendered in the HTML header block instead.
    """
    lines = md_text.splitlines()
    out = []
    skip_next_blank = False
    for line in lines:
        if line.startswith("# 🏈"):
            skip_next_blank = True
            continue
        if line.startswith("*Generated:"):
            skip_next_blank = True
            continue
        if skip_next_blank and line.strip() == "":
            skip_next_blank = False
            continue
        skip_next_blank = False
        out.append(line)
    return "\n".join(out)


def _build_html_email(md_text: str, round_label: str) -> str:
    """Converts newsletter Markdown to a fully styled HTML email."""
    try:
        import markdown as md_lib
    except ImportError:
        log.warning("'markdown' package not installed — sending plain-text HTML fallback.")
        safe = md_text.replace("&", "&amp;").replace("<", "&lt;")
        return f"<html><body><pre style='font-family:monospace;font-size:13px'>{safe}</pre></body></html>"

    round_label_parsed, run_time = _extract_header_meta(md_text)
    if round_label_parsed:
        round_label = round_label_parsed

    md_text, alert_html = _extract_bat_notice(md_text)
    md_text = _strip_h1_and_meta(md_text)

    html_body = md_lib.markdown(
        md_text,
        extensions=["tables", "sane_lists"],
        output_format="html",
    )
    html_body = _inline_table_styles(html_body)
    html_body = _style_pickup_labels(html_body)

    # Extract just "Round N" for the badge (drop ", 2026" suffix if present)
    import re as _re
    badge = _re.sub(r",?\s*\d{4}$", "", round_label).strip() or round_label

    return (
        _HTML_TEMPLATE
        .replace("%%SUBJECT%%", f"Waiver Wire Newsletter — {round_label}")
        .replace("%%ROUND_BADGE%%", badge.upper())
        .replace("%%RUN_TIME%%", run_time)
        .replace("%%BODY%%", alert_html + html_body)
        .replace("%%SENDER%%", SENDER)
    )


# ── Public send function ──────────────────────────────────────────────────────

def send_newsletter(md_path: Path, round_label: str) -> bool:
    """
    Emails the newsletter as a styled HTML email.

    Parameters
    ----------
    md_path      : path to the generated .md file
    round_label  : e.g. "Round 3, 2026" — used in the subject line

    Returns True on success, False on any failure.
    """
    if not SENDER:
        log.warning("EMAIL_SENDER not set in .env — skipping email.")
        return False
    if not PASSWORD:
        log.warning(
            "EMAIL_PASSWORD not set in .env — skipping email. "
            "Add a Gmail app password: myaccount.google.com → Security → "
            "2-Step Verification → App passwords."
        )
        return False

    try:
        md_text = md_path.read_text(encoding="utf-8")
    except Exception as exc:
        log.error("Could not read newsletter file for email: %s", exc)
        return False

    html = _build_html_email(md_text, round_label)

    msg = MIMEMultipart("alternative")
    msg["From"] = f"Waiver Wire Newsletter <{SENDER}>"
    msg["To"] = RECIPIENT
    msg["Subject"] = f"🏈 Waiver Wire Newsletter — {round_label}"

    # Plain-text fallback (stripped-down version)
    msg.attach(MIMEText(md_text, "plain", "utf-8"))
    # HTML version (email clients prefer the last part)
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SENDER, PASSWORD)
            server.sendmail(SENDER, RECIPIENT, msg.as_string())
        log.info("Newsletter emailed to %s.", RECIPIENT)
        return True
    except smtplib.SMTPAuthenticationError:
        log.error(
            "Gmail authentication failed. EMAIL_PASSWORD must be an app password, "
            "not your regular Gmail password. Create one at: "
            "myaccount.google.com → Security → 2-Step Verification → App passwords."
        )
        return False
    except Exception as exc:
        log.error("Failed to send newsletter email: %s", exc)
        return False
