"""Opponent-scout look for the app, modeled on the Cornell D4 gt tables (gt_theme_538):
a title band and column labels in the team's colors, bold uppercase labels, zebra rows,
centered cells, a small blue "rank/N" stacked under each value, and cells shaded from
teal (ranked best) to pink (ranked worst).
"""
from __future__ import annotations

import html
import re

import pandas as pd

GOOD, BAD = "#B2DFDB", "#FFCDD2"     # the D4 data_color() palette (good -> bad)
ZEBRA = "#F9F9F9"
CORNELL = ("#B31B1B", "#FFFFFF")

# nflverse-style primary / secondary colors
TEAM_COLORS = {
    "ARI": ("#97233F", "#000000"), "ATL": ("#A71930", "#000000"), "BAL": ("#241773", "#9E7C0C"),
    "BUF": ("#00338D", "#C60C30"), "CAR": ("#0085CA", "#101820"), "CHI": ("#0B162A", "#C83803"),
    "CIN": ("#FB4F14", "#000000"), "CLE": ("#311D00", "#FF3C00"), "DAL": ("#002244", "#B0B7BC"),
    "DEN": ("#FB4F14", "#002244"), "DET": ("#0076B6", "#B0B7BC"), "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#A71930"), "IND": ("#002C5F", "#A2AAAD"), "JAX": ("#006778", "#D7A22A"),
    "KC": ("#E31837", "#FFB81C"), "LV": ("#000000", "#A5ACAF"), "LAC": ("#0080C6", "#FFC20E"),
    "LA": ("#003594", "#FFA300"), "MIA": ("#008E97", "#FC4C02"), "MIN": ("#4F2683", "#FFC62F"),
    "NE": ("#002244", "#C60C30"), "NO": ("#D3BC8D", "#101820"), "NYG": ("#0B2265", "#A71930"),
    "NYJ": ("#125740", "#000000"), "PHI": ("#004C54", "#A5ACAF"), "PIT": ("#FFB612", "#101820"),
    "SEA": ("#002244", "#69BE28"), "SF": ("#AA0000", "#B3995D"), "TB": ("#D50A0A", "#34302B"),
    "TEN": ("#0C2340", "#4B92DB"), "WAS": ("#5A1414", "#FFB612"),
}


# ---------------------------------------------------------------- color helpers
def _rgb(hex_: str) -> tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lum(hex_: str) -> float:
    r, g, b = (c / 255 for c in _rgb(hex_))
    return 0.299 * r + 0.587 * g + 0.114 * b


def on_color(bg: str) -> str:
    """Readable text color on a filled background."""
    return "#FFFFFF" if _lum(bg) < 0.6 else "#101820"


def team_ink(team: str) -> str:
    """The team's color as text on white (falls back to its darker color if the primary is light)."""
    p, s = TEAM_COLORS.get(team, ("#333333", "#333333"))
    if _lum(p) < 0.6:
        return p
    return s if _lum(s) < 0.6 else "#333333"


def team_ink_dark(team: str) -> str:
    """The team's color as text on a dark background (lightened if both team colors are dark)."""
    p, s = TEAM_COLORS.get(team, ("#CCCCCC", "#CCCCCC"))
    if _lum(p) >= 0.38:
        return p
    if _lum(s) >= 0.38:
        return s
    return mix(p, "#FFFFFF", 0.6)


def mix(a: str, b: str, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    ra, rb = _rgb(a), _rgb(b)
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ra, rb))


def rank_fill(rank: float, n: int) -> str:
    """Rank 1 = teal, rank n = pink, like D4's col_numeric over ranks."""
    return mix(GOOD, BAD, (rank - 1) / max(n - 1, 1))


def ranks(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """R's rank(-x, ties.method = "max"): tied teams all get the worse rank."""
    return (s if not higher_is_better else -s).rank(method="max").astype(int)


# ---------------------------------------------------------------- page CSS
def page_css(primary: str, text: str) -> str:
    """Neutral 538-style page: white title blocks with a dark rule, no colored bars.
    The report color is only an accent (buttons, tab underline, page rule)."""
    return f"""
<style>
:root {{ --sc-ink:#222222; --sc-head:#111111; --sc-muted:#666666; --sc-faint:#999999; --sc-rule:#222222;
    --sc-line:#E4E4E4; --sc-zebra:#F9F9F9; --sc-bg:#FFFFFF; }}
:root[data-sc-theme="dark"] {{ --sc-ink:#E4E6EA; --sc-head:#F3F4F6; --sc-muted:#A3A8B0; --sc-faint:#6F757E;
    --sc-rule:#C9CDD3; --sc-line:#2A2F38; --sc-zebra:#151920; --sc-bg:#0E1117; }}
@import url('https://fonts.googleapis.com/css2?family=Chivo:wght@300;400;700;900&display=swap');
.block-container {{ padding-top: 2.2rem; }}
h1, h2, h3, h4 {{ font-weight: 900 !important; letter-spacing: -0.01em; }}
[data-testid="stTab"], [data-testid="stTab"] p, [role="tab"], [role="tab"] p {{
    font-family: 'Chivo', sans-serif !important; font-weight: 900 !important; color: var(--sc-ink);
    text-transform: uppercase; letter-spacing: 0.03em; font-size: 0.9rem !important; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {primary} !important; }}
[data-testid="stTabs"] .react-aria-SelectionIndicator, [data-baseweb="tab-highlight"] {{
    background-color: {primary} !important; }}
.stButton button[kind="primary"] {{ background: {primary}; border-color: {primary}; color: {text};
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; }}
.stButton button[kind="primary"]:hover {{ filter: brightness(1.12); border-color: {primary}; color: {text}; }}
.stButton button p {{ font-weight: 700; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 2px; }}

.scout-band {{ color: var(--sc-head); padding: 2px 0 6px; border-bottom: 2px solid var(--sc-rule); }}
.scout-band .t {{ font-size: 1.5rem; font-weight: 900; line-height: 1.15; }}
.scout-band .s {{ font-size: 0.88rem; font-weight: 400; color: var(--sc-muted); margin-top: 3px; }}
.scout-band.page {{ border-bottom: 4px solid {primary}; padding-bottom: 8px; }}
.scout-band.page .t {{ font-size: 2.3rem; }}
.scout-band.mini {{ padding: 0 0 4px; }}
.scout-band.mini .t {{ font-size: 1rem; text-transform: uppercase; letter-spacing: 0.03em; }}
.scout-wrap .scout-band {{ border-bottom: none; }}

.scout-wrap {{ margin-bottom: 0.6rem; }}
.scout-scroll {{ overflow: auto; }}
table.scout {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; color: var(--sc-ink);
    font-family: 'Chivo', sans-serif; font-variant-numeric: tabular-nums; }}
table.scout thead th {{ background: var(--sc-bg) !important; color: var(--sc-ink) !important; text-transform: uppercase;
    font-weight: 700; font-size: 0.76rem; letter-spacing: 0.02em; padding: 6px 6px 6px; text-align: center;
    white-space: nowrap; border-top: 1px solid var(--sc-rule); border-bottom: 2px solid var(--sc-rule) !important;
    position: sticky; top: 0; z-index: 1; }}
table.scout.sortable thead th {{ cursor: pointer; user-select: none; }}
table.scout.sortable thead th:hover {{ color: {primary} !important; }}
table.scout.sortable thead th::after {{ content: " ↕"; color: var(--sc-faint); font-size: 0.7rem; }}
table.scout.sortable thead th[data-dir="asc"]::after {{ content: " ▲"; color: var(--sc-ink); }}
table.scout.sortable thead th[data-dir="desc"]::after {{ content: " ▼"; color: var(--sc-ink); }}
table.scout td {{ padding: 6px 6px; text-align: center; white-space: nowrap;
    border-bottom: 1px solid var(--sc-line); vertical-align: middle; }}
table.scout tbody tr:nth-child(odd) td:not(.f) {{ background-color: var(--sc-zebra); }}
table.scout td.l, table.scout th.l {{ text-align: left; }}
table.scout td .v {{ font-weight: 700; display: block; line-height: 1.15; }}
table.scout td .r {{ font-size: 0.68rem; font-weight: 700; color: var(--sc-muted); display: block; line-height: 1.1; }}
table.scout td .m {{ color: var(--sc-faint); }}
table.scout td.b {{ font-weight: 700; }}
table.scout img {{ height: 24px; width: 24px; object-fit: contain; vertical-align: middle; margin-right: 6px; }}
.tm {{ font-weight: 900; color: var(--tl); }}
:root[data-sc-theme="dark"] .tm {{ color: var(--td); }}
table.scout td.f, table.scout td.f b {{ color: #111; }}
table.scout img.vi, img.vi {{ height: 13px; width: auto; margin: 0; vertical-align: middle; object-fit: contain; }}
table.scout thead th img.vi {{ height: 11px; vertical-align: -1px; }}
.src-txt {{ font-size: 0.66rem; font-weight: 700; color: var(--sc-muted); letter-spacing: 0.04em; text-transform: uppercase; }}
.scout-card-mid img.vi {{ height: 11px; margin-top: 3px; }}
.vi-head {{ font-weight: 900; font-size: 1.15rem; display: flex; align-items: center; gap: 8px; margin: 0.2rem 0 0.4rem; }}
.vi-head img {{ height: 18px; }}
/* team pickers: logo grid inside a popover */
div[data-testid="stPopover"] button img, div[data-testid="stPopoverBody"] button img {{
    height: 22px !important; max-height: 22px !important; width: auto !important; vertical-align: middle; }}
div[data-testid="stPopover"] > div > button {{ min-height: 44px; justify-content: flex-start; }}
.tp-label {{ font-size: 0.875rem; margin-bottom: 0.25rem; color: var(--sc-ink); }}
.scout-foot {{ font-size: 0.8rem; color: var(--sc-muted); padding: 5px 2px 0; }}
.scout-card-mid {{ text-align: center; line-height: 1.2; font-size: 0.72rem; color: var(--sc-muted);
    text-transform: uppercase; letter-spacing: 0.03em; font-weight: 700; }}
.scout-card-mid .at {{ font-size: 1.05rem; color: var(--sc-ink); font-weight: 900; }}
.scout-card-mid .lock {{ color: var(--sc-ink); }}

/* VegasInsider logo: dark lettering on light pages, the original white lettering on dark */
.vi-d {{ display: none !important; }}
:root[data-sc-theme="dark"] .vi-l {{ display: none !important; }}
:root[data-sc-theme="dark"] .vi-d {{ display: inline !important; }}
.sc-muted {{ color: var(--sc-muted); }}
.sc-faint {{ color: var(--sc-faint); }}
/* Team outlook keeps the D4 team-colored bands (set per team by team_band_css). */
.team-band .scout-band {{ padding: 10px 12px 8px; }}
.team-band .scout-band .s {{ color: inherit; opacity: 0.9; }}
</style>
"""


def team_band_css(primary: str) -> str:
    text = on_color(primary)
    return (f"<style>.team-band .scout-band {{ background:{primary}; color:{text}; }}"
            f".team-band table.scout thead th {{ background:{primary} !important; color:{text} !important;"
            f" border-top:none; border-bottom:2px solid {text} !important; }}"
            f".team-band table.scout.sortable thead th::after, .team-band table.scout.sortable thead th[data-dir]::after"
            f" {{ color:{text}; }}</style>")


def pick_button_css(game_buttons: list[tuple[str, str, bool]]) -> str:
    """Each game button gets its team's color as an accent; a picked team fills with its colors."""
    rules = []
    for key, team, picked in game_buttons:
        p, _ = TEAM_COLORS.get(team, ("#555555", "#FFFFFF"))
        sel = f".st-key-{key} button"
        if picked:
            rules.append(f"{sel}{{background:{p} !important;border:1px solid {p} !important;"
                         f"color:{on_color(p)} !important;box-shadow:inset 0 0 0 2px rgba(255,255,255,.35);}}")
        else:
            rules.append(f"{sel}{{border-left:6px solid {p} !important;}}"
                         f"{sel}:hover{{border-color:{p} !important;}}")
    return "<style>" + "".join(rules) + "</style>"


# ---------------------------------------------------------------- HTML pieces
def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def band(title: str, subtitle: str | None = None, kind: str = "") -> str:
    sub = f"<div class='s'>{subtitle}</div>" if subtitle else ""
    return f"<div class='scout-band {kind}'><div class='t'>{title}</div>{sub}</div>"


def team_name(team: str) -> str:
    """Bold team abbreviation in its color, with a separate shade for dark mode."""
    return f"<span class='tm' style='--tl:{team_ink(team)};--td:{team_ink_dark(team)}'>{esc(team)}</span>"


def team_cell(team: str, logo: str, extra: str = "") -> str:
    return (f"<img src='{logo}' alt=''>{team_name(team)}"
            f"{extra}")


def stack(value: str, rank: int | None = None, n: int | None = None, rank_color: str | None = None) -> str:
    color = f' style="color:{rank_color}"' if rank_color else ""
    r = (f"<span class='r'{color}>{rank}/{n}</span>"
         if rank is not None else "")
    return f"<span class='v'>{value}</span>{r}"


def td(content: str, fill: str | None = None, cls: str = "", sort=None) -> dict:
    """sort = the value used when the column is clicked (defaults to the cell's text)."""
    return {"html": content, "fill": fill, "cls": cls, "sort": sort}


_TAG = re.compile(r"<[^>]+>")

# Click a header to sort; click again to flip. Numbers sort numerically, text alphabetically.
SORT_JS = """<script>
(function () {
  document.querySelectorAll('table.scout.sortable:not([data-bound])').forEach(function (t) {
    t.setAttribute('data-bound', '1');
    var heads = t.querySelectorAll('thead th');
    heads.forEach(function (th, i) {
      th.addEventListener('click', function () {
        var asc = th.getAttribute('data-dir') !== 'asc';
        heads.forEach(function (h) { h.removeAttribute('data-dir'); });
        th.setAttribute('data-dir', asc ? 'asc' : 'desc');
        var body = t.tBodies[0], rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (a, b) {
          var x = a.cells[i].getAttribute('data-s'), y = b.cells[i].getAttribute('data-s');
          var nx = parseFloat(x), ny = parseFloat(y), c;
          if (!isNaN(nx) && !isNaN(ny)) c = nx - ny; else c = String(x).localeCompare(String(y));
          return asc ? c : -c;
        });
        rows.forEach(function (r) { body.appendChild(r); });
      });
    });
  });
})();
</script>"""


def table(columns: list[tuple[str, str]], rows: list[list[dict | str]], title: str | None = None,
          subtitle: str | None = None, footnote: str | None = None, max_height: int | None = None,
          left_cols: tuple[int, ...] = (), sortable: bool = False) -> str:
    """columns = [(label, hover help)], rows = lists of td() dicts or plain HTML strings."""
    head = "".join(
        f"<th class='{'l' if i in left_cols else ''}' title='{esc(h)}'>{lbl if lbl.startswith('<') else esc(lbl)}</th>"
        for i, (lbl, h) in enumerate(columns))
    body = []
    for row in rows:
        cells = []
        for i, c in enumerate(row):
            c = c if isinstance(c, dict) else td(c)
            cls = " ".join(x for x in (c["cls"], "l" if i in left_cols else "", "f" if c["fill"] else "") if x)
            style = f" style='background-color:{c['fill']}'" if c["fill"] else ""
            key = c["sort"] if c["sort"] is not None else _TAG.sub("", c["html"]).strip()
            cells.append(f"<td class='{cls}' data-s='{esc(key)}'{style}>{c['html']}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    scroll = f" style='max-height:{max_height}px'" if max_height else ""
    out = ["<div class='scout-wrap'>"]
    if title:
        out.append(band(title, subtitle))
    out.append(f"<div class='scout-scroll'{scroll}><table class='scout{' sortable' if sortable else ''}'><thead><tr>{head}</tr></thead>"
               f"<tbody>{''.join(body)}</tbody></table></div>")
    if footnote:
        out.append(f"<div class='scout-foot'>{footnote}</div>")
    out.append("</div>")
    if sortable:
        out.append(SORT_JS)
    return "".join(out)


# Streamlit picks light or dark (system setting, or the menu: Settings > Theme). This watches the
# app background and flags <html data-sc-theme="dark"> so the custom tables switch instantly too.
THEME_JS = r"""<script>
(function () {
  function apply() {
    var el = document.querySelector('.stApp') || document.body;
    var m = (getComputedStyle(el).backgroundColor || '').match(/\d+/g);
    if (!m) return;
    var lum = (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) / 255;
    var t = lum < 0.5 ? 'dark' : 'light';
    if (document.documentElement.getAttribute('data-sc-theme') !== t)
      document.documentElement.setAttribute('data-sc-theme', t);
  }
  apply();
  if (!window.__scThemeWatch) window.__scThemeWatch = setInterval(apply, 400);
})();
</script>"""
