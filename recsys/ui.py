"""Presentation helpers for the Streamlit app (pure python: HTML snippets, colours, CSS)."""
from __future__ import annotations

import html
from typing import Iterable, Sequence

GENRE_STYLE = {
    "Action": ("💥", "#ff5f6d", "#ffc371"), "Adventure": ("🧭", "#f7971e", "#ffd200"),
    "Animation": ("🎨", "#7f7fd5", "#91eae4"), "Children's": ("🧸", "#f9d423", "#ff4e50"),
    "Comedy": ("😂", "#f12711", "#f5af19"), "Crime": ("🕵️", "#434343", "#6b6b83"),
    "Documentary": ("🎙️", "#1d976c", "#93f9b9"), "Drama": ("🎭", "#614385", "#516395"),
    "Fantasy": ("🧙", "#8e2de2", "#4a00e0"), "Film-Noir": ("🎞️", "#232526", "#414345"),
    "Horror": ("👻", "#870000", "#190a05"), "Musical": ("🎵", "#ee0979", "#ff6a00"),
    "Mystery": ("🔍", "#3a1c71", "#d76d77"), "Romance": ("💘", "#ff758c", "#ff7eb3"),
    "Sci-Fi": ("🚀", "#2193b0", "#6dd5ed"), "Thriller": ("🔪", "#cb2d3e", "#ef473a"),
    "War": ("🎖️", "#56ab2f", "#a8e063"), "Western": ("🤠", "#c79081", "#dfa579"),
    "unknown": ("🎬", "#485563", "#29323c"),
}

PALETTE = ["#7C5CFF", "#FF5C8A", "#FFB347", "#2EC4B6", "#4EA8DE", "#B8C0D4"]


def inject_css() -> str:
    return """
<style>
.block-container {padding-top: 1.4rem; max-width: 1280px;}
.hero {padding: 26px 32px; border-radius: 22px; margin-bottom: 14px; color: #fff;
       background: linear-gradient(120deg, #6a4cff 0%, #c04cff 38%, #ff5c8a 72%, #ffb347 100%);
       box-shadow: 0 10px 34px rgba(124, 92, 255, .28);}
.hero h1 {margin: 0; font-size: 2.15rem; font-weight: 800; letter-spacing: -.5px;}
.hero p {margin: 6px 0 0; opacity: .93; font-size: 1.02rem;}
.pill {display: inline-block; padding: 3px 12px; margin: 10px 8px 0 0; border-radius: 999px; font-size: .78rem;
       background: rgba(255,255,255,.2); backdrop-filter: blur(4px);}
.warn {padding: 12px 16px; border-radius: 12px; background: #3b2a12; border: 1px solid #ffb347; color: #ffd9a0;
       margin-bottom: 12px; font-size: .9rem;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 8px 0 16px;}
.kpi {padding: 14px 16px; border-radius: 16px; background: #161B26; border: 1px solid #232B3B;}
.kpi .l {font-size: .74rem; letter-spacing: .6px; text-transform: uppercase; color: #8E9ABB;}
.kpi .v {font-size: 1.65rem; font-weight: 750; margin-top: 2px;}
.kpi .s {font-size: .76rem; color: #8E9ABB; margin-top: 2px;}
.grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 14px;}
.mcard {display: flex; gap: 14px; padding: 14px; border-radius: 18px; background: #161B26;
        border: 1px solid #232B3B; transition: transform .15s ease, border-color .15s ease;}
.mcard:hover {transform: translateY(-3px); border-color: #7C5CFF;}
.poster {width: 70px; min-width: 70px; height: 100px; border-radius: 14px; display: flex; align-items: center;
         justify-content: center; font-size: 32px; position: relative;}
.poster .rk {position: absolute; top: -7px; left: -7px; background: #0E1117; color: #fff; border-radius: 999px;
             width: 26px; height: 26px; font-size: .78rem; font-weight: 700; display: flex; align-items: center;
             justify-content: center; border: 2px solid #7C5CFF;}
.mbody {flex: 1; min-width: 0;}
.mtitle {font-weight: 700; font-size: 1.02rem; line-height: 1.25; color: #F2F4FA;}
.myear {color: #8E9ABB; font-weight: 400; font-size: .85rem;}
.chip {display: inline-block; padding: 2px 9px; margin: 5px 5px 0 0; border-radius: 999px; font-size: .7rem;
       background: #222B3D; color: #B8C0D4;}
.stars {color: #FFC857; letter-spacing: 1px; margin-top: 6px; font-size: .95rem;}
.stars b {color: #F2F4FA; letter-spacing: 0; margin-left: 6px;}
.why {font-size: .78rem; color: #8E9ABB; margin-top: 5px; line-height: 1.3;}
.mini {padding: 8px 12px; border-radius: 12px; background: #161B26; border: 1px solid #232B3B; margin-bottom: 8px;
       font-size: .84rem;}
.mini .t {font-weight: 650; color: #F2F4FA;}
.mini .m {color: #8E9ABB; font-size: .74rem;}
.section-title {font-size: 1.05rem; font-weight: 700; margin: 6px 0 8px;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {border-radius: 12px 12px 0 0; padding: 10px 16px;}
</style>
"""


def hero(title: str, subtitle: str, pills: Iterable[str]) -> str:
    pill_html = "".join(f'<span class="pill">{html.escape(p)}</span>' for p in pills)
    return (f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{pill_html}</div>')


def kpi_row(items: Sequence[tuple]) -> str:
    """items: (label, value, sub) tuples."""
    cells = "".join(
        f'<div class="kpi"><div class="l">{html.escape(str(l))}</div><div class="v">{html.escape(str(v))}</div>'
        f'<div class="s">{html.escape(str(s))}</div></div>' for l, v, s in items)
    return f'<div class="kpis">{cells}</div>'


def poster_style(genres: Sequence[str]) -> tuple:
    g = next((x for x in genres if x in GENRE_STYLE), "unknown")
    emoji, c1, c2 = GENRE_STYLE[g]
    return emoji, f"linear-gradient(145deg,{c1},{c2})"


def stars(pred: float) -> str:
    full = int(round(pred))
    return "★" * full + "☆" * (5 - full)


def movie_card(rank: int, title: str, year, genres: Sequence[str], pred: float, why: str = "") -> str:
    emoji, bg = poster_style(genres)
    chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in list(genres)[:4])
    yr = f' <span class="myear">({int(year)})</span>' if year == year and year else ""
    why_html = f'<div class="why">{html.escape(why)}</div>' if why else ""
    return (f'<div class="mcard"><div class="poster" style="background:{bg}">{emoji}<div class="rk">{rank}</div></div>'
            f'<div class="mbody"><div class="mtitle">{html.escape(str(title))}{yr}</div>{chips}'
            f'<div class="stars">{stars(pred)}<b>{pred:.2f}</b></div>{why_html}</div></div>')


def card_grid(cards: Iterable[str]) -> str:
    return f'<div class="grid">{"".join(cards)}</div>'


def mini_item(title: str, meta: str) -> str:
    return f'<div class="mini"><div class="t">{html.escape(str(title))}</div><div class="m">{html.escape(meta)}</div></div>'
