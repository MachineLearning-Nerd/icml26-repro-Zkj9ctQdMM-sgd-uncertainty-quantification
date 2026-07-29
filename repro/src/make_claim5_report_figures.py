"""Generate deterministic SVG figures for the Claim 5 report.

The script uses only the Python standard library and the committed raw evidence.
It deliberately fails when the expected full-run structure is absent.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "outputs" / "c5_falsification.json"
OUT = ROOT / "reports" / "claim5-falsification" / "images"

BG = "#fbfcfe"
INK = "#172033"
MUTED = "#566176"
GRID = "#d9deea"
BLUE = "#3166c9"
GREEN = "#16846c"
RED = "#c84242"
ORANGE = "#c77719"
PURPLE = "#7257b8"


def esc(value: object) -> str:
    return html.escape(str(value))


def svg_open(title: str, width: int = 1200, height: int = 650) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id='title'>{esc(title)}</title>",
        f"<desc id='desc'>{esc(title)} generated from outputs/c5_falsification.json</desc>",
        f"<rect width='{width}' height='{height}' fill='{BG}'/>",
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}"
        ".title{font-size:28px;font-weight:700}.sub{font-size:17px;fill:#566176}"
        ".label{font-size:16px}.small{font-size:14px;fill:#566176}"
        ".value{font-size:15px;font-weight:700}</style>",
    ]


def text(x: float, y: float, value: object, cls: str = "label", anchor: str = "start",
         fill: str | None = None) -> str:
    color = f" fill='{fill}'" if fill else ""
    return f"<text x='{x:.1f}' y='{y:.1f}' class='{cls}' text-anchor='{anchor}'{color}>{esc(value)}</text>"


def save(name: str, lines: list[str]) -> None:
    lines.append("</svg>")
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_y(value: float, top: float, bottom: float, low: float = -2.0, high: float = 9.0) -> float:
    z = max(low, min(high, math.log10(value)))
    return bottom - (z - low) / (high - low) * (bottom - top)


def headline(data: dict) -> None:
    methods = ["DQ+exact", "CT", "LR+WS"]
    colors = {"DQ+exact": GREEN, "CT": BLUE, "LR+WS": RED}
    paper = data["paper_table3_log"]
    replay = data["primary"]["cells"]
    lines = svg_open("Paper Table 3 and independent full-Boston replay")
    lines += [
        text(55, 52, "The claimed CT divergence is absent in both evidence routes", "title"),
        text(55, 82, "Relative covariance error on a log scale; lower is better. LR+WS alone explodes.", "sub"),
    ]
    panels = [("Paper Table 3", 80, lambda b, m: paper[str(b)][m]),
              ("Independent replay (simulation median)", 635,
               lambda b, m: replay[str(b)][m]["sim"]["median"])]
    for heading, x0, getter in panels:
        top, bottom = 145, 555
        lines.append(text(x0 + 225, 122, heading, "label", "middle"))
        for tick in [-2, 0, 2, 4, 6, 8]:
            yy = log_y(10 ** tick, top, bottom)
            lines.append(f"<line x1='{x0}' y1='{yy:.1f}' x2='{x0+460}' y2='{yy:.1f}' stroke='{GRID}'/>")
            lines.append(text(x0 - 10, yy + 5, f"10^{tick}", "small", "end"))
        for bi, b in enumerate((16, 50)):
            gx = x0 + 95 + bi * 220
            lines.append(text(gx + 45, 585, f"B={b}", "label", "middle"))
            for mi, method in enumerate(methods):
                value = float(getter(b, method))
                capped = min(value, 1e9)
                xx = gx + (mi - 1) * 52
                yy = log_y(capped, top, bottom)
                base = log_y(1e-2, top, bottom)
                lines.append(f"<rect x='{xx-17:.1f}' y='{yy:.1f}' width='34' height='{base-yy:.1f}' "
                             f"rx='4' fill='{colors[method]}' opacity='0.88'/>")
                label = "∞" if value > 1e300 else (f"{value:.2g}" if value >= 100 else f"{value:.3f}")
                lines.append(text(xx, max(yy - 8, 137), label, "value", "middle"))
        for mi, method in enumerate(methods):
            lx = x0 + 70 + mi * 135
            lines.append(f"<rect x='{lx}' y='615' width='16' height='16' rx='3' fill='{colors[method]}'/>")
            lines.append(text(lx + 23, 629, method, "small"))
    save("headline.svg", lines)


def stability(data: dict) -> None:
    primary = data["primary"]["cells"]
    ws = data["controls"]["wellspec_gauss"]["cells"]
    rows = [
        ("DQ+exact", primary["16"]["DQ+exact"]["rho"], primary["50"]["DQ+exact"]["rho"]),
        ("CT", primary["16"]["CT"]["rho"], primary["50"]["CT"]["rho"]),
        ("LR+WS", primary["16"]["LR+WS"]["rho"], primary["50"]["LR+WS"]["rho"]),
        ("DQ+const", primary["16"]["DQ+const"]["rho"], primary["50"]["DQ+const"]["rho"]),
        ("CT ×50 detector", primary["16"]["CTx50-control"]["rho"], primary["50"]["CTx50-control"]["rho"]),
        ("LR+WS well-specified", ws["16"]["LR+WS"]["rho"], ws["50"]["LR+WS"]["rho"]),
    ]
    lines = svg_open("Exact second-moment stability boundary and controls")
    lines += [
        text(55, 52, "Exact stability test separates finite chains from divergent ones", "title"),
        text(55, 82, "ρ(T) < 1 is stable. The detector crosses the boundary; the well-specified control returns below it.", "sub"),
    ]
    left, right, top, row_h = 330, 1120, 135, 75
    for tick in [-0.2, 0, 0.5, 1, 2]:
        value = 10 ** tick
        xx = left + (tick + 0.3) / 2.6 * (right - left)
        lines.append(f"<line x1='{xx:.1f}' y1='{top-20}' x2='{xx:.1f}' y2='{top+row_h*len(rows)-20}' stroke='{GRID}'/>")
        lines.append(text(xx, 600, f"{value:.1f}", "small", "middle"))
    boundary = left + (0 + 0.3) / 2.6 * (right - left)
    lines.append(f"<line x1='{boundary:.1f}' y1='105' x2='{boundary:.1f}' y2='565' stroke='{RED}' stroke-width='3'/>")
    lines.append(text(boundary + 8, 118, "ρ=1 boundary", "small", "start", RED))
    for i, (label, r16, r50) in enumerate(rows):
        yy = top + i * row_h
        lines.append(text(55, yy + 6, label, "label"))
        lines.append(f"<line x1='{left}' y1='{yy}' x2='{right}' y2='{yy}' stroke='{GRID}'/>")
        for value, color, offset, batch in [(r16, BLUE, -9, 16), (r50, PURPLE, 9, 50)]:
            xx = left + (math.log10(value) + 0.3) / 2.6 * (right - left)
            xx = min(right, max(left, xx))
            lines.append(f"<circle cx='{xx:.1f}' cy='{yy+offset}' r='8' fill='{color}'/>")
            lines.append(text(min(xx + 13, 1150), yy + offset + 5, f"{value:.3g}", "small"))
    lines += [
        f"<circle cx='410' cy='623' r='7' fill='{BLUE}'/>", text(425, 628, "B=16", "small"),
        f"<circle cx='505' cy='623' r='7' fill='{PURPLE}'/>", text(520, 628, "B=50", "small"),
    ]
    save("stability-controls.svg", lines)


def replicates(data: dict) -> None:
    cells = data["primary"]["cells"]
    lines = svg_open("Thirty seeded chain replicates")
    lines += [
        text(55, 52, "Thirty seeded chains agree with the exact stability calculation", "title"),
        text(55, 82, "Finite-method errors use a linear axis; LR+WS uses a separate log axis because it is four to eight orders larger.", "sub"),
    ]
    # finite panel
    left, right, top, bottom = 90, 720, 135, 555
    finite_max = 0.18
    for tick in [0, .045, .09, .135, .18]:
        yy = bottom - tick / finite_max * (bottom - top)
        lines.append(f"<line x1='{left}' y1='{yy:.1f}' x2='{right}' y2='{yy:.1f}' stroke='{GRID}'/>")
        lines.append(text(left - 10, yy + 5, f"{tick:.3f}".rstrip("0").rstrip("."), "small", "end"))
    groups = [("DQ B16", cells["16"]["DQ+exact"], GREEN),
              ("CT B16", cells["16"]["CT"], BLUE),
              ("DQ B50", cells["50"]["DQ+exact"], GREEN),
              ("CT B50", cells["50"]["CT"], BLUE)]
    for gi, (label, cell, color) in enumerate(groups):
        xx = left + 90 + gi * 145
        vals = cell["sim"]["per_rep"]
        for j, value in enumerate(vals):
            jitter = ((j * 17) % 23 - 11) * 0.7
            yy = bottom - min(value, finite_max) / finite_max * (bottom - top)
            lines.append(f"<circle cx='{xx+jitter:.1f}' cy='{yy:.1f}' r='4' fill='{color}' opacity='0.55'/>")
        med = cell["sim"]["median"]
        yy = bottom - med / finite_max * (bottom - top)
        lines.append(f"<line x1='{xx-28}' y1='{yy:.1f}' x2='{xx+28}' y2='{yy:.1f}' stroke='{INK}' stroke-width='4'/>")
        lines.append(text(xx, 585, label, "small", "middle"))
    # LR+WS panel
    l2, r2 = 815, 1130
    lines.append(text((l2+r2)/2, 122, "LR+WS errors (log₁₀)", "label", "middle"))
    for tick in [3, 4, 5, 6, 7]:
        yy = bottom - (tick - 3) / 4 * (bottom - top)
        lines.append(f"<line x1='{l2}' y1='{yy:.1f}' x2='{r2}' y2='{yy:.1f}' stroke='{GRID}'/>")
        lines.append(text(l2 - 10, yy + 5, f"10^{tick}", "small", "end"))
    for gi, b in enumerate(("16", "50")):
        xx = l2 + 95 + gi * 145
        vals = cells[b]["LR+WS"]["sim"]["per_rep"]
        for j, value in enumerate(vals):
            jitter = ((j * 19) % 23 - 11) * 0.7
            yy = bottom - (max(3, min(7, math.log10(value))) - 3) / 4 * (bottom - top)
            lines.append(f"<circle cx='{xx+jitter:.1f}' cy='{yy:.1f}' r='4' fill='{RED}' opacity='0.55'/>")
        med = cells[b]["LR+WS"]["sim"]["median"]
        yy = bottom - (math.log10(med) - 3) / 4 * (bottom - top)
        lines.append(f"<line x1='{xx-28}' y1='{yy:.1f}' x2='{xx+28}' y2='{yy:.1f}' stroke='{INK}' stroke-width='4'/>")
        lines.append(text(xx, 585, f"LR+WS B{b}", "small", "middle"))
    lines.append(text(390, 625, "Dots = individual runs; black bar = median", "small", "middle"))
    save("replicate-errors.svg", lines)


def sensitivity(data: dict) -> None:
    rows = [
        ("Raw y + intercept", data["variants"][0]),
        ("Centered y", data["variants"][1]),
        ("Standardized y", data["variants"][2]),
        ("Without replacement", data["controls"]["without_replacement"]),
        ("Unscaled target (rejected)", data["variants"][3]),
    ]
    lines = svg_open("CT sensitivity across protocol choices")
    lines += [
        text(55, 52, "CT stays finite across every admissible posterior-scale protocol", "title"),
        text(55, 82, "Only the unscaled target—rejected because it also destabilizes DQ+exact—changes the conclusion.", "sub"),
    ]
    x0, y0, cw, ch = 500, 135, 275, 82
    lines.append(text(x0 + cw/2, 120, "B=16", "label", "middle"))
    lines.append(text(x0 + 1.5*cw, 120, "B=50", "label", "middle"))
    for i, (label, variant) in enumerate(rows):
        yy = y0 + i * ch
        lines.append(text(55, yy + 47, label, "label"))
        for j, b in enumerate(("16", "50")):
            rho = variant["cells"][b]["CT"]["rho"]
            stable = rho < 1
            color = GREEN if stable else RED
            xx = x0 + j * cw
            lines.append(f"<rect x='{xx+8}' y='{yy+8}' width='{cw-16}' height='{ch-16}' rx='9' fill='{color}' opacity='0.14'/>")
            lines.append(f"<circle cx='{xx+55}' cy='{yy+41}' r='11' fill='{color}'/>")
            label_value = f"ρ={rho:.3f}" if rho < 100 else f"ρ={rho:.0f}"
            lines.append(text(xx + 82, yy + 47, label_value + (" stable" if stable else " diverges"), "value"))
    lines.append(text(775, 600, "Green = stable (ρ<1) · red = divergent (ρ≥1)", "small", "middle"))
    save("protocol-sensitivity.svg", lines)


def mechanism(data: dict) -> None:
    boston_var = data["primary"]["resid_var"]
    well_var = data["controls"]["wellspec_gauss"]["resid_var"]
    boston_rho = data["primary"]["cells"]["16"]["LR+WS"]["rho"]
    well_rho = data["controls"]["wellspec_gauss"]["cells"]["16"]["LR+WS"]["rho"]
    lines = svg_open("Misspecification mechanism and well-specified control")
    lines += [
        text(55, 52, "The well-specified control reverses the LR+WS instability", "title"),
        text(55, 82, "The only changed ingredient is whether the unit-noise model matches the data-generating residual scale.", "sub"),
    ]
    max_var = 24
    for i, (label, variance, rho, color) in enumerate([
        ("Boston housing", boston_var, boston_rho, RED),
        ("Gaussian control", well_var, well_rho, GREEN),
    ]):
        yy = 175 + i * 205
        lines.append(text(55, yy, label, "label"))
        lines.append(text(55, yy + 32, f"residual variance = {variance:.3f}", "value"))
        lines.append(f"<rect x='310' y='{yy-25}' width='520' height='46' rx='8' fill='{GRID}'/>")
        width = variance / max_var * 520
        lines.append(f"<rect x='310' y='{yy-25}' width='{width:.1f}' height='46' rx='8' fill='{color}' opacity='0.82'/>")
        unit_x = 310 + 1 / max_var * 520
        lines.append(f"<line x1='{unit_x:.1f}' y1='{yy-40}' x2='{unit_x:.1f}' y2='{yy+35}' stroke='{INK}' stroke-width='2'/>")
        lines.append(text(unit_x, yy - 50, "model σ²=1", "small", "middle"))
        lines.append(f"<path d='M850 {yy-2} L930 {yy-2}' stroke='{color}' stroke-width='4' marker-end='url(#a{i})'/>")
        lines.append(f"<defs><marker id='a{i}' markerWidth='10' markerHeight='10' refX='8' refY='3' orient='auto'>"
                     f"<path d='M0,0 L0,6 L9,3 z' fill='{color}'/></marker></defs>")
        lines.append(text(950, yy - 10, f"LR+WS ρ={rho:.3f}", "value"))
        lines.append(text(950, yy + 22, "DIVERGES" if rho >= 1 else "stable", "label", "start", color))
    lines += [
        text(55, 590, "Positive control", "small"),
        text(180, 590, "Boston misspecification reproduces LR+WS divergence.", "label"),
        text(55, 622, "Negative control", "small"),
        text(180, 622, "When σ²≈1 is correct, the same implementation becomes stable and accurate.", "label"),
    ]
    save("misspecification-control.svg", lines)


def main() -> None:
    data = json.loads(RAW.read_text(encoding="utf-8"))
    if data.get("fast") or data.get("reps") != 30 or data.get("primary_tag") != "icpt-post":
        raise SystemExit("full-run Claim 5 evidence is required")
    OUT.mkdir(parents=True, exist_ok=True)
    headline(data)
    stability(data)
    replicates(data)
    sensitivity(data)
    mechanism(data)
    print(f"wrote 5 SVG figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
