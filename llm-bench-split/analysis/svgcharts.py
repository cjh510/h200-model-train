# -*- coding: utf-8 -*-
"""极简 SVG 图表生成器（纯标准库，无外部依赖）。"""
from html import escape
import math

C = {
    "blue": "#2563eb", "cyan": "#0891b2", "purple": "#7c3aed", "orange": "#ea580c",
    "green": "#16a34a", "red": "#dc2626", "amber": "#d97706", "slate": "#64748b",
    "grid": "#e8edf3", "axis": "#94a3b8", "text": "#334155", "muted": "#94a3b8",
    "light": "#f1f5f9",
}
SERIES_COLORS = [C["blue"], C["orange"], C["green"], C["purple"], C["cyan"], C["red"]]


def _t(x, y, s, size=12, fill=None, anchor="start", weight="normal", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill or C["text"]}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{escape(str(s))}</text>')


def _line(x1, y1, x2, y2, stroke, width=1, dash="", opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{d}/>')


def _rect(x, y, w, h, fill, rx=3, opacity=1, stroke="none"):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0,w):.1f}" height="{max(0,h):.1f}" '
            f'rx="{rx}" fill="{fill}" opacity="{opacity}" stroke="{stroke}"/>')


def _svg(w, h, body, desc=""):
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'role="img" aria-label="{escape(desc)}">{body}</svg>')


def nice_max(v):
    if v <= 0:
        return 1
    e = math.floor(math.log10(v))
    base = 10 ** e
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * base >= v:
            return m * base
    return 10 * base


def hbar(rows, *, label_w=360, w=900, row_h=30, unit="", max_val=None, decimals=1,
         ref=None, ref_label="", desc="", note_key="note"):
    """rows: [{label, value, color, note?, tip?}]"""
    n = len(rows)
    top, bottom = 34, 54
    h = top + n * row_h + bottom
    x0, x1 = label_w, w - 70
    mx = max_val or nice_max(max([r["value"] for r in rows] + [0.001]))
    b = []
    # 网格与刻度
    for i in range(5):
        gx = x0 + (x1 - x0) * i / 4
        b.append(_line(gx, top - 8, gx, top + n * row_h, C["grid"], 1))
        b.append(_t(gx, top + n * row_h + 18, f"{mx*i/4:.0f}", 11, C["muted"], "middle"))
    if unit:
        b.append(_t((x0 + x1) / 2, top + n * row_h + 38, unit, 11, C["muted"], "middle"))
    for i, r in enumerate(rows):
        y = top + i * row_h
        cy = y + row_h / 2
        bw = (x1 - x0) * r["value"] / mx
        tip = r.get("tip") or f'{r["label"]}: {r["value"]:.{decimals}f} {unit}'
        b.append(f'<g><title>{escape(tip)}</title>')
        b.append(_t(x0 - 10, cy + 4, r["label"], 11.5, C["text"], "end"))
        b.append(_rect(x0, y + 5, bw, row_h - 11, r.get("color") or C["blue"], 2))
        b.append(_t(x0 + bw + 6, cy + 4, f'{r["value"]:.{decimals}f}', 11, C["text"], "start", "600"))
        if r.get(note_key):
            b.append(_t(x1 + 8, cy + 4, r[note_key], 10.5, C["muted"], "start"))
        b.append("</g>")
    if ref is not None and ref <= mx:
        rx = x0 + (x1 - x0) * ref / mx
        b.append(_line(rx, top - 10, rx, top + n * row_h + 2, C["red"], 1.5, "4 3"))
        on_right = rx > w * 0.62
        b.append(_t(rx - 5 if on_right else rx + 5, top - 14, ref_label, 10.5, C["red"],
                    "end" if on_right else "start"))
    return _svg(w, h, "".join(b), desc)


def vbar(cats, groups, *, w=900, h=330, unit="", max_val=None, decimals=0, desc="",
         ylabel="", legend=True, labels=None):
    """cats: 类目; groups: [{name,color,values}]"""
    left, right, top, bottom = 62, 24, 42 if legend else 20, 58
    pw, ph = w - left - right, h - top - bottom
    mx = max_val or nice_max(max([v for g in groups for v in g["values"]] + [0.001]))
    b = []
    for i in range(5):
        gy = top + ph - ph * i / 4
        b.append(_line(left, gy, w - right, gy, C["grid"], 1))
        b.append(_t(left - 8, gy + 4, f"{mx*i/4:.0f}", 10.5, C["muted"], "end"))
    b.append(_t(left - 8, top - 10, ylabel or unit, 10.5, C["muted"], "end"))
    slot = pw / len(cats)
    ng = len(groups)
    bw = min(46, slot * 0.72 / ng)
    for ci, cat in enumerate(cats):
        cx = left + slot * (ci + 0.5)
        for gi, g in enumerate(groups):
            val = g["values"][ci]
            bh = ph * val / mx
            bx = cx - (ng * bw) / 2 + gi * bw + 1
            b.append(f'<g><title>{escape(cat)} · {escape(g["name"])}: {val:.{decimals}f} {unit}</title>')
            b.append(_rect(bx, top + ph - bh, bw - 2, bh, g["color"], 2))
            b.append(_t(bx + (bw - 2) / 2, top + ph - bh - 5, f"{val:.{decimals}f}", 10,
                        C["text"], "middle", "600"))
            b.append("</g>")
        lab = labels[ci] if labels else str(cat)
        b.append(_t(cx, top + ph + 20, lab, 11, C["text"], "middle"))
    b.append(_line(left, top + ph, w - right, top + ph, C["axis"], 1))
    if legend:
        lx = left
        for g in groups:
            b.append(_rect(lx, 12, 12, 10, g["color"], 2))
            b.append(_t(lx + 17, 21, g["name"], 11, C["text"]))
            lx += 22 + len(g["name"]) * 7.2
    return _svg(w, h, "".join(b), desc)


def _scale(v, lo, hi, px0, px1, log=False):
    """把数值 v 映射到 [px0, px1] 像素区间（px0 对应 lo，px1 对应 hi）。"""
    if log:
        lv, llo, lhi = math.log10(max(v, 1e-9)), math.log10(max(lo, 1e-9)), math.log10(max(hi, 1e-9))
        return px0 + (lv - llo) / (lhi - llo) * (px1 - px0) if lhi > llo else px0
    return px0 + (v - lo) / (hi - lo) * (px1 - px0) if hi > lo else px0


def lines(xs, series, *, w=900, h=330, xlog=False, ylog=False, y2log=False,
          unit="", unit2="", xlabels=None, desc="", legend=True, ymin=None, ymax=None,
          y2min=None, y2max=None, xticks=6, xlabel="", x_tick_values=None):
    """series: [{name,color,values,axis('left'|'right'),dash?,band?}]  band: [(lo,hi),...]"""
    left, right, top, bottom = 62, 62 if any(s.get("axis") == "right" for s in series) else 24, 40, 58
    pw, ph = w - left - right, h - top - bottom
    xlo, xhi = min(xs), max(xs)
    lefts = [v for s in series if s.get("axis", "left") == "left" for v in s["values"] if v is not None]
    rights = [v for s in series if s.get("axis") == "right" for v in s["values"] if v is not None]
    ylo = ymin if ymin is not None else (0 if not ylog else min(lefts) * 0.8)
    yhi = ymax if ymax is not None else nice_max(max(lefts or [1]) * 1.05)
    y2lo = y2min if y2min is not None else (0 if not y2log else min(rights or [1]) * 0.8)
    y2hi = y2max if y2max is not None else nice_max(max(rights or [1]) * 1.05)
    b = []
    def _fmt_tick(v):
        if v >= 1e6:
            return f"{v/1e6:.1f}M"
        if v >= 10000:
            return f"{v/1000:.1f}k"
        if v >= 10:
            return f"{v:,.0f}"
        return f"{v:g}"

    for i in range(5):
        gy = top + ph - ph * i / 4
        b.append(_line(left, gy, w - right, gy, C["grid"], 1))
        if ylog:
            v = 10 ** (math.log10(max(ylo, 1e-9)) + (math.log10(max(yhi, 1e-8)) - math.log10(max(ylo, 1e-9))) * i / 4)
        else:
            v = ylo + (yhi - ylo) * i / 4
        b.append(_t(left - 8, gy + 4, _fmt_tick(v), 10.5, C["muted"], "end"))
        if rights:
            if y2log:
                v2 = 10 ** (math.log10(max(y2lo, 1e-9)) + (math.log10(max(y2hi, 1e-8)) - math.log10(max(y2lo, 1e-9))) * i / 4)
            else:
                v2 = y2lo + (y2hi - y2lo) * i / 4
            b.append(_t(w - right + 8, gy + 4, _fmt_tick(v2), 10.5, C["muted"], "start"))
    if rights:
        b.append(_t(w - right + 8, top - 12, unit2, 10.5, C["muted"], "start"))
    # 单位标注：短单位右对齐到 y 轴；长单位（如 TFLOP/s/GPU）若仍右对齐会越出画布左侧被裁掉，
    # 改为从画布左边缘起排，保证完整可见。
    if unit:
        if len(unit) > 7:
            b.append(_t(4, top - 12, unit, 10.5, C["muted"], "start"))
        else:
            b.append(_t(left - 8, top - 12, unit, 10.5, C["muted"], "end"))
    # x 刻度
    if x_tick_values:
        ticks = x_tick_values
    elif xlog and xlo > 0:
        ticks = [10 ** (math.log10(xlo) + (math.log10(xhi) - math.log10(xlo)) * i / xticks)
                 for i in range(xticks + 1)]
    else:
        ticks = [xlo + (xhi - xlo) * i / xticks for i in range(xticks + 1)]
    for i, xv in enumerate(ticks):
        gx = _scale(xv, xlo, xhi, left, w - right, xlog)
        b.append(_line(gx, top, gx, top + ph, C["grid"], 1, opacity=0.6))
        if xlabels and i < len(xlabels):
            lab = xlabels[i]
        elif xlog and xv >= 1e9:
            lab = f"{xv/1e9:g}G"
        elif xlog and xv >= 1e6:
            lab = f"{xv/1e6:g}M"
        elif xlog and xv >= 1e3:
            lab = f"{xv/1e3:g}K"
        else:
            lab = f"{xv:,.0f}" if xv >= 1000 else f"{xv:g}"
        b.append(_t(gx, top + ph + 20, lab, 10.5, C["muted"], "middle"))
    if xlabel:
        b.append(_t((left + w - right) / 2, h - 8, xlabel, 11, C["muted"], "middle"))
    for s in series:
        col = s["color"]
        is_right = s.get("axis") == "right"
        lo, hi, lg = (y2lo, y2hi, y2log) if is_right else (ylo, yhi, ylog)
        pts = []
        for x, v in zip(xs, s["values"]):
            if v is None:
                continue
            pts.append((_scale(x, xlo, xhi, left, w - right, xlog), _scale(v, lo, hi, top + ph, top, lg)))
        if s.get("band") and len(pts) > 1:
            band = s["band"]
            up = [(p[0], _scale(hi_v, lo, hi, top + ph, top, lg)) for p, (_, hi_v) in zip(pts, band) if hi_v is not None]
            dn = [(p[0], _scale(lo_v, lo, hi, top + ph, top, lg)) for p, (lo_v, _) in zip(pts, band) if lo_v is not None]
            if len(up) > 1 and len(dn) > 1:
                poly = " ".join(f"{a},{b:.1f}" for a, b in up + dn[::-1])
                b.append(f'<polygon points="{poly}" fill="{col}" opacity="0.10"/>')
        if len(pts) > 1:
            dash_attr = ' stroke-dasharray="5 3"' if s.get("dash") else ""
            b.append('<polyline points="' + " ".join(f"{a:.1f},{c:.1f}" for a, c in pts) +
                     f'" fill="none" stroke="{col}" stroke-width="2.2" stroke-linejoin="round"{dash_attr}/>')
        for (px, py), x, v in zip(pts, xs, s["values"]):
            if v is None:
                continue
            b.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{col}"><title>{escape(s["name"])}: x={x:g}, {v:.2f}</title></circle>')
    b.append(_line(left, top + ph, w - right, top + ph, C["axis"], 1))
    if legend:
        lx = left
        for s in series:
            b.append(_line(lx, 16, lx + 16, 16, s["color"], 2.5, s.get("dash", "")))
            b.append(_t(lx + 21, 20, s["name"], 11, C["text"]))
            lx += 32 + len(s["name"]) * 7.4
    return _svg(w, h, "".join(b), desc)


def scatter_status(points, *, w=900, h=250, desc="", xlabel="每卡 tokens", xmax=None):
    """points: [{x,y,label,status,color}] 用气泡位置展示 OK / OOM 边界"""
    left, right, top, bottom = 60, 30, 34, 48
    pw, ph = w - left - right, h - top - bottom
    mx = xmax or nice_max(max(p["x"] for p in points) * 1.05)
    b = []
    for i in range(5):
        gx = left + pw * i / 4
        b.append(_line(gx, top - 6, gx, top + ph, C["grid"], 1))
        b.append(_t(gx, top + ph + 20, f"{mx*i/4:,.0f}", 10.5, C["muted"], "middle"))
    b.append(_t(left + pw / 2, h - 8, xlabel, 11, C["muted"], "middle"))
    lanes = {}
    for p in points:
        lanes.setdefault(p.get("y", 0), []).append(p)
    for lane, ps in sorted(lanes.items()):
        y = top + 30 + lane * 52
        b.append(_t(left - 8, y + 5, f"{ps[0].get('lane_label','')}", 11, C["muted"], "end"))
        for p in ps:
            x = left + pw * p["x"] / mx
            col = p["color"]
            b.append(f'<g><title>{escape(p["label"])}: {p["x"]:,} tokens/GPU · {p["status"]}</title>')
            b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="{col}" opacity="0.85"/>')
            b.append(_t(x, y + 4, "OK" if p["status"] == "OK" else "OOM", 9.5, "#fff", "middle", "700"))
            b.append(_t(x, y + 30, p["label"], 10, C["text"], "middle"))
            b.append("</g>")
    # 边界带
    ok = [p["x"] for p in points if p["status"] == "OK"]
    oom = [p["x"] for p in points if p["status"] != "OK"]
    if ok and oom:
        xa, xb = max(ok), min(oom)
        gx1, gx2 = left + pw * xa / mx, left + pw * xb / mx
        b.append(_rect(gx1, top - 6, gx2 - gx1, ph + 6, C["amber"], 0, 0.10))
        b.append(_t((gx1 + gx2) / 2, top + ph + 38, "★ 边界区间", 10.5, C["amber"], "middle", "600"))
    return _svg(w, h, "".join(b), desc)


def mini_progress(ok, warn, fail, skip, w=900, h=34, labels=None):
    total = max(1, ok + warn + fail + skip)
    b = []
    x = 0
    labs = labels or ["PASS", "边界/部分", "FAIL", "SKIP/缺失"]
    for n, col, lab in [(ok, C["green"], labs[0]), (warn, C["amber"], labs[1]),
                        (fail, C["red"], labs[2]), (skip, C["muted"], labs[3])]:
        if n <= 0:
            continue
        bw = w * n / total
        b.append(f'<g><title>{lab}: {n}/{total}</title>')
        b.append(_rect(x, 4, bw - 2, h - 8, col, 3))
        if bw > 42:
            b.append(_t(x + bw / 2 - 1, h / 2 + 4, f"{lab} {n}", 11, "#fff", "middle", "600"))
        b.append("</g>")
        x += bw
    return _svg(w, h, "".join(b), "阶段状态分布")
