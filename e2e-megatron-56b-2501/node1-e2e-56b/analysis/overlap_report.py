# -*- coding: utf-8 -*-
"""基于 SVG text 包围盒估算，检测图表内的文字重叠与画布溢出"""
import json
import sys
from collections import defaultdict
from pathlib import Path

T = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else '_svg_texts.json').read_text(encoding='utf-8'))

def width(s, fs):
    w = 0.0
    for ch in s:
        w += fs * (1.0 if ord(ch) > 127 else 0.56)
    return w

def bbox(t):
    w = width(t['t'], t['fs'])
    x = t['x']
    x0 = x - w if t['a'] == 'end' else (x - w / 2 if t['a'] == 'middle' else x)
    return (x0, t['y'] - t['fs'] * .85, x0 + w, t['y'] + t['fs'] * .2)

by_svg = defaultdict(list)
for t in T:
    by_svg[t['svg']].append(t)

overlaps = []
overflow = []
for sid, items in by_svg.items():
    W, H = items[0]['W'], items[0]['H']
    boxes = [(t, bbox(t)) for t in items if t['t'].strip()]
    for t, b in boxes:
        if b[0] < -1 or b[2] > W + 1 or b[1] < -1 or b[3] > H + 1:
            overflow.append((sid, t['host'], t['t'], [round(v, 1) for v in b], W, H))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, ba = boxes[i]; c, bc = boxes[j]
            ix = max(0, min(ba[2], bc[2]) - max(ba[0], bc[0]))
            iy = max(0, min(ba[3], bc[3]) - max(ba[1], bc[1]))
            if ix * iy > 12:
                overlaps.append((sid, a['host'], round(ix * iy), a['t'], c['t'],
                                 [round(v) for v in ba], [round(v) for v in bc]))

print(f"检测 {len(by_svg)} 个 SVG / {len(T)} 个 text 元素")
print(f"\n=== 文字重叠 {len(overlaps)} 处 ===")
for o in sorted(overlaps, key=lambda x: -x[2])[:60]:
    print(f"  svg#{o[0]} {o[1]} 面积={o[2]}: '{o[3]}' × '{o[4]}'  bbox={o[5]} / {o[6]}")
print(f"\n=== 画布溢出 {len(overflow)} 处 ===")
for o in overflow[:60]:
    print(f"  svg#{o[0]} {o[1]}: '{o[2]}' bbox={o[3]} 画布 {o[4]}x{o[5]}")
