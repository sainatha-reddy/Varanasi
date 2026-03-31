"""
futuristic_walls.py  (curved triple-screen edition)
=====================================================
Transforms a classroom photo into a futuristic sci-fi room by overlaying
THREE symmetrically-arranged screens in a curved triple-monitor arc:

        ┌──────────┐   ┌──────────────────┐   ┌──────────┐
        │  LEFT    │   │    CENTRE (main) │   │  RIGHT   │
        │  screen  │   │                  │   │  screen  │
        │ angled ◄ │   │   straight-on    │   │ ► angled │
        └──────────┘   └──────────────────┘   └──────────┘

Each screen is perspective-warped (warpPerspective) onto the front-wall area
so they appear to curve away from the viewer like a real curved ultra-wide
or cockpit display.  Each screen carries a dense, unique JARVIS-style dashboard.

Post-processing (colour grade, bloom, scanlines, chromatic aberration, vignette)
and HUD decorations are kept identical to the original pipeline.

NO GANs, NO neural networks — pure OpenCV + NumPy + PIL only.

Usage:
    python futuristic_walls.py --input classroom.png --output result.jpg

Dependencies:
    pip install opencv-python numpy Pillow
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import argparse
import math

# ─────────────────────────────────────────────
# SECTION 1 – EDGE & LINE DETECTION  (unchanged)
# ─────────────────────────────────────────────

def detect_edges_and_lines(img_bgr, save_debug=False):
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 40, 120, apertureSize=3)
    lines_raw = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=60, minLineLength=80, maxLineGap=15)
    lines = lines_raw[:, 0, :] if lines_raw is not None else np.empty((0, 4), dtype=int)
    debug_img = None
    if save_debug:
        debug_img = img_bgr.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite("debug_edges.jpg", edges)
        cv2.imwrite("debug_lines.jpg", debug_img)
    return edges, lines, debug_img


# ─────────────────────────────────────────────
# SECTION 2 – VANISHING POINT  (unchanged)
# ─────────────────────────────────────────────

def line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    return (x1 + t*(x2-x1), y1 + t*(y2-y1))


def estimate_vanishing_point(lines, img_w, img_h):
    diagonal = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        angle  = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
        length = math.hypot(x2-x1, y2-y1)
        if 5 < angle < 40 and length > 120:
            diagonal.append((x1, y1, x2, y2))
    intersections = []
    for i in range(len(diagonal)):
        for j in range(i+1, len(diagonal)):
            pt = line_intersection(diagonal[i], diagonal[j])
            if pt and -img_w < pt[0] < 2*img_w and -img_h < pt[1] < 2*img_h:
                intersections.append(pt)
    if len(intersections) < 3:
        print("[WARN] Too few intersections; using image centre as vanishing point.")
        return (img_w//2, img_h//2)
    vp = (int(np.median([p[0] for p in intersections])),
          int(np.median([p[1] for p in intersections])))
    print(f"[INFO] Estimated vanishing point: {vp}")
    return vp


# ─────────────────────────────────────────────
# SECTION 3 – CURVED TRIPLE-SCREEN QUADS
# ─────────────────────────────────────────────

def get_triple_screen_quads(img_w, img_h, vp):
    """
    Returns three perspective-correct screen quads forming a curved
    FORWARD-bending triple-monitor arc (like a cockpit or JARVIS display).

    The side screens tilt TOWARD the viewer:
      - Their OUTER edges are pushed further apart horizontally (foreshortened)
      - Their INNER edges (the joint with the centre screen) are pulled inward
        (recede), so the panel appears to wrap forward around the viewer.

    Visually this means the left screen leans left-forward and the right
    screen leans right-forward — the opposite of a flat wall.

    Corner order for every quad: TL, TR, BR, BL
    """
    W, H = img_w, img_h

    # ── Centre screen  ────────────────────────────────────────────────
    mid_top = int(H * 0.13)
    mid_bot = int(H * 0.81)
    cx_l    = int(W * 0.355)
    cx_r    = int(W * 0.645)
    taper   = int(H * 0.010)          # slight downward-view taper

    centre = np.array([
        [cx_l - taper,   mid_top],    # TL
        [cx_r + taper,   mid_top],    # TR
        [cx_r + taper*2, mid_bot],    # BR
        [cx_l - taper*2, mid_bot],    # BL
    ], dtype=np.float32)

    # ── Forward-bend geometry ─────────────────────────────────────────
    #
    #  The centre screen inner corners act as the "hinge".
    #  The outer edge of each side screen is pushed OUTWARD (wider) AND
    #  the whole side panel is compressed vertically at the inner edge
    #  (inner top raised, inner bottom lowered) so the screen looks like
    #  it's coming toward the viewer at its outer side.
    #
    #  Inner edge  = aligns flush with centre screen sides
    #  Outer edge  = stretched outward AND brought forward (larger, taller)

    forward_x  = int(W * 0.055)   # how much outer edge expands beyond the wall
    forward_v  = int(H * 0.065)   # how much taller the outer edge is vs inner

    # Inner corners: flush with centre screen edges, slight inward squeeze
    inner_shrink = int(H * 0.030)  # inner top pushed down, inner bottom pushed up
    l_in_top_x  = cx_l - taper
    l_in_bot_x  = cx_l - taper*2
    l_in_top_y  = mid_top + inner_shrink
    l_in_bot_y  = mid_bot - inner_shrink

    l_out_top_x = int(W * 0.08)  - forward_x
    l_out_bot_x = int(W * 0.08)  - forward_x
    l_out_top_y = mid_top - forward_v
    l_out_bot_y = mid_bot + forward_v

    left = np.array([
        [l_out_top_x, l_out_top_y],   # TL outer (far left, raised)
        [l_in_top_x,  l_in_top_y ],   # TR inner (joint with centre, lowered)
        [l_in_bot_x,  l_in_bot_y ],   # BR inner (joint with centre, raised)
        [l_out_bot_x, l_out_bot_y],   # BL outer (far left, dropped)
    ], dtype=np.float32)

    # Right screen: mirror
    r_in_top_x  = cx_r + taper
    r_in_bot_x  = cx_r + taper*2
    r_in_top_y  = mid_top + inner_shrink
    r_in_bot_y  = mid_bot - inner_shrink

    r_out_top_x = int(W * 0.92) + forward_x
    r_out_bot_x = int(W * 0.92) + forward_x
    r_out_top_y = mid_top - forward_v
    r_out_bot_y = mid_bot + forward_v

    right = np.array([
        [r_in_top_x,  r_in_top_y ],   # TL inner
        [r_out_top_x, r_out_top_y],   # TR outer (far right, raised)
        [r_out_bot_x, r_out_bot_y],   # BR outer (far right, dropped)
        [r_in_bot_x,  r_in_bot_y ],   # BL inner
    ], dtype=np.float32)

    return {
        "left_screen":   left,
        "centre_screen": centre,
        "right_screen":  right,
    }


# ─────────────────────────────────────────────
# SECTION 4 – COMPLEX DASHBOARD TEXTURES
# ─────────────────────────────────────────────

# ── Palette helpers ──────────────────────────

PALETTES = {
    "cyan":  {"bg": (0, 10, 24),   "pri": (0, 220, 255),   "sec": (0, 90, 140),  "acc": (160, 245, 255), "hi": (255, 255, 255)},
    "gold":  {"bg": (18, 10, 0),   "pri": (255, 185, 0),   "sec": (130, 70, 0),  "acc": (255, 230, 120), "hi": (255, 255, 220)},
    "green": {"bg": (0, 14, 6),    "pri": (0, 255, 130),   "sec": (0, 90, 45),   "acc": (160, 255, 200), "hi": (220, 255, 240)},
}


def _bg(draw, W, H, pal):
    """Dark background with subtle grid — preserves existing image alpha."""
    sec = pal["sec"]
    gx, gy = W // 14, H // 10
    for x in range(0, W, gx):
        draw.line([(x, 0), (x, H)], fill=sec + (20,), width=1)
    for y in range(0, H, gy):
        draw.line([(0, y), (W, y)], fill=sec + (20,), width=1)
    # Scanlines
    for y in range(0, H, 3):
        draw.line([(0, y), (W, y)], fill=(0, 0, 0, 18), width=1)


def _bar_h(draw, x, y, w, h, pct, col, bg_col):
    """Horizontal progress bar."""
    draw.rectangle([x, y, x+w, y+h], fill=bg_col+(40,), outline=col+(80,), width=1)
    if pct > 0:
        draw.rectangle([x+1, y+1, x+1+int((w-2)*pct), y+h-1], fill=col+(200,))


def _bar_v(draw, x, y, w, h, pct, col, bg_col):
    """Vertical bar (for bar charts)."""
    bh = int(h * pct)
    draw.rectangle([x, y, x+w, y+h], fill=bg_col+(30,))
    if bh > 0:
        draw.rectangle([x, y+h-bh, x+w, y+h], fill=col+(190,))
    draw.rectangle([x, y, x+w, y+h], outline=col+(60,), width=1)


def _ring(draw, cx, cy, r, pct, col, thick=5):
    """Partial arc ring gauge (PIL arc in degrees)."""
    start = -210
    span  = int(240 * pct)
    draw.arc([cx-r, cy-r, cx+r, cy+r], start, start+240, fill=col+(50,), width=thick)
    if span > 0:
        draw.arc([cx-r, cy-r, cx+r, cy+r], start, start+span, fill=col+(220,), width=thick)
    draw.ellipse([cx-r-2, cy-r-2, cx+r+2, cy+r+2], outline=col+(30,), width=1)


def _label_bar(draw, lx, ly, lw, lh, col):
    """Simulate a short text label as coloured rectangles."""
    draw.rectangle([lx, ly, lx+lw, ly+lh], fill=col+(120,))
    draw.rectangle([lx+2, ly+2, lx+lw*3//4, ly+lh-2], fill=col+(60,))


def _section_header(draw, x, y, w, h, col):
    draw.rectangle([x, y, x+w, y+h], fill=col+(40,), outline=col+(150,), width=1)
    draw.rectangle([x+3, y+3, x+w//2, y+h-3], fill=col+(160,))


def _hex_grid_small(draw, x, y, w, h, col):
    """Small honeycomb fill inside a bounding box."""
    hs = 12
    dx = hs * math.sqrt(3)
    dy = hs * 1.5
    rng = np.random.default_rng(7)
    row = 0
    cy = y
    while cy < y + h + hs:
        col_i = 0
        cx = x + (dx/2 if row % 2 else 0)
        while cx < x + w + dx:
            pts = []
            for a in range(0, 360, 60):
                rad = math.radians(a)
                pts.append((cx + hs*math.cos(rad), cy + hs*math.sin(rad)))
            alpha = int(rng.integers(15, 60))
            draw.polygon(pts, outline=col+(alpha,), fill=col+(8,))
            cx += dx
            col_i += 1
        cy += dy
        row += 1


def _mini_line_graph(draw, x, y, w, h, col, seed=1):
    """Small sparkline / line graph."""
    rng = np.random.default_rng(seed)
    vals = [rng.uniform(0.1, 0.95) for _ in range(20)]
    pts  = [(x + int(i*(w-1)/(len(vals)-1)), y + h - int(vals[i]*(h-2))) for i in range(len(vals))]
    draw.rectangle([x, y, x+w, y+h], fill=(0,0,0,40), outline=col+(80,), width=1)
    for i in range(len(pts)-1):
        draw.line([pts[i], pts[i+1]], fill=col+(200,), width=2)
    for pt in pts:
        draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2], fill=col+(255,))


def _radar_mini(draw, cx, cy, r, col, seed=3):
    rng = np.random.default_rng(seed)
    axes = 6
    # Background spokes & rings
    for ring in [r, r*0.66, r*0.33]:
        pts = []
        for i in range(axes):
            ang = math.radians(i*60 - 90)
            pts.append((int(cx + ring*math.cos(ang)), int(cy + ring*math.sin(ang))))
        pts.append(pts[0])
        for j in range(len(pts)-1):
            draw.line([pts[j], pts[j+1]], fill=col+(30,), width=1)
    for i in range(axes):
        ang = math.radians(i*60 - 90)
        draw.line([(cx, cy), (int(cx+r*math.cos(ang)), int(cy+r*math.sin(ang)))],
                  fill=col+(40,), width=1)
    # Data polygon
    vals = [rng.uniform(0.3, 0.95) for _ in range(axes)]
    dpts = []
    for i in range(axes):
        ang = math.radians(i*60 - 90)
        rv  = r * vals[i]
        dpts.append((int(cx + rv*math.cos(ang)), int(cy + rv*math.sin(ang))))
    dpts.append(dpts[0])
    draw.polygon(dpts[:-1], fill=col+(40,), outline=col+(200,))
    for pt in dpts[:-1]:
        draw.ellipse([pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3], fill=col+(255,))


def _world_map_lines(draw, x, y, w, h, pri, sec):
    """Compact lat/lon globe lines inside a box."""
    draw.rectangle([x, y, x+w, y+h], fill=(0,0,0,50), outline=pri+(80,), width=1)
    # Latitude lines
    for lat in range(-90, 91, 20):
        ly = y + int(h/2 + (lat/90)*(h/2))
        draw.line([(x, ly), (x+w, ly)], fill=pri+(35,), width=1)
    # Longitude ellipses
    for lon in range(-180, 181, 30):
        lcx  = x + int(w/2 + (lon/180)*(w/2))
        semi = max(1, int(w/2 * abs(math.cos(math.radians(lon)))))
        draw.ellipse([lcx-semi, y+2, lcx+semi, y+h-2], outline=sec+(40,), width=1)
    # Cross
    draw.line([(x+w//2, y), (x+w//2, y+h)], fill=pri+(50,), width=1)
    draw.line([(x, y+h//2), (x+w, y+h//2)], fill=pri+(50,), width=1)
    # Blips
    rng = np.random.default_rng(9)
    for _ in range(12):
        bx = int(rng.integers(x+6, x+w-6))
        by = int(rng.integers(y+6, y+h-6))
        draw.ellipse([bx-3, by-3, bx+3, by+3], outline=pri+(200,), width=1)


def _matrix_strip(draw, x, y, w, h, col):
    """Vertical streams of hex digit blocks (matrix rain)."""
    rng  = np.random.default_rng(5)
    cw   = 14
    for cx in range(x, x+w, cw):
        sl   = int(rng.integers(5, 15))
        sy   = int(rng.integers(y, y+h))
        for i in range(sl):
            py  = (sy + i*14) % (y+h - y) + y
            if py > y+h: break
            al  = int(255 * (1 - i/sl))
            draw.rectangle([cx+2, py, cx+2+10, py+12], fill=col+(al//4,), outline=col+(al,), width=1)


def _corner_ticks(draw, W, H, col, tick=22):
    """Corner bracket marks + glowing border."""
    for (cx, cy, sdx, sdy) in [(0,0,1,1),(W,0,-1,1),(W,H,-1,-1),(0,H,1,-1)]:
        draw.line([(cx, cy),(cx+sdx*tick, cy)], fill=col+(255,), width=3)
        draw.line([(cx, cy),(cx, cy+sdy*tick)], fill=col+(255,), width=3)
    for i in range(5, 0, -1):
        a = int(190*(6-i)/5)
        draw.rectangle([i, i, W-i, H-i], outline=col+(a,), width=1)
    draw.rectangle([0, 0, W-1, H-1], outline=col+(255,), width=2)


# ════════════════════════════════════════════════════════════
#  CENTRE SCREEN  –  Main JARVIS command centre
#  Layout (800×600):
#    Top strip:    status bars + gauges row
#    Left col:     radar + arc reactor + suit stats
#    Centre col:   world map + central targeting ring + bar chart
#    Right col:    line graphs + repulsor charge + waveform
#    Bottom strip: waveform / bio + readout blocks
# ════════════════════════════════════════════════════════════

def make_centre_screen(W=800, H=600):
    pal  = PALETTES["cyan"]
    pri, sec, acc = pal["pri"], pal["sec"], pal["acc"]
    img  = Image.new("RGBA", (W, H), pal["bg"]+(195,))
    draw = ImageDraw.Draw(img)
    _bg(draw, W, H, pal)
    rng = np.random.default_rng(42)

    # ── TOP STATUS STRIP ─────────────────────────────────────────
    strip_h = 52
    draw.rectangle([0, 0, W, strip_h], fill=(0,8,22,210))
    draw.line([(0, strip_h), (W, strip_h)], fill=pri+(160,), width=1)

    # 5 status segment groups
    labels = ["POWER", "SHIELD", "WEAPONS", "COMMS", "NAVIGATION"]
    pcts   = [0.94,    0.77,     0.60,      0.88,    1.00]
    seg_w  = W // 5
    for i, (lbl, pct) in enumerate(zip(labels, pcts)):
        sx = i * seg_w + 4
        _section_header(draw, sx, 4, seg_w-8, 12, pri)
        _bar_h(draw, sx, 20, seg_w-8, 10, pct, pri, sec)
        # Value blocks
        for b in range(4):
            bx = sx + b*((seg_w-8)//4)
            filled = b < int(4*pct)+1
            c = pri if filled else sec
            draw.rectangle([bx+1, 34, bx+(seg_w-8)//4-2, 46], fill=c+(180 if filled else 40,))

    # ── LEFT COLUMN (x: 0 → W//3) ────────────────────────────────
    LC = W // 3
    col_y = strip_h + 8

    # Arc reactor circle
    arc_cx, arc_cy = LC//2, col_y + 55
    _ring(draw, arc_cx, arc_cy, 40, 0.94, pri, thick=6)
    _ring(draw, arc_cx, arc_cy, 30, 0.94, acc, thick=3)
    for r2 in [18, 10]:
        draw.ellipse([arc_cx-r2, arc_cy-r2, arc_cx+r2, arc_cy+r2], outline=pri+(120,), width=1)
    draw.ellipse([arc_cx-6, arc_cy-6, arc_cx+6, arc_cy+6], fill=acc+(240,))
    _label_bar(draw, arc_cx-28, arc_cy+46, 56, 8, pri)

    # Radar
    radar_y = col_y + 125
    _radar_mini(draw, LC//2, radar_y, 48, pri, seed=11)
    _label_bar(draw, LC//2-30, radar_y+54, 60, 8, acc)

    # Suit status bars
    sy = radar_y + 70
    suit_rows = [("HULL INTEGRITY", 0.88), ("POWER CELL", 0.94), ("WEAPON SYS", 0.60),
                 ("SHIELDS", 0.77), ("O2 SUPPLY", 0.99), ("TEMP", 0.45)]
    for i, (lbl, pct) in enumerate(suit_rows):
        ry = sy + i * 36
        _section_header(draw, 6, ry, LC-12, 10, pri)
        _bar_h(draw, 6, ry+12, LC-12, 12, pct,
               pri if pct < 0.5 else (acc if pct > 0.85 else pri), sec)
        # tiny value blocks
        for b in range(5):
            filled = b < int(5*pct)+1
            bx2 = 6 + b*((LC-12)//5)
            draw.rectangle([bx2+1, ry+26, bx2+(LC-12)//5-2, ry+32],
                           fill=pri+(160 if filled else 30,))

    # ── CENTRE COLUMN (x: W//3 → 2*W//3) ────────────────────────
    MC  = W // 3
    mcx = LC
    mcy = col_y

    # World map globe
    map_h = 160
    _world_map_lines(draw, mcx+6, mcy, MC-12, map_h, pri, sec)
    # Crosshair on map
    mcc_x, mcc_y = mcx + MC//2, mcy + map_h//2
    for r3, a3 in [(30, 180), (20, 120), (10, 80)]:
        draw.ellipse([mcc_x-r3, mcc_y-r3, mcc_x+r3, mcc_y+r3], outline=acc+(a3,), width=1)
    draw.line([(mcc_x-38, mcc_y),(mcc_x+38, mcc_y)], fill=acc+(100,), width=1)
    draw.line([(mcc_x, mcc_y-38),(mcc_x, mcc_y+38)], fill=acc+(100,), width=1)

    # Central large targeting ring
    ring_cy = mcy + map_h + 80
    _ring(draw, mcx+MC//2, ring_cy, 64, 0.78, pri, thick=7)
    _ring(draw, mcx+MC//2, ring_cy, 50, 0.55, acc, thick=4)
    for r4 in [40, 28, 14]:
        draw.ellipse([mcx+MC//2-r4, ring_cy-r4, mcx+MC//2+r4, ring_cy+r4],
                     outline=pri+(60,), width=1)
    # Crosshair
    draw.line([(mcx+MC//2-72, ring_cy),(mcx+MC//2+72, ring_cy)], fill=pri+(80,), width=1)
    draw.line([(mcx+MC//2, ring_cy-72),(mcx+MC//2, ring_cy+72)], fill=pri+(80,), width=1)
    # Corner brackets on target
    h4 = 64
    for (dx4, dy4, sx4, sy4) in [(-h4,-h4,1,1),(h4,-h4,-1,1),(h4,h4,-1,-1),(-h4,h4,1,-1)]:
        px4, py4 = mcx+MC//2+dx4, ring_cy+dy4
        t4 = 18
        draw.line([(px4,py4),(px4+sx4*t4,py4)], fill=acc+(220,), width=2)
        draw.line([(px4,py4),(px4,py4+sy4*t4)], fill=acc+(220,), width=2)
    _label_bar(draw, mcx+MC//2-35, ring_cy+70, 70, 9, acc)

    # Bar chart row (bottom of centre col)
    chart_y = ring_cy + 88
    chart_h = H - chart_y - 8
    if chart_h > 40:
        nbar = 10
        bar_w2 = (MC-20) // nbar
        for i in range(nbar):
            pct2 = rng.uniform(0.2, 0.95)
            _bar_v(draw, mcx+10+i*bar_w2, chart_y, bar_w2-3, chart_h, pct2, pri, sec)
        draw.line([(mcx+6, H-8),(mcx+MC-6, H-8)], fill=pri+(100,), width=1)

    # ── RIGHT COLUMN (x: 2*W//3 → W) ────────────────────────────
    RC  = W // 3
    rcx = 2*W//3

    # Three line graphs stacked
    graph_h = 80
    for gi, (gcol, gseed) in enumerate([(pri, 1),(acc, 3),(pri, 7)]):
        gx = rcx + 6
        gy = col_y + gi * (graph_h + 10)
        _mini_line_graph(draw, gx, gy, RC-12, graph_h, gcol, seed=gseed)
        _section_header(draw, gx, gy, 55, 9, gcol)

    # Hex grid fill block
    hex_y = col_y + 3*(graph_h+10) + 4
    hex_h = 90
    hex_bg = Image.new("RGBA", (RC-12, hex_h), pal["bg"]+(200,))
    hex_draw = ImageDraw.Draw(hex_bg)
    _hex_grid_small(hex_draw, 0, 0, RC-12, hex_h, pri)
    hex_bg_bordered = hex_bg.copy()
    ImageDraw.Draw(hex_bg_bordered).rectangle([0,0,RC-13,hex_h-1], outline=pri+(140,), width=1)
    img.paste(hex_bg_bordered, (rcx+6, hex_y), hex_bg_bordered)
    draw = ImageDraw.Draw(img)   # re-acquire draw handle after paste

    # Repulsor charge gauge
    rep_cx, rep_cy = rcx + RC//2, hex_y + hex_h + 55
    _ring(draw, rep_cx, rep_cy, 44, 0.88, pal["acc"], thick=6)
    _ring(draw, rep_cx, rep_cy, 32, 0.62, pri, thick=4)
    draw.ellipse([rep_cx-8, rep_cy-8, rep_cx+8, rep_cy+8], fill=acc+(200,))
    _label_bar(draw, rep_cx-30, rep_cy+50, 60, 9, acc)

    # Speed / altitude row of mini gauges
    gauge_y = rep_cy + 68
    for gi2, (gval, gmax, gcol2) in enumerate([(742,1000,pri),(3240,5000,acc),(88,100,pri)]):
        gcx = rcx + 20 + gi2 * (RC//3)
        if gauge_y + 28 < H:
            _ring(draw, gcx, gauge_y+20, 20, gval/gmax, gcol2, thick=3)
            _label_bar(draw, gcx-14, gauge_y+44, 28, 7, gcol2)

    # ── BOTTOM STRIP ─────────────────────────────────────────────
    bot_y = H - 48
    draw.rectangle([0, bot_y, W, H], fill=(0,6,18,200))
    draw.line([(0, bot_y),(W, bot_y)], fill=pri+(140,), width=1)

    # Waveform (bio / heartbeat)
    rng2 = np.random.default_rng(17)
    pts_w = []
    for xi in range(W):
        wave = math.sin(xi * 0.08) * 0.3 + math.sin(xi * 0.23) * 0.15
        pts_w.append((xi, bot_y + 24 + int(wave * 14)))
    for i in range(len(pts_w)-1):
        draw.line([pts_w[i], pts_w[i+1]], fill=pri+(180,), width=1)

    # Matrix strip on far right of bottom
    _matrix_strip(draw, W-100, bot_y+2, 98, 44, pri)

    # readout blocks at bottom left
    for bi3 in range(8):
        bx3 = 8 + bi3*38
        bh3 = int(rng2.integers(10, 34))
        draw.rectangle([bx3, H-4-bh3, bx3+30, H-4], fill=pri+(180,))
    draw.line([(4, H-6),(W//3, H-6)], fill=acc+(100,), width=1)

    _corner_ticks(draw, W, H, acc, tick=24)
    return img


# ════════════════════════════════════════════════════════════
#  LEFT SCREEN  –  Environmental / threat assessment
#  Layout (700×560):
#    Top band:   threat level bar + scrolling alerts
#    Left half:  radar scan + hex grid + matrix strip
#    Right half: 4 stacked bar meters + line graph + blip panel
#    Bottom:     waveform readout
# ════════════════════════════════════════════════════════════

def _draw_neural_network(draw, x, y, w, h, pri, acc, sec, rng):
    """
    Draws a full multi-layer neural network diagram inside bounding box (x,y,w,h).

    Architecture visualised:
        Input layer  (4 nodes)  →
        Hidden 1     (6 nodes)  →
        Hidden 2     (6 nodes)  →
        Hidden 3     (5 nodes)  →
        Output layer (3 nodes)

    Each node is a glowing circle.  Every connection is drawn as a thin line
    whose colour interpolates cyan→gold based on a random "weight" value.
    Active / high-weight connections are brighter.  Bias nodes (+1) shown at
    top of each hidden layer.  A small weight histogram sits below.
    """
    layers = [4, 6, 6, 5, 3]
    n_layers = len(layers)
    max_nodes = max(layers)

    # Layout: columns evenly spaced, nodes vertically centred
    col_xs = [x + int(w * (i + 0.5) / n_layers) for i in range(n_layers)]
    node_r = max(5, min(12, h // (max_nodes * 3)))
    v_gap  = min(28, (h - 30) // (max_nodes + 1))

    # Pre-compute node positions
    node_pos = {}   # (layer_i, node_j) → (px, py)
    for li, n in enumerate(layers):
        total_h = n * v_gap
        start_y = y + h // 2 - total_h // 2 + v_gap // 2
        for ni in range(n):
            node_pos[(li, ni)] = (col_xs[li], start_y + ni * v_gap)

    # ── Draw connections (weights) ──────────────────────────────
    for li in range(n_layers - 1):
        for ni in range(layers[li]):
            for nj in range(layers[li + 1]):
                weight = rng.uniform(-1.0, 1.0)
                abs_w  = abs(weight)
                # Colour: positive weight → cyan tint, negative → gold tint
                if weight > 0:
                    line_col = (int(acc[0]*abs_w), int(acc[1]*abs_w), int(acc[2]*abs_w))
                else:
                    line_col = (int(pri[0]*abs_w), int(pri[1]*abs_w), int(pri[2]*abs_w))
                alpha = int(40 + 160 * abs_w)
                px1, py1 = node_pos[(li, ni)]
                px2, py2 = node_pos[(li+1, nj)]
                draw.line([(px1, py1), (px2, py2)], fill=line_col+(alpha,), width=1)

    # ── Draw nodes ──────────────────────────────────────────────
    layer_colors = [
        acc,    # input  – bright accent
        pri,    # hidden1
        pri,    # hidden2
        acc,    # hidden3
        (255, 80, 80),  # output – red/alert
    ]
    layer_labels = ["INPUT", "H-1", "H-2", "H-3", "OUTPUT"]

    for li, n in enumerate(layers):
        lc = layer_colors[li]
        for ni in range(n):
            px, py = node_pos[(li, ni)]
            # Outer glow ring
            draw.ellipse([px-node_r-3, py-node_r-3, px+node_r+3, py+node_r+3],
                         outline=lc+(40,), width=1)
            # Main circle fill
            draw.ellipse([px-node_r, py-node_r, px+node_r, py+node_r],
                         fill=lc+(50,), outline=lc+(220,), width=2)
            # Inner bright dot
            draw.ellipse([px-3, py-3, px+3, py+3], fill=lc+(240,))

        # Layer label bar at top
        lbl_x = col_xs[li]
        lbl_y = y + 6
        lbl_w = 28
        draw.rectangle([lbl_x-lbl_w//2, lbl_y, lbl_x+lbl_w//2, lbl_y+8],
                       fill=lc+(80,), outline=lc+(160,), width=1)
        draw.rectangle([lbl_x-lbl_w//2+2, lbl_y+2, lbl_x+lbl_w//4, lbl_y+6],
                       fill=lc+(180,))

    # ── Bias nodes (small +1 circles above hidden layers) ───────
    for li in range(1, n_layers - 1):
        bx, by = col_xs[li], y + 18
        draw.ellipse([bx-5, by-5, bx+5, by+5],
                     fill=sec+(40,), outline=pri+(140,), width=1)
        draw.ellipse([bx-2, by-2, bx+2, by+2], fill=pri+(200,))

    # ── Activation curve mini-plot (bottom right of NN area) ────
    curve_x = x + w - 70
    curve_y = y + h - 32
    curve_w, curve_h = 62, 26
    draw.rectangle([curve_x, curve_y, curve_x+curve_w, curve_y+curve_h],
                   fill=(0,0,0,40), outline=acc+(80,), width=1)
    pts_act = []
    for xi in range(curve_w):
        t = (xi / curve_w) * 6 - 3        # -3 to 3
        sig = 1 / (1 + math.exp(-t))       # sigmoid
        pts_act.append((curve_x + xi, int(curve_y + curve_h - sig*(curve_h-4) - 2)))
    for i in range(len(pts_act)-1):
        draw.line([pts_act[i], pts_act[i+1]], fill=acc+(200,), width=1)
    # Axis lines
    draw.line([(curve_x, curve_y+curve_h//2),(curve_x+curve_w, curve_y+curve_h//2)],
              fill=acc+(60,), width=1)

    # ── Weight histogram (below NN) ─────────────────────────────
    hist_x = x + 6
    hist_y = y + h - 32
    hist_w = w - 80
    hist_h = 26
    if hist_w > 40:
        draw.rectangle([hist_x, hist_y, hist_x+hist_w, hist_y+hist_h],
                       fill=(0,0,0,40), outline=pri+(80,), width=1)
        n_bins = 16
        bin_w  = (hist_w - 4) // n_bins
        bin_vals = [rng.uniform(0.05, 1.0) for _ in range(n_bins)]
        for bi, bv in enumerate(bin_vals):
            bx_h = hist_x + 2 + bi * bin_w
            bh_h = int(bv * (hist_h - 4))
            bc   = pri if bv > 0.5 else acc
            draw.rectangle([bx_h, hist_y+hist_h-2-bh_h, bx_h+bin_w-1, hist_y+hist_h-2],
                           fill=bc+(180,))
        draw.line([(hist_x+hist_w//2, hist_y),(hist_x+hist_w//2, hist_y+hist_h)],
                  fill=pri+(60,), width=1)


def make_left_screen(W=700, H=560):
    """
    Left screen: Neural Network visualisation dashboard.
    Sections:
      Top band:     network title + epoch/loss readouts
      Main area:    large multi-layer NN diagram
      Right strip:  loss curve + accuracy bar + layer stats
      Bottom strip: training waveform
    """
    pal  = PALETTES["gold"]
    pri, sec, acc = pal["pri"], pal["sec"], pal["acc"]
    # Make background semi-transparent from the start
    img  = Image.new("RGBA", (W, H), pal["bg"] + (200,))
    draw = ImageDraw.Draw(img)
    rng  = np.random.default_rng(77)

    # ── subtle grid background (reuse _bg but on semi-transparent canvas) ──
    gx, gy = W // 14, H // 10
    for xi in range(0, W, gx):
        draw.line([(xi, 0),(xi, H)], fill=sec+(18,), width=1)
    for yi in range(0, H, gy):
        draw.line([(0, yi),(W, yi)], fill=sec+(18,), width=1)
    for yi in range(0, H, 3):
        draw.line([(0, yi),(W, yi)], fill=(0,0,0,12), width=1)

    # ── TOP HEADER BAND ──────────────────────────────────────────
    band_h = 46
    draw.rectangle([0, 0, W, band_h], fill=(18,10,0,210))
    draw.line([(0, band_h),(W, band_h)], fill=pri+(200,), width=2)

    # Title bar blocks (simulate "NEURAL NETWORK ANALYSIS")
    title_blocks = [60, 45, 55, 70, 30]
    tx = 10
    for tb in title_blocks:
        draw.rectangle([tx, 10, tx+tb, 22], fill=pri+(160,))
        draw.rectangle([tx, 26, tx+int(tb*0.6), 34], fill=acc+(80,))
        tx += tb + 8

    # Epoch counter (right side)
    epoch_segs = 20
    seg_w2 = 18
    for i in range(epoch_segs):
        bx2 = W - 10 - (epoch_segs-i)*(seg_w2+2)
        filled = i < 14
        draw.rectangle([bx2, 8, bx2+seg_w2, band_h-8],
                       fill=pri+(190 if filled else 40,),
                       outline=pri+(80,), width=1)

    # ── MAIN NEURAL NETWORK DIAGRAM ─────────────────────────────
    # Takes up 60% of width, full height minus top/bottom bands
    nn_x = 6
    nn_y = band_h + 8
    nn_w = int(W * 0.60)
    nn_h = H - band_h - 56

    # NN bounding box
    draw.rectangle([nn_x, nn_y, nn_x+nn_w, nn_y+nn_h],
                   fill=(0,0,0,30), outline=pri+(80,), width=1)

    _draw_neural_network(draw, nn_x+6, nn_y+6, nn_w-12, nn_h-12, pri, acc, sec, rng)

    # ── RIGHT STRIP: metrics panels ──────────────────────────────
    rs_x = nn_x + nn_w + 8
    rs_w = W - rs_x - 8
    cy_r = nn_y

    # Loss curve
    lc_h = 100
    _mini_line_graph(draw, rs_x, cy_r, rs_w, lc_h, pri, seed=3)
    _section_header(draw, rs_x, cy_r, 50, 9, pri)

    # Accuracy ring gauge
    cy_r2 = cy_r + lc_h + 12
    ring_cx = rs_x + rs_w // 2
    _ring(draw, ring_cx, cy_r2 + 44, 38, 0.87, acc, thick=6)
    _ring(draw, ring_cx, cy_r2 + 44, 27, 0.92, pri, thick=3)
    draw.ellipse([ring_cx-6, cy_r2+38, ring_cx+6, cy_r2+50], fill=acc+(200,))
    _label_bar(draw, ring_cx-24, cy_r2+88, 48, 9, acc)

    # Per-layer stats (5 stacked bars)
    cy_r3 = cy_r2 + 106
    layer_names_stats = ["INPUT", "HIDDEN-1", "HIDDEN-2", "HIDDEN-3", "OUTPUT"]
    layer_acts        = [1.00,    0.72,       0.65,       0.81,       0.91  ]
    for i, (ln, la) in enumerate(zip(layer_names_stats, layer_acts)):
        my2 = cy_r3 + i * 34
        if my2 + 30 > H - 52: break
        _section_header(draw, rs_x, my2, rs_w, 9, pri)
        bar_c = (255,80,30) if la > 0.85 else (acc if la > 0.65 else pri)
        _bar_h(draw, rs_x, my2+11, rs_w, 12, la, bar_c, sec)
        for ti in range(4):
            tx2 = rs_x + ti*(rs_w//4)
            draw.line([(tx2, my2+23),(tx2, my2+27)], fill=pri+(70,), width=1)

    # Gradient flow mini heatmap (below layer stats)
    hm_y = cy_r3 + len(layer_names_stats)*34 + 6
    hm_h = H - hm_y - 56
    if hm_h > 24:
        draw.rectangle([rs_x, hm_y, rs_x+rs_w, hm_y+hm_h],
                       fill=(0,0,0,40), outline=acc+(80,), width=1)
        cell_h2 = max(3, hm_h // 8)
        cell_w2 = max(3, rs_w // 5)
        for row2 in range(hm_h // cell_h2 + 1):
            for col2 in range(rs_w // cell_w2 + 1):
                val2 = rng.uniform(0, 1)
                r_c  = int(pri[0]*val2)
                g_c  = int(pri[1]*val2)
                b_c  = int(pri[2]*val2)
                a_c  = int(60 + 120*val2)
                draw.rectangle([rs_x+col2*cell_w2, hm_y+row2*cell_h2,
                                 rs_x+(col2+1)*cell_w2-1, hm_y+(row2+1)*cell_h2-1],
                                fill=(r_c,g_c,b_c,a_c))
        draw.rectangle([rs_x, hm_y, rs_x+rs_w, hm_y+hm_h], outline=acc+(100,), width=1)

    # ── BOTTOM TRAINING WAVEFORM STRIP ───────────────────────────
    bot_y2 = H - 50
    draw.rectangle([0, bot_y2, W, H], fill=(16,8,0,200))
    draw.line([(0, bot_y2),(W, bot_y2)], fill=pri+(140,), width=1)
    rng3 = np.random.default_rng(33)
    # Two overlapping waves: loss (gold) and accuracy (acc)
    for wave_col, seed_w, amp, freq in [(pri,33,0.35,0.08),(acc,44,0.22,0.13)]:
        pts3 = []
        rng_w = np.random.default_rng(seed_w)
        for xi in range(W):
            v = math.sin(xi*freq)*amp + rng_w.uniform(-0.08, 0.08)
            pts3.append((xi, bot_y2 + 24 + int(v*14)))
        for i in range(len(pts3)-1):
            draw.line([pts3[i], pts3[i+1]], fill=wave_col+(150,), width=1)

    # Mini bar readouts bottom-left
    for bi3 in range(7):
        bx3 = 8 + bi3*36
        bh3 = int(rng3.integers(8, 34))
        draw.rectangle([bx3, H-4-bh3, bx3+28, H-4], fill=pri+(160,))

    _corner_ticks(draw, W, H, acc, tick=22)
    return img


# ════════════════════════════════════════════════════════════
#  RIGHT SCREEN  –  Navigation / mission data
#  Layout (700×560):
#    Top band:  navigation status + waypoints
#    Left half: arc gauges (speed/altitude/fuel) + bar chart
#    Right half: world map + matrix rain + waveform bars
#    Bottom:    readout panel
# ════════════════════════════════════════════════════════════

def make_right_screen(W=700, H=560):
    pal  = PALETTES["green"]
    pri, sec, acc = pal["pri"], pal["sec"], pal["acc"]
    img  = Image.new("RGBA", (W, H), pal["bg"]+(195,))
    draw = ImageDraw.Draw(img)
    _bg(draw, W, H, pal)
    rng = np.random.default_rng(55)

    # ── TOP NAVIGATION BAND ──────────────────────────────────────
    band_h3 = 48
    draw.rectangle([0, 0, W, band_h3], fill=(0,12,6,220))
    draw.line([(0, band_h3),(W, band_h3)], fill=pri+(180,), width=2)

    # Waypoint progress dots
    n_wp = 12
    wp_w = (W - 60) // n_wp
    for i in range(n_wp):
        wx = 20 + i*wp_w + wp_w//2
        done = i < 5
        draw.ellipse([wx-8, 16, wx+8, 32],
                     fill=pri+(200,) if done else sec+(60,),
                     outline=pri+(200,), width=1)
        if i < n_wp-1:
            draw.line([(wx+8, 24),(wx+wp_w-8, 24)], fill=pri+(80,), width=1)

    # ── LEFT HALF ────────────────────────────────────────────────
    HW3  = W // 2
    cy3  = band_h3 + 10

    # Three arc gauges: speed / altitude / fuel
    gauge_defs = [(742, 1000, "SPD"), (3240, 5000, "ALT"), (88, 100, "FUEL")]
    gr = 52
    for gi3, (gval3, gmax3, glbl3) in enumerate(gauge_defs):
        gcx3 = HW3//4 + gi3*(HW3//3) - HW3//12
        gcy3 = cy3 + gr + 14
        _ring(draw, gcx3, gcy3, gr, gval3/gmax3, pri, thick=6)
        _ring(draw, gcx3, gcy3, gr-12, gval3/gmax3, acc, thick=3)
        for r6 in [24, 14]:
            draw.ellipse([gcx3-r6, gcy3-r6, gcx3+r6, gcy3+r6], outline=pri+(50,), width=1)
        draw.ellipse([gcx3-5, gcy3-5, gcx3+5, gcy3+5], fill=acc+(200,))
        _label_bar(draw, gcx3-22, gcy3+gr+8, 44, 9, acc)

    # Large bar chart below gauges
    bc_y = cy3 + gr*2 + 36
    bc_h = 120
    n_bars = 12
    bw_bar = (HW3-20) // n_bars
    for i in range(n_bars):
        pct4 = rng.uniform(0.15, 0.98)
        _bar_v(draw, 10+i*bw_bar, bc_y, bw_bar-3, bc_h, pct4, pri, sec)
    draw.rectangle([8, bc_y, HW3-8, bc_y+bc_h], outline=pri+(80,), width=1)
    draw.line([(8, bc_y+bc_h+1),(HW3-8, bc_y+bc_h+1)], fill=acc+(100,), width=1)

    # Hex grid panel
    hg_y3 = bc_y + bc_h + 12
    hg_h3 = H - hg_y3 - 52
    if hg_h3 > 30:
        hex_img3 = Image.new("RGBA", (HW3-12, hg_h3), pal["bg"]+(180,))
        _hex_grid_small(ImageDraw.Draw(hex_img3), 0, 0, HW3-12, hg_h3, pri)
        img.paste(hex_img3, (6, hg_y3), hex_img3)
        draw = ImageDraw.Draw(img)
        draw.rectangle([6, hg_y3, HW3-6, hg_y3+hg_h3], outline=pri+(100,), width=1)

    # ── RIGHT HALF ───────────────────────────────────────────────
    rcx3 = HW3 + 6

    # World map
    map_h3 = 150
    _world_map_lines(draw, rcx3, cy3, HW3-12, map_h3, pri, sec)

    # Line graphs x2
    for lg_i, lg_seed in enumerate([9, 13]):
        lgx3 = rcx3
        lgy3 = cy3 + map_h3 + 10 + lg_i*(90+8)
        _mini_line_graph(draw, lgx3, lgy3, HW3-12, 80, acc if lg_i else pri, seed=lg_seed)

    # Matrix rain strip
    mr_y3 = cy3 + map_h3 + 10 + 2*(90+8)
    mr_h3 = H - mr_y3 - 52
    if mr_h3 > 20:
        _matrix_strip(draw, rcx3, mr_y3, HW3-12, mr_h3, pri)
        draw.rectangle([rcx3, mr_y3, W-6, mr_y3+mr_h3], outline=pri+(80,), width=1)

    # ── BOTTOM STRIP ─────────────────────────────────────────────
    bot_y3 = H - 46
    draw.rectangle([0, bot_y3, W, H], fill=(0,10,4,210))
    draw.line([(0, bot_y3),(W, bot_y3)], fill=pri+(130,), width=1)
    rng4 = np.random.default_rng(44)
    pts4 = []
    for xi in range(W):
        v = math.sin(xi*0.07)*0.4 + rng4.uniform(-0.12, 0.12)
        pts4.append((xi, bot_y3 + 22 + int(v*13)))
    for i in range(len(pts4)-1):
        draw.line([pts4[i], pts4[i+1]], fill=pri+(170,), width=1)
    # Altitude bar mini
    for bi4 in range(10):
        bx5 = W - 10 - bi4*22
        bh5 = int(rng4.integers(8, 34))
        draw.rectangle([bx5, H-4-bh5, bx5+16, H-4], fill=pri+(160,))

    _corner_ticks(draw, W, H, acc, tick=22)
    return img


def make_screen_texture(width, height, hue="cyan", style="data"):
    """
    Drop-in replacement for the original make_screen_texture.
    Routes to the correct complex dashboard based on hue/style tag.
    """
    if hue == "cyan":
        return make_centre_screen(width, height)
    elif hue == "gold" or hue == "amber":
        return make_left_screen(width, height)
    else:                            # green / blue → right screen
        return make_right_screen(width, height)


# ─────────────────────────────────────────────
# SECTION 5 – PERSPECTIVE WARP & COMPOSITE  (unchanged)
# ─────────────────────────────────────────────

def warp_screen_to_wall(screen_pil, wall_quad, canvas_w, canvas_h):
    sw, sh = screen_pil.size
    src_pts = np.array([[0,0],[sw-1,0],[sw-1,sh-1],[0,sh-1]], dtype=np.float32)
    dst_pts = wall_quad.astype(np.float32)
    H_mat, _ = cv2.findHomography(src_pts, dst_pts)
    screen_np = cv2.cvtColor(np.array(screen_pil.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
    warped = cv2.warpPerspective(screen_np, H_mat, (canvas_w, canvas_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0,0))
    return warped


def composite_screen_on_image(base_bgr, warped_bgra, opacity=0.82):
    alpha       = (warped_bgra[:,:,3:4].astype(np.float32)/255.0) * opacity
    screen_bgr  = warped_bgra[:,:,:3].astype(np.float32)
    base_f      = base_bgr.astype(np.float32)
    composited  = base_f * (1-alpha) + screen_bgr * alpha
    return np.clip(composited, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# SECTION 6 – POST-PROCESSING  (unchanged)
# ─────────────────────────────────────────────

def apply_color_grade(img_bgr):
    arr = img_bgr.astype(np.float32)
    arr[:,:,0] = np.clip(arr[:,:,0]*1.20, 0, 255)
    arr[:,:,1] = np.clip(arr[:,:,1]*0.90, 0, 255)
    arr[:,:,2] = np.clip(arr[:,:,2]*0.72, 0, 255)
    arr = np.clip((arr-20)*1.15+20, 0, 255)
    arr = np.clip(arr*0.82, 0, 255)
    return arr.astype(np.uint8)


def apply_bloom(img_bgr, radius=9, strength=0.22):
    pil     = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    b_arr   = np.array(blurred).astype(np.float32)
    o_arr   = np.array(pil).astype(np.float32)
    blue_dominant = (b_arr[:,:,2] > 80).astype(np.float32)
    bloom   = b_arr * blue_dominant[:,:,np.newaxis] * strength
    result  = np.clip(o_arr+bloom, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def apply_scanlines(img_bgr, spacing=3, alpha=0.12):
    mask = np.zeros(img_bgr.shape[0], dtype=np.float32)
    mask[::spacing] = alpha
    mask = mask[:,np.newaxis,np.newaxis]
    result = img_bgr.astype(np.float32) * (1-mask)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_chromatic_aberration(img_bgr, shift=3):
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([np.roll(b,-shift,axis=1), g, np.roll(r,shift,axis=1)])


def apply_vignette_glow(img_bgr, color=(255,220,0), layers=50):
    h, w   = img_bgr.shape[:2]
    overlay = img_bgr.copy().astype(np.float32)
    for i in range(layers, 0, -1):
        alpha2 = 0.006*(layers-i+1)
        x0,y0,x1,y1 = i,i,w-i,h-i
        if x1<=x0 or y1<=y0: break
        cv2.rectangle(overlay, (x0,y0),(x1,y1), color[::-1], 2)
    result = cv2.addWeighted(img_bgr.astype(np.float32),1.0,
                             overlay-img_bgr.astype(np.float32), 0.3, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# SECTION 7 – HUD DECORATIONS  (unchanged)
# ─────────────────────────────────────────────

def draw_hud_overlays(img_bgr):
    pil  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    hud  = Image.new("RGBA", pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(hud)
    W, H = pil.size

    # Targeting rings at fan/light positions
    fan_positions = [(160,175),(540,285),(830,200),(1050,300)]
    for (cx,cy) in fan_positions:
        for r,a in [(45,160),(32,110),(18,70)]:
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline=(0,220,255,a), width=1)
        draw.line([(cx-55,cy),(cx+55,cy)], fill=(0,220,255,80), width=1)
        draw.line([(cx,cy-55),(cx,cy+55)], fill=(0,220,255,80), width=1)
        for angle in [0,90,180,270]:
            rad = math.radians(angle)
            ix,iy = int(cx+50*math.cos(rad)), int(cy+50*math.sin(rad))
            ox,oy = int(cx+58*math.cos(rad)), int(cy+58*math.sin(rad))
            draw.line([(ix,iy),(ox,oy)], fill=(0,220,255,200), width=2)

    # Top status bar
    draw.rectangle([0,0,W,30], fill=(0,10,25,180))
    draw.line([(0,30),(W,30)], fill=(0,200,255,180), width=1)
    for i in range(20):
        bx = 20+i*35
        col = (0,220,255,200) if i<14 else (0,60,90,120)
        draw.rectangle([bx,8,bx+28,22], fill=col, outline=(0,200,255,80), width=1)

    # Bottom-left data box
    draw.rectangle([10,H-85,180,H-10], fill=(0,10,25,160), outline=(0,220,255,160), width=1)
    for bi in range(6):
        bhi = 15+bi*5%25
        bxi = 20+bi*25
        draw.rectangle([bxi,H-20-bhi,bxi+18,H-20], fill=(0,200,255,160))

    # Bottom-right data box
    draw.rectangle([W-180,H-85,W-10,H-10], fill=(0,10,25,160), outline=(60,140,255,160), width=1)
    for bi in range(6):
        bhi = 20+(bi*7%20)
        bxi = W-170+bi*25
        draw.rectangle([bxi,H-20-bhi,bxi+18,H-20], fill=(60,140,255,160))

    # Perspective floor grid
    vp_x, vp_y = W//2, int(H*0.82)
    floor_start = int(H*0.83)
    for gy in range(floor_start, H, 40):
        draw.line([(0,gy),(W,gy)], fill=(0,180,255,20), width=1)
    for gx in range(0, W+1, 70):
        draw.line([(vp_x,vp_y),(gx,H)], fill=(0,180,255,15), width=1)

    # ── Curved screen edge glow lines ──
    # Draw thin glowing arcs along the joints between screens to sell the curve
    screen_join_left  = int(W * 0.365)
    screen_join_right = int(W * 0.635)
    top_y  = int(H * 0.14)
    bot_y  = int(H * 0.80)
    for dy in range(-2, 3):
        a = max(0, 80 - abs(dy)*25)
        draw.line([(screen_join_left, top_y+dy),(screen_join_left, bot_y+dy)],
                  fill=(0,220,255,a), width=1)
        draw.line([(screen_join_right, top_y+dy),(screen_join_right, bot_y+dy)],
                  fill=(0,220,255,a), width=1)
    # Top arc connecting all three screens
    for dx in range(-2, 3):
        a = max(0, 60 - abs(dx)*18)
        draw.line([(int(W*0.10), top_y+dx),(int(W*0.90), top_y+dx)],
                  fill=(0,180,255,a), width=1)
    # Bottom arc
    for dx in range(-2, 3):
        a = max(0, 60 - abs(dx)*18)
        draw.line([(int(W*0.10), bot_y+dx),(int(W*0.90), bot_y+dx)],
                  fill=(0,180,255,a), width=1)

    composited = Image.alpha_composite(pil, hud)
    return cv2.cvtColor(np.array(composited.convert("RGB")), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# SECTION 8 – MAIN PIPELINE
# ─────────────────────────────────────────────

def run(input_path, output_path, debug=False):
    print(f"[1/8] Loading image: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")
    H, W = img.shape[:2]
    print(f"      Size: {W}×{H}")

    print("[2/8] Detecting edges and Hough lines...")
    edges, lines, _ = detect_edges_and_lines(img, save_debug=debug)

    print("[3/8] Estimating vanishing point...")
    vp = estimate_vanishing_point(lines, W, H)

    print("[4/8] Defining curved triple-screen quads...")
    screens = get_triple_screen_quads(W, H, vp)

    # Map screen name → (hue_tag, tex_w, tex_h, opacity)
    screen_config = {
        "left_screen":   ("gold",  700, 560, 0.62),
        "centre_screen": ("cyan",  800, 600, 0.78),
        "right_screen":  ("green", 700, 560, 0.62),
    }

    print("[5/8] Generating complex dashboard textures and warping to screens...")
    result = img.copy()
    for screen_name, quad in screens.items():
        hue, tw, th, opacity = screen_config[screen_name]
        print(f"      Generating {screen_name} ({hue}) dashboard …")
        screen_tex = make_screen_texture(tw, th, hue=hue, style="data")
        warped = warp_screen_to_wall(screen_tex, quad, W, H)
        result = composite_screen_on_image(result, warped, opacity=opacity)
        print(f"      ✓ {screen_name}")

    print("[6/8] Applying colour grading (sci-fi cool tone)...")
    result = apply_color_grade(result)

    print("[7/8] Applying post-processing effects (bloom, scanlines, CA, vignette)...")
    result = apply_bloom(result, radius=9, strength=0.25)
    result = apply_scanlines(result, spacing=3, alpha=0.10)
    result = apply_chromatic_aberration(result, shift=3)
    result = apply_vignette_glow(result, color=(0,200,255))

    print("[8/8] Drawing HUD overlays...")
    result = draw_hud_overlays(result)

    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n✅  Saved → {output_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turn classroom walls into a curved triple-screen sci-fi display.")
    parser.add_argument("--input",  "-i", default="classroom.png", help="Input image path")
    parser.add_argument("--output", "-o", default="futuristic_result.jpg", help="Output image path")
    parser.add_argument("--debug",  "-d", action="store_true", help="Save debug edge/line images")
    args = parser.parse_args()
    run(args.input, args.output, debug=args.debug)