#!/usr/bin/env python3
"""Builds the block texture atlas `assets/textures.png`.

The atlas is an 8-column grid of 256x256 tiles (16x16 pixel art scaled up
16x with nearest-neighbour). Tile `i` sits at column `i % 8`, row `i // 8`
and becomes texture-array layer `i` in the game, so the order below must
match the `TEX_*` constants in `src/constants.rs`.

Layers 0..16 are the original hand-made textures. They are read from the
existing atlas (either the old 4x4 layout or this 8-column one), so the
script can be re-run safely. The remaining layers are generated here from
those tiles (ores reuse the stone pattern, mossy stone the stone, podzol and
mycelium sides the dirt side...) so the new blocks keep the same style.

Usage: python3 tools/generate_block_textures.py [assets/textures.png]
"""

import random
import sys
from pathlib import Path

from PIL import Image

PIXELS = 16
TILE = 256
COLUMNS = 8
LAYERS = 56
ORIGINAL = 16

ROOT = Path(__file__).resolve().parent.parent
ATLAS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "textures.png"


# ── Reading the original tiles ─────────────────────────────────────────── #

def read_original_tiles(path):
    atlas = Image.open(path).convert("RGBA")
    columns = atlas.width // TILE
    tiles = []
    for i in range(ORIGINAL):
        x, y = (i % columns) * TILE, (i // columns) * TILE
        tile = atlas.crop((x, y, x + TILE, y + TILE))
        tiles.append(tile)
    return tiles


def to_pixels(tile):
    """Samples the centre of every 16x16 block of a 256px tile."""
    step = TILE // PIXELS
    return [[list(tile.getpixel((x * step + step // 2, y * step + step // 2)))
             for x in range(PIXELS)] for y in range(PIXELS)]


def blank(color=(0, 0, 0, 0)):
    return [[list(color) for _ in range(PIXELS)] for _ in range(PIXELS)]


def copy(px):
    return [[list(c) for c in row] for row in px]


def to_tile(px):
    img = Image.new("RGBA", (PIXELS, PIXELS))
    for y in range(PIXELS):
        for x in range(PIXELS):
            img.putpixel((x, y), tuple(int(max(0, min(255, round(v)))) for v in px[y][x]))
    return img.resize((TILE, TILE), Image.NEAREST)


# ── Colour helpers ─────────────────────────────────────────────────────── #

def luma(c):
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0


def ramp(palette, t):
    """Picks a colour from a dark→light palette by brightness `t` (0..1)."""
    t = max(0.0, min(0.999, t))
    return list(palette[int(t * len(palette))]) + [255]


def remap(px, palette, lo=None, hi=None):
    """Recolours a tile, keeping its brightness pattern."""
    values = [luma(c) for row in px for c in row if c[3] > 0]
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    out = copy(px)
    for y in range(PIXELS):
        for x in range(PIXELS):
            if px[y][x][3] == 0:
                continue
            t = (luma(px[y][x]) - lo) / max(hi - lo, 1e-3)
            out[y][x] = ramp(palette, t)
    return out


def noise_fill(rng, palette, weights=None):
    px = blank()
    for y in range(PIXELS):
        for x in range(PIXELS):
            px[y][x] = list(rng.choices(palette, weights=weights)[0]) + [255]
    return px


def cluster(px, rng, cx, cy, colors, size):
    """Paints an ore-like blob of `size` pixels grown from (cx, cy)."""
    cells = [(cx, cy)]
    painted = set()
    while cells and len(painted) < size:
        x, y = cells.pop(rng.randrange(len(cells)))
        if (x, y) in painted or not (0 <= x < PIXELS and 0 <= y < PIXELS):
            continue
        painted.add((x, y))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cells.append((x + dx, y + dy))
    for x, y in painted:
        px[y][x] = list(rng.choice(colors)) + [255]
    # One highlight pixel per blob reads as a glint.
    if painted:
        x, y = min(painted, key=lambda p: (p[1], p[0]))
        px[y][x] = list(colors[-1]) + [255]


def ore(base, seed, colors, blobs=5, size=(3, 6)):
    rng = random.Random(seed)
    px = copy(base)
    spots = [(2, 3), (10, 2), (6, 8), (13, 10), (3, 12), (9, 13), (12, 5)]
    rng.shuffle(spots)
    for cx, cy in spots[:blobs]:
        cluster(px, rng, cx + rng.randint(-1, 1), cy + rng.randint(-1, 1), colors,
                rng.randint(*size))
    return px


def top_band(side, top_px, rows, seed):
    """Replaces the top rows of a side texture with a ragged band of `top_px`."""
    rng = random.Random(seed)
    px = copy(side)
    for x in range(PIXELS):
        depth = rows + rng.choice((-1, 0, 0, 1))
        for y in range(depth):
            px[y][x] = list(top_px[y][x])
    return px


# ── New textures ───────────────────────────────────────────────────────── #

def build(original):
    base = [to_pixels(t) for t in original]
    stone, dirt, sand, leaves = base[3], base[2], base[4], base[8]
    wood_side = base[6]

    deepslate = remap(stone, [(38, 38, 44), (52, 52, 60), (66, 66, 74), (80, 80, 90), (96, 96, 106)])
    rng = random.Random(20)
    for y in range(PIXELS):
        if y % 4 == 1:
            for x in range(PIXELS):
                if rng.random() < 0.55:
                    deepslate[y][x] = [34, 34, 40, 255]

    tex = {}
    tex[16] = ore(stone, 16, [(18, 18, 20), (30, 30, 32), (46, 46, 50)], blobs=6, size=(4, 7))
    tex[17] = ore(stone, 17, [(160, 112, 86), (196, 150, 118), (226, 192, 165)])
    tex[18] = ore(deepslate, 18, [(196, 140, 20), (240, 200, 40), (255, 240, 120)], blobs=4)
    tex[19] = ore(deepslate, 19, [(28, 150, 150), (70, 220, 210), (200, 255, 250)], blobs=4,
                  size=(2, 5))
    tex[20] = deepslate
    tex[21] = remap(sand, [(150, 70, 22), (176, 88, 30), (196, 104, 38), (214, 122, 50)])

    def clay(seed, palette):
        rng = random.Random(seed)
        return noise_fill(rng, palette, weights=[1, 6, 2])

    tex[22] = clay(22, [(140, 82, 56), (158, 94, 66), (172, 106, 76)])
    tex[23] = clay(23, [(126, 52, 38), (146, 62, 46), (160, 74, 56)])
    tex[24] = clay(24, [(196, 166, 150), (212, 180, 163), (224, 194, 178)])

    rng = random.Random(25)
    podzol = noise_fill(rng, [(70, 46, 20), (94, 64, 28), (118, 82, 38), (140, 100, 46)],
                        weights=[2, 5, 3, 1])
    tex[25] = podzol
    tex[26] = top_band(dirt, podzol, 4, 26)

    rng = random.Random(27)
    moss = noise_fill(rng, [(64, 86, 30), (84, 108, 40), (100, 128, 46), (122, 148, 56)],
                      weights=[2, 5, 4, 1])
    tex[27] = moss
    rng = random.Random(28)
    mossy = copy(stone)
    for _ in range(6):
        cluster(mossy, rng, rng.randrange(16), rng.randrange(16),
                [(70, 96, 34), (88, 116, 42), (104, 134, 50)], rng.randint(5, 14))
    tex[28] = mossy

    rng = random.Random(29)
    mycelium = noise_fill(rng, [(96, 84, 96), (112, 98, 110), (128, 112, 124), (150, 134, 148)],
                          weights=[2, 5, 4, 1])
    tex[29] = mycelium
    tex[30] = top_band(dirt, mycelium, 3, 30)

    rng = random.Random(31)
    stem = blank()
    for x in range(PIXELS):
        column = rng.choice([(196, 190, 178), (206, 200, 188), (214, 208, 196)])
        for y in range(PIXELS):
            c = column if rng.random() < 0.8 else (186, 180, 168)
            stem[y][x] = list(c) + [255]
    tex[31] = stem

    rng = random.Random(32)
    red_cap = noise_fill(rng, [(170, 32, 32), (196, 44, 42), (212, 58, 52)], weights=[2, 5, 2])
    for cx, cy, r in ((4, 4, 2), (11, 3, 1), (12, 11, 2), (4, 12, 1), (8, 8, 1)):
        for y in range(PIXELS):
            for x in range(PIXELS):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    red_cap[y][x] = [230, 226, 214, 255]
    tex[32] = red_cap
    rng = random.Random(33)
    tex[33] = noise_fill(rng, [(124, 92, 66), (146, 110, 80), (160, 122, 90)], weights=[2, 5, 2])

    sandstone = remap(sand, [(196, 176, 120), (214, 196, 140), (226, 210, 156), (236, 222, 170)])
    for y in (0, 5, 11):
        for x in range(PIXELS):
            sandstone[y][x] = [180, 160, 108, 255]
    for x in range(PIXELS):
        sandstone[15][x] = [170, 150, 100, 255]
    tex[34] = sandstone

    rng = random.Random(35)
    birch = noise_fill(rng, [(214, 212, 204), (226, 224, 216), (236, 234, 228)], weights=[2, 4, 3])
    for _ in range(14):
        y, x, length = rng.randrange(16), rng.randrange(14), rng.randint(2, 5)
        for i in range(length):
            if x + i < PIXELS:
                birch[y][x + i] = [44, 42, 40, 255] if i else [72, 68, 62, 255]
    tex[35] = birch

    cherry = copy(leaves)
    rng = random.Random(36)
    for y in range(PIXELS):
        for x in range(PIXELS):
            if cherry[y][x][3] > 0:
                t = luma(leaves[y][x]) * 2.2 + rng.uniform(-0.15, 0.15)
                cherry[y][x] = ramp([(206, 120, 160), (228, 150, 186), (242, 178, 206),
                                     (252, 204, 224)], t)
    tex[36] = cherry

    # Magma: dark crust cells separated by glowing cracks.
    rng = random.Random(37)
    seeds = [(rng.uniform(0, 16), rng.uniform(0, 16)) for _ in range(7)]
    magma = blank()
    for y in range(PIXELS):
        for x in range(PIXELS):
            d = sorted(min((x - sx - ox) ** 2 + (y - sy - oy) ** 2
                           for ox in (-16, 0, 16) for oy in (-16, 0, 16)) ** 0.5
                       for sx, sy in seeds)
            edge = d[1] - d[0]
            if edge < 0.9:
                c = (255, 196, 60)
            elif edge < 1.8:
                c = (232, 110, 24)
            else:
                c = rng.choice([(92, 26, 14), (110, 34, 18), (76, 20, 12)])
            magma[y][x] = list(c) + [255]
    tex[37] = magma

    rng = random.Random(38)
    drip = blank()
    for x in range(PIXELS):
        shade = rng.choice([(118, 92, 78), (134, 106, 90), (148, 118, 100)])
        for y in range(PIXELS):
            c = shade if rng.random() < 0.75 else (104, 80, 68)
            drip[y][x] = list(c) + [255]
    tex[38] = drip

    amethyst = blank()
    rng = random.Random(39)
    for y in range(PIXELS):
        for x in range(PIXELS):
            facet = ((x + y) // 3 + (x - y + 16) // 5) % 4
            c = [(92, 62, 148), (122, 88, 182), (150, 116, 210), (188, 156, 236)][facet]
            if rng.random() < 0.06:
                c = (236, 214, 255)
            amethyst[y][x] = list(c) + [255]
    tex[39] = amethyst

    rng = random.Random(40)
    tex[40] = noise_fill(rng, [(196, 198, 194), (214, 216, 212), (226, 228, 224), (176, 178, 176)],
                         weights=[3, 5, 3, 1])
    tex[41] = remap(wood_side, [(40, 24, 32), (56, 34, 44), (74, 46, 56), (92, 58, 66)])

    lamp = blank((62, 42, 24, 255))
    for y in range(2, 14):
        for x in range(2, 14):
            lamp[y][x] = [126, 78, 28, 255]
    for y in range(4, 12):
        for x in range(4, 12):
            lamp[y][x] = [255, 190, 62, 255]
    for y in range(5, 11):
        for x in range(5, 11):
            lamp[y][x] = [255, 232, 132, 255]
    for y, x in ((3, 3), (3, 12), (12, 3), (12, 12)):
        lamp[y][x] = [216, 150, 54, 255]
    tex[42] = lamp

    rng = random.Random(43)
    glowstone = noise_fill(rng, [(174, 92, 26), (214, 138, 36), (246, 190, 62), (255, 226, 114)],
                           weights=[1, 4, 4, 2])
    for cx, cy in ((3, 4), (11, 3), (8, 11), (4, 12)):
        cluster(glowstone, rng, cx, cy, [(255, 232, 136), (255, 246, 188)], 3)
    tex[43] = glowstone

    rng = random.Random(44)
    shroomlight = noise_fill(rng, [(176, 62, 28), (214, 92, 38), (244, 142, 58), (255, 194, 106)],
                             weights=[1, 4, 4, 2])
    for x in range(PIXELS):
        if x % 4 == 1:
            for y in range(PIXELS):
                if (x + y) % 3 != 0:
                    shroomlight[y][x] = [255, 218, 142, 255]
    tex[44] = shroomlight

    rng = random.Random(45)
    glow_crystal = blank()
    for y in range(PIXELS):
        for x in range(PIXELS):
            facet = ((x + y // 2) // 3 + (x - y + 16) // 5) % 4
            colors = [(72, 54, 126), (112, 82, 178), (158, 124, 224), (210, 184, 255)]
            color = colors[facet]
            if rng.random() < 0.1:
                color = (232, 224, 255)
            glow_crystal[y][x] = list(color) + [255]
    tex[45] = glow_crystal

    rng = random.Random(46)
    cobblestone = noise_fill(rng, [(82, 82, 86), (112, 112, 116), (140, 140, 144), (162, 162, 164)],
                             weights=[2, 5, 4, 1])
    for y in (4, 10):
        for x in range(PIXELS):
            if rng.random() < 0.75:
                cobblestone[y][x] = [66, 66, 70, 255]
    tex[46] = cobblestone

    rng = random.Random(47)
    planks = noise_fill(rng, [(112, 68, 34), (142, 88, 42), (166, 108, 54), (188, 130, 70)],
                        weights=[1, 4, 4, 1])
    for y in (0, 7, 15):
        for x in range(PIXELS):
            planks[y][x] = [90, 54, 30, 255]
    for y, x in ((3, 5), (11, 12)):
        for dx in range(2):
            planks[y][x + dx] = [94, 56, 30, 255]
    tex[47] = planks

    rng = random.Random(48)
    bricks = noise_fill(rng, [(142, 54, 42), (168, 70, 52), (192, 88, 64)], weights=[2, 5, 2])
    for y in (0, 7, 15):
        for x in range(PIXELS):
            bricks[y][x] = [94, 58, 52, 255]
    for y in range(1, PIXELS):
        offset = 4 if y < 7 or y > 8 else 0
        for x in (offset, offset + 8):
            if x < PIXELS:
                bricks[y][x] = [94, 58, 52, 255]
    tex[48] = bricks

    rng = random.Random(49)
    tex[49] = noise_fill(rng, [(88, 92, 84), (112, 116, 108), (136, 140, 130), (158, 160, 148)],
                         weights=[2, 5, 4, 1])

    rng = random.Random(50)
    basalt_side = noise_fill(rng, [(36, 36, 42), (50, 50, 56), (64, 64, 70), (78, 78, 84)],
                             weights=[2, 5, 3, 1])
    for x in range(PIXELS):
        if x % 5 == 2:
            for y in range(PIXELS):
                basalt_side[y][x] = [30, 30, 36, 255]
    tex[50] = basalt_side

    rng = random.Random(51)
    basalt_top = noise_fill(rng, [(40, 40, 46), (56, 56, 62), (72, 72, 78), (88, 88, 94)],
                            weights=[2, 5, 3, 1])
    for x in range(PIXELS):
        if x in (2, 13):
            for y in range(PIXELS):
                if rng.random() < 0.75:
                    basalt_top[y][x] = [32, 32, 38, 255]
    tex[51] = basalt_top

    rng = random.Random(52)
    obsidian = noise_fill(rng, [(18, 14, 34), (28, 22, 52), (40, 30, 72), (54, 42, 90)],
                          weights=[2, 5, 3, 1])
    for y, x in ((2, 5), (6, 11), (12, 3), (9, 14)):
        obsidian[y][x] = [104, 78, 150, 255]
    tex[52] = obsidian

    # Spare layers stay usable but obvious: plain stone.
    for i in range(53, LAYERS):
        tex[i] = stone
    return tex


def main():
    original = read_original_tiles(ATLAS)
    generated = build(original)
    rows = (LAYERS + COLUMNS - 1) // COLUMNS
    atlas = Image.new("RGBA", (COLUMNS * TILE, rows * TILE))
    for i in range(LAYERS):
        tile = original[i] if i < ORIGINAL else to_tile(generated[i])
        atlas.paste(tile, ((i % COLUMNS) * TILE, (i // COLUMNS) * TILE))
    atlas.save(ATLAS)
    print(f"wrote {ATLAS} ({atlas.width}x{atlas.height}, {LAYERS} layers)")


if __name__ == "__main__":
    main()
