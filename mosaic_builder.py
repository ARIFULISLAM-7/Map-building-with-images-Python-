import os
import re
import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import offset_copy
from PIL import Image

# Many "JPG"-named photos here are actually HEIC data underneath (iPhone
# photos copied over without real conversion), and even genuine .HEIC
# files won't open unless this opener is registered -- installing the
# pillow-heif package alone isn't enough, Pillow only gains HEIC support
# once this call is made.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed -- HEIC files (and JPG-named "
          "files that are secretly HEIC) will fail to open. "
          "Install with: pip install pillow-heif")

# ---------------------------------------------------------
# Path Configurations
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEOJSON_PATH = os.path.join(SCRIPT_DIR, "bangladesh_districts_polygons.geojson")  # dissolved district-level polygons
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output_map.png")

# ---------------------------------------------------------
# Poster text & style (edit these to match your reference)
# ---------------------------------------------------------
TITLE_TEXT = '"64 DISTRICTS OF BANGLADESH"'
SUBTITLE_TEXT = ""  # e.g. "Time: ... / Age: ..." — leave blank to omit
CAPTION_NAME = "YOUR NAME"
CAPTION_SUB = "YOUR INSTITUTION"

BACKGROUND_COLOR = "#1a1712"       # dark warm background like the reference
DISTRICT_OUTLINE_COLOR = "white"
DISTRICT_OUTLINE_WIDTH = 0.5       # thin seam between districts; set to 0 for no border at all
SHADOW_OFFSET_POINTS = 3           # drop-shadow offset behind each photo
SHADOW_ALPHA = 0.45

if not os.path.exists(GEOJSON_PATH):
    raise FileNotFoundError(f"GeoJSON file not found: {GEOJSON_PATH}")

# ---------------------------------------------------------
# Load boundaries
# ---------------------------------------------------------
try:
    gdf = gpd.read_file(GEOJSON_PATH, engine="pyogrio")
except Exception:
    gdf = gpd.read_file(GEOJSON_PATH, engine="fiona")

if "geometry" not in gdf.columns or gdf.geometry.isna().all():
    raise ValueError(
        "This GeoJSON has no usable polygon geometry (it's likely a flat list of "
        "points/centroids, not a FeatureCollection of Polygon/MultiPolygon features). "
        "You need a district BOUNDARY file, not a district CENTROID file."
    )

NAME_FIELDS = ["name", "ADM2_EN", "district", "NAME_2", "District", "shapeName", "District_N", "ADM2_NAME"]


def get_name(row):
    for f in NAME_FIELDS:
        if f in row and row.get(f):
            return str(row.get(f)).strip()
    return None


def normalize(name):
    """Lowercase, strip punctuation/whitespace/'District' suffix for fuzzier matching."""
    n = name.lower()
    n = re.sub(r"\bdistrict\b", "", n)
    n = re.sub(r"[^a-z]", "", n)
    return n


# Common alternate spellings between older/newer district-name romanizations.
ALIASES = {
    "chittagong": "chattogram",
    "bogra": "bogura",
    "jessore": "jashore",
    "comilla": "cumilla",
    "barisal": "barishal",
    "jhenaidah": "jhenaidaha",
}


def candidate_keys(name):
    n = normalize(name)
    keys = {n}
    if n in ALIASES:
        keys.add(ALIASES[n])
    reverse = {v: k for k, v in ALIASES.items()}
    if n in reverse:
        keys.add(reverse[n])
    return keys


def polygon_to_path(poly):
    """Path for a single (Shapely) Polygon using ONLY its exterior ring --
    interior rings (holes) are intentionally ignored so the district's own
    photo fills the whole shape edge-to-edge, including any spots that
    would otherwise show up as a punched-out black gap (real enclosed
    holes are rare and, per the user's choice, are filled with the same
    photo rather than left as a hole)."""
    return Path(np.array(poly.exterior.coords.xy).T)


# ---------------------------------------------------------
# Index available photos by normalized district name
# ---------------------------------------------------------
photo_index = {}
if os.path.isdir(IMAGES_DIR):
    for fname in os.listdir(IMAGES_DIR):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".heic"):
            continue
        for key in candidate_keys(stem):
            photo_index[key] = os.path.join(IMAGES_DIR, fname)
else:
    print(f"Warning: images folder not found at {IMAGES_DIR}")

# ---------------------------------------------------------
# Setup canvas
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 24), dpi=300)
ax.set_facecolor(BACKGROUND_COLOR)
fig.patch.set_facecolor(BACKGROUND_COLOR)

missing_photos = []

# Country-wide bounds, used below purely as a sanity-check yardstick: a
# single district's main polygon should never approach the size of the
# whole country. If one does, that's a dissolve/topology glitch, not a
# real district, and we print exactly which one plus its measurements so
# we can pin down the cause instead of guessing.
_country_minx, _country_miny, _country_maxx, _country_maxy = gdf.total_bounds
_country_w = _country_maxx - _country_minx
_country_h = _country_maxy - _country_miny
OVERSIZE_FRACTION_THRESHOLD = 0.30  # generous: real largest districts sit ~0.20

# ---------------------------------------------------------
# Process each district
# ---------------------------------------------------------
for idx, row in gdf.iterrows():
    district_name = get_name(row)
    if not district_name:
        continue

    img_path = None
    for key in candidate_keys(district_name):
        if key in photo_index:
            img_path = photo_index[key]
            break

    if img_path:
        try:
            img = Image.open(img_path)

            # Cap resolution right after opening. Phone photos here run
            # 4000-6000px per side, and set_clip_path() below rasterizes a
            # full-resolution boolean mask at the IMAGE's native pixel size
            # every time it's called -- once for the main polygon, then
            # AGAIN for every minor fragment (up to 68 for Bhola). At full
            # resolution that's dozens of 30-70MB allocations per district,
            # which is what was blowing out memory. A district-sized patch
            # on the poster never needs more than ~1600px of detail, so
            # downsizing once here (mutates img in place, so every use of
            # it below -- main polygon and every minor fragment -- benefits)
            # cuts every one of those allocations by 10-20x with no visible
            # quality loss at print size.
            img.thumbnail((1600, 1600))
            img = img.convert("RGB")

            poly = row.geometry
            polys = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)

            # Many districts here are MultiPolygons that include tiny,
            # far-flung river-char/island fragments alongside the main
            # landmass (Bhola alone dissolves into 69 separate pieces).
            # Stretching one photo across the bounding box of ALL those
            # fragments makes each distant sliver show an unrelated crop
            # of the image -- the photo reads as torn/fragmented confetti
            # instead of one coherent picture (visible today on Sylhet,
            # Bhola, Kushtia, etc. in output_map.png).
            #
            # Fix: place the photo only on the district's largest polygon
            # (its main body), and draw every smaller fragment as a plain
            # filled shape -- the same look a no-photo district gets --
            # so the map stays geographically complete without the photo
            # itself getting shredded.
            polys_sorted = sorted(polys, key=lambda p: p.area, reverse=True)
            main_poly, minor_polys = polys_sorted[0], polys_sorted[1:]

            minx, miny, maxx, maxy = main_poly.bounds
            w_frac = (maxx - minx) / _country_w
            h_frac = (maxy - miny) / _country_h

            # DEBUG: print what we're about to draw for every matched
            # district, so the console log shows exactly what happened.
            print(f"[{district_name}] photo={os.path.basename(img_path)} "
                  f"bbox_w={w_frac:.1%} bbox_h={h_frac:.1%} "
                  f"holes={len(main_poly.interiors)} minor_parts={len(minor_polys)}")

            if w_frac > OVERSIZE_FRACTION_THRESHOLD or h_frac > OVERSIZE_FRACTION_THRESHOLD:
                # This "district" polygon spans an implausibly large chunk
                # of the whole country -- almost certainly a dissolve/
                # topology glitch, not a real district shape. Refuse to
                # stretch a photo across it (that's what produced the
                # giant, wrong-looking fill) and fall back to plain gray
                # instead, so the map stays safe/legible either way.
                print(f"  -> SKIPPING photo for {district_name}: bbox is "
                      f"{w_frac:.1%} x {h_frac:.1%} of the whole country, "
                      f"over the {OVERSIZE_FRACTION_THRESHOLD:.0%} sanity threshold. "
                      f"Filling gray instead -- this district's geometry needs "
                      f"to be inspected.")
                missing_photos.append(district_name + "  [SKIPPED: oversized geometry]")
                gdf.iloc[[idx]].plot(ax=ax, color="#252525", edgecolor=DISTRICT_OUTLINE_COLOR, linewidth=DISTRICT_OUTLINE_WIDTH)
                continue

            im = ax.imshow(
                img,
                extent=[minx, maxx, miny, maxy],
                origin="upper",
                zorder=1,
                aspect="auto",
            )

            main_path = polygon_to_path(main_poly)

            # Soft drop-shadow behind the photo (offset a few points in
            # figure space so it looks consistent regardless of the
            # district's real-world size), like the cutout-photo look
            # in the reference poster.
            shadow_transform = offset_copy(
                ax.transData, fig=fig,
                x=SHADOW_OFFSET_POINTS, y=-SHADOW_OFFSET_POINTS, units="points",
            )
            shadow_patch = PathPatch(
                main_path,
                facecolor="black",
                edgecolor="none",
                alpha=SHADOW_ALPHA,
                transform=shadow_transform,
                zorder=0.5,
            )
            ax.add_patch(shadow_patch)

            outline_patch = PathPatch(
                main_path,
                facecolor="none",
                edgecolor=DISTRICT_OUTLINE_COLOR,
                linewidth=DISTRICT_OUTLINE_WIDTH,
                transform=ax.transData,
                zorder=3,
            )
            ax.add_patch(outline_patch)

            clip_patch = PathPatch(main_path, transform=ax.transData)
            im.set_clip_path(clip_patch)

            # Minor fragments (small chars/islands) of this same district:
            # rather than a flat gray fill, stretch the SAME photo across
            # each fragment's own bounds too, so every black/gray gap on
            # the map becomes that district's photo instead of empty fill.
            # Each fragment gets its own imshow (stretched to its own small
            # bbox) since fragments are scattered and don't share one
            # bounding box -- otherwise the photo would smear across the
            # empty space between them.
            for minor_poly in minor_polys:
                mminx, mminy, mmaxx, mmaxy = minor_poly.bounds
                minor_im = ax.imshow(
                    img,
                    extent=[mminx, mmaxx, mminy, mmaxy],
                    origin="upper",
                    zorder=1,
                    aspect="auto",
                )
                minor_path = polygon_to_path(minor_poly)
                minor_im.set_clip_path(PathPatch(minor_path, transform=ax.transData))
                ax.add_patch(PathPatch(
                    minor_path,
                    facecolor="none",
                    edgecolor=DISTRICT_OUTLINE_COLOR,
                    linewidth=DISTRICT_OUTLINE_WIDTH,
                    zorder=3,
                ))

        except Exception as e:
            print(f"Error rendering photo for {district_name}: {e}")
            gdf.iloc[[idx]].plot(ax=ax, color="#252525", edgecolor=DISTRICT_OUTLINE_COLOR, linewidth=DISTRICT_OUTLINE_WIDTH)
    else:
        missing_photos.append(district_name)
        gdf.iloc[[idx]].plot(ax=ax, color="#252525", edgecolor=DISTRICT_OUTLINE_COLOR, linewidth=DISTRICT_OUTLINE_WIDTH)

# ---------------------------------------------------------
# Force the view to the whole country's true bounds. Relying on
# matplotlib's automatic autoscaling is fragile here: ax.add_patch()
# (used for shadows/outlines/clip paths) never extends the axes'
# data limits, only ax.imshow() and geopandas' .plot() do -- so
# depending on draw order the final view can silently "lock on" to
# just the last few districts that happened to extend it, instead
# of the full country. Setting xlim/ylim explicitly, using the same
# total_bounds computed earlier, guarantees the whole map is shown
# regardless of which artists extended autoscale during the loop.
_pad_x = _country_w * 0.02
_pad_y = _country_h * 0.02
ax.set_xlim(_country_minx - _pad_x, _country_maxx + _pad_x)
ax.set_ylim(_country_miny - _pad_y, _country_maxy + _pad_y)
ax.set_aspect("equal")

# ---------------------------------------------------------
# Title / caption text (poster-style, matches the reference layout)
# ---------------------------------------------------------
ax.axis("off")

if TITLE_TEXT:
    fig.text(0.5, 0.965, TITLE_TEXT, ha="center", va="top",
              color="white", fontsize=30, fontweight="bold")
if SUBTITLE_TEXT:
    fig.text(0.98, 0.93, SUBTITLE_TEXT, ha="right", va="top",
              color="white", fontsize=16)
if CAPTION_NAME:
    fig.text(0.02, 0.03, CAPTION_NAME, ha="left", va="bottom",
              color="white", fontsize=22, fontweight="bold")
if CAPTION_SUB:
    fig.text(0.02, 0.015, CAPTION_SUB, ha="left", va="bottom",
              color="white", fontsize=15)

plt.savefig(
    OUTPUT_PATH,
    bbox_inches="tight",
    pad_inches=0.2,
    facecolor=fig.get_facecolor(),
)

print(f"Map saved successfully: {OUTPUT_PATH}")
if missing_photos:
    print(f"\n{len(missing_photos)} district(s) had no matching photo (filled with dark gray instead):")
    for name in missing_photos:
        print(f"  - {name}")