"""
Run this ONCE, locally, in the same folder as your bangladesh.geojson
(the 544-upazila file). It uses geopandas' real dissolve (a proper
geometric union via shapely) to merge upazilas into 64 clean district
polygons with NO internal seams -- unlike the quick JSON-grouping
version used earlier, which just bundled upazila pieces together
without merging their shared edges.

Usage:
    python dissolve_districts.py
"""
import os
import geopandas as gpd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(SCRIPT_DIR, "bangladesh.geojson")          # the 544-upazila file
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "bangladesh_districts_polygons.geojson")

if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(
        f"Couldn't find {SOURCE_PATH}. Put the upazila-level bangladesh.geojson "
        "in the same folder as this script (or edit SOURCE_PATH above)."
    )

gdf = gpd.read_file(SOURCE_PATH)

before = len(gdf)
gdf = gdf[gdf["district_name"].astype(str).str.strip() != ""]
dropped = before - len(gdf)

# True geometric dissolve: merges all upazila polygons sharing the
# same district_name into one clean, seamless district polygon.
districts = gdf.dissolve(by="district_name", as_index=False)

# The source file's own "name"/"bn_name" columns are the UPAZILA's name,
# not the district's -- drop those before renaming district_name into
# "name", or we'd end up with two columns both called "name".
districts = districts.drop(columns=["name", "bn_name"], errors="ignore")
districts = districts.rename(columns={"district_name": "name", "district_bn_name": "bn_name"})
keep_cols = [c for c in ["name", "bn_name", "district_id", "division_id", "division_name", "geometry"] if c in districts.columns]
districts = districts[keep_cols]

districts.to_file(OUTPUT_PATH, driver="GeoJSON")

print(f"Dissolved {before - dropped} upazila polygons into {len(districts)} clean district polygons.")
print(f"Dropped {dropped} upazila polygon(s) with no district_name assigned.")
print(f"Saved: {OUTPUT_PATH}")