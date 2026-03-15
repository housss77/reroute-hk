import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# 1. Read estate data
# =========================
estate_path = r"D:\datahack\MHE_21C_converted.geojson"
gdf = gpd.read_file(estate_path)

gdf = gdf[["estate", "estate_eng", "estate_chi", "age_5", "t_pop", "geometry"]].copy()
gdf["age_5"] = gdf["age_5"].astype(float)
gdf["t_pop"] = gdf["t_pop"].astype(float)
gdf = gdf[gdf["t_pop"] > 0].copy()

gdf["elderly_share"] = gdf["age_5"] / gdf["t_pop"]
gdf["elder_dominant"] = gdf["elderly_share"] > 0.20

# =========================
# 2. Read HK coastline / boundary
# =========================
boundary_path = r"D:\datahack\Shoreline.geojson"
hk_boundary = gpd.read_file(boundary_path)

print("Estate CRS:", gdf.crs)
print("Boundary CRS:", hk_boundary.crs)

# 如果坐标系不同，统一到 estate 的 CRS
if hk_boundary.crs != gdf.crs:
    hk_boundary = hk_boundary.to_crs(gdf.crs)

# =========================
# 3. Plot
# =========================
fig, ax = plt.subplots(figsize=(12, 12))

# 先画海岸线/香港轮廓
hk_boundary.plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    linewidth=0.8
)

# 再画所有 estate 作为浅灰色
gdf.plot(
    ax=ax,
    color="lightgrey",
    edgecolor="white",
    linewidth=0.2
)

# 高亮 elder-dominant estates
gdf[gdf["elder_dominant"]].plot(
    ax=ax,
    color="red",
    edgecolor="black",
    linewidth=0.4
)

# =========================
# 4. Add legend
# =========================
legend_elements = [
    Patch(facecolor="lightgrey", edgecolor="white", label="Non-elder-dominant estates"),
    Patch(facecolor="red", edgecolor="black", label="Elder-dominant estates"),
]

ax.legend(handles=legend_elements, loc="lower left", fontsize=10)

ax.set_title("Elder-dominant Estates in Hong Kong", fontsize=14)
ax.set_axis_off()

plt.tight_layout()

# =========================
# 5. Save output
# =========================
output_path = r"D:\datahack\elder_dominant_estates_map.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Map saved to: {output_path}")

plt.show()