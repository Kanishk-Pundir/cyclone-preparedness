"""
Generate presentation-ready dataset acquisition cards matching the user's reference design.
Creates:
  1. docs/dataset_cards_grid.png  (2x3 layout for 16:9 slide)
  2. docs/dataset_cards_row1.png  (Row 1: 3 cards)
  3. docs/dataset_cards_row2.png  (Row 2: 3 cards)
  4. Individual card PNGs in docs/cards/
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

os.makedirs("docs/cards", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# 6 Dataset definitions
DATASETS = [
    {
        "id": "census",
        "title": "Demographics & Amenities",
        "source": "Census 2011 Village Directory (DCHB)",
        "desc": "Population, households, SC/ST share, tap water",
        "subtext": "Pucca road access + rural socio-economic baseline",
        "bg_color": "#F0F6FF",      # Soft blue
        "border_color": "#4A7BB0",  # Steel blue border
        "accent_color": "#2B6CB0",  # Darker blue text/icon
        "icon_type": "people"
    },
    {
        "id": "shrug",
        "title": "Spatial Boundaries",
        "source": "SHRUG 2.1 Polygons (DDL)",
        "desc": "1,879 village polygons (EPSG:32643 UTM)",
        "subtext": "Canonical GIS backbone & pc11_id matching",
        "bg_color": "#F0FDF4",      # Soft mint green
        "border_color": "#489A65",  # Green border
        "accent_color": "#276749",  # Forest green
        "icon_type": "map"
    },
    {
        "id": "health",
        "title": "Healthcare Facilities",
        "source": "National HMIS Directory (MoHFW)",
        "desc": "1,351 geocoded physical health centres & hospitals",
        "subtext": "Nearest facility distance & 10 km density counts",
        "bg_color": "#FFF7ED",      # Soft peach / amber
        "border_color": "#C97A3E",  # Amber border
        "accent_color": "#C05621",  # Warm terracotta
        "icon_type": "hospital"
    },
    {
        "id": "gauges",
        "title": "Telemetric Rain Gauges",
        "source": "KSNDMC Weather Portal",
        "desc": "2,843 coastal telemetric rain stations (KML)",
        "subtext": "Localized early warning & 20 km density coverage",
        "bg_color": "#F5F3FF",      # Soft lavender / purple
        "border_color": "#7C5CBF",  # Purple border
        "accent_color": "#553C9A",  # Deep purple
        "icon_type": "rain"
    },
    {
        "id": "shelters",
        "title": "Evacuation Shelters",
        "source": "NCRMP-II DPRs (NDMA / World Bank)",
        "desc": "16 dedicated Multi-Purpose Cyclone Shelters",
        "subtext": "12,050 total capacity & capacity per capita",
        "bg_color": "#FEFCE8",      # Soft pale yellow
        "border_color": "#B89B24",  # Ochre border
        "accent_color": "#975A16",  # Dark gold / bronze
        "icon_type": "shelter"
    },
    {
        "id": "hazard",
        "title": "Hazard & Ground Truth",
        "source": "CSIR-NIO (CVI) & KSDMA Tauktae Reports",
        "desc": "Coastal Vulnerability Index + 194 Tauktae hit areas",
        "subtext": "Physical surge exposure & empirical ground-truth",
        "bg_color": "#FFF1F2",      # Soft rose
        "border_color": "#C44D68",  # Rose border
        "accent_color": "#9F1239",  # Deep crimson
        "icon_type": "cyclone"
    }
]

def draw_icon(ax, icon_type, x, y, size, color):
    """Draw clean vector icons inside the card."""
    if icon_type == "people":
        # Group / Building symbol
        circle = patches.Circle((x + size*0.35, y + size*0.7), size*0.2, facecolor="none", edgecolor=color, lw=1.8)
        circle2 = patches.Circle((x + size*0.7, y + size*0.7), size*0.16, facecolor="none", edgecolor=color, lw=1.6)
        arc = patches.Arc((x + size*0.35, y + size*0.15), size*0.6, size*0.5, theta1=0, theta2=180, edgecolor=color, lw=1.8)
        arc2 = patches.Arc((x + size*0.7, y + size*0.2), size*0.48, size*0.4, theta1=0, theta2=180, edgecolor=color, lw=1.6)
        ax.add_patch(circle)
        ax.add_patch(circle2)
        ax.add_patch(arc)
        ax.add_patch(arc2)
    elif icon_type == "map":
        # Polygon / GIS layer
        verts = [
            (x + size*0.1, y + size*0.3),
            (x + size*0.35, y + size*0.85),
            (x + size*0.85, y + size*0.8),
            (x + size*0.9, y + size*0.2),
            (x + size*0.45, y + size*0.1),
            (x + size*0.1, y + size*0.3)
        ]
        poly = patches.Polygon(verts, closed=True, facecolor="none", edgecolor=color, lw=1.8)
        ax.add_patch(poly)
        # Inner centroid node
        c = patches.Circle((x + size*0.5, y + size*0.45), size*0.08, facecolor=color, edgecolor=color)
        ax.add_patch(c)
    elif icon_type == "hospital":
        # Medical cross
        w = size * 0.22
        h = size * 0.7
        r1 = patches.Rectangle((x + (size-w)/2, y + (size-h)/2), w, h, facecolor=color, edgecolor=color)
        r2 = patches.Rectangle((x + (size-h)/2, y + (size-w)/2), h, w, facecolor=color, edgecolor=color)
        ax.add_patch(r1)
        ax.add_patch(r2)
    elif icon_type == "rain":
        # Cloud with rain droplets
        cloud = patches.Arc((x + size*0.45, y + size*0.65), size*0.6, size*0.45, theta1=0, theta2=180, edgecolor=color, lw=1.8)
        c1 = patches.Arc((x + size*0.25, y + size*0.5), size*0.35, size*0.35, theta1=90, theta2=270, edgecolor=color, lw=1.8)
        c2 = patches.Arc((x + size*0.75, y + size*0.5), size*0.35, size*0.35, theta1=-90, theta2=90, edgecolor=color, lw=1.8)
        line = patches.PathPatch(Path([(x+size*0.25, y+size*0.33), (x+size*0.75, y+size*0.33)]), edgecolor=color, lw=1.8)
        ax.add_patch(cloud); ax.add_patch(c1); ax.add_patch(c2); ax.add_patch(line)
        # Drops
        for dx in [0.32, 0.52, 0.72]:
            ax.plot([x + size*dx, x + size*(dx-0.08)], [y + size*0.22, y + size*0.05], color=color, lw=1.8)
    elif icon_type == "shelter":
        # Safe shelter / house with roof
        roof = patches.Polygon([
            (x + size*0.1, y + size*0.45),
            (x + size*0.5, y + size*0.85),
            (x + size*0.9, y + size*0.45)
        ], closed=True, facecolor="none", edgecolor=color, lw=1.8)
        base = patches.Rectangle((x + size*0.2, y + size*0.1), size*0.6, size*0.35, facecolor="none", edgecolor=color, lw=1.8)
        door = patches.Rectangle((x + size*0.42, y + size*0.1), size*0.16, size*0.22, facecolor=color, edgecolor=color)
        ax.add_patch(roof); ax.add_patch(base); ax.add_patch(door)
    elif icon_type == "cyclone":
        # Wind / Cyclone spiral
        ax.plot([x + size*0.15, x + size*0.75], [y + size*0.75, y + size*0.75], color=color, lw=2.0)
        c1 = patches.Arc((x + size*0.75, y + size*0.65), size*0.2, size*0.2, theta1=-90, theta2=90, edgecolor=color, lw=2.0)
        ax.add_patch(c1)
        ax.plot([x + size*0.1, x + size*0.6], [y + size*0.5, y + size*0.5], color=color, lw=2.0)
        c2 = patches.Arc((x + size*0.6, y + size*0.4), size*0.2, size*0.2, theta1=-90, theta2=90, edgecolor=color, lw=2.0)
        ax.add_patch(c2)
        ax.plot([x + size*0.25, x + size*0.8], [y + size*0.25, y + size*0.25], color=color, lw=2.0)

def draw_single_card(ax, card, x_left, y_bottom, width, height):
    """Renders one card exactly styled like the user reference."""
    # Rounded card background
    box = patches.FancyBboxPatch(
        (x_left, y_bottom), width, height,
        boxstyle="round,pad=0,rounding_size=0.03",
        facecolor=card["bg_color"],
        edgecolor=card["border_color"],
        linewidth=1.4,
        zorder=1
    )
    ax.add_patch(box)
    
    # Coordinates inside card
    pad_x = width * 0.06
    icon_w = width * 0.09
    icon_h = height * 0.16
    icon_x = x_left + pad_x
    icon_y = y_bottom + height - icon_h - height * 0.08
    
    # 1. Icon
    draw_icon(ax, card["icon_type"], icon_x, icon_y, icon_w, card["accent_color"])
    
    # 2. Title (Bold)
    ax.text(
        x_left + pad_x, y_bottom + height * 0.68,
        card["title"],
        fontsize=12, fontweight="bold", color="#111827",
        fontfamily="DejaVu Sans", va="center", zorder=3
    )
    
    # 3. Source Subtitle
    ax.text(
        x_left + pad_x, y_bottom + height * 0.54,
        card["source"],
        fontsize=9.2, color="#4B5563",
        fontfamily="DejaVu Sans", va="center", zorder=3
    )
    
    # 4. Thin horizontal divider line
    div_y = y_bottom + height * 0.40
    ax.plot(
        [x_left + pad_x, x_left + width - pad_x],
        [div_y, div_y],
        color=card["border_color"], alpha=0.35, linewidth=1.0, zorder=2
    )
    
    # 5. Description line
    ax.text(
        x_left + pad_x, y_bottom + height * 0.26,
        card["desc"],
        fontsize=9.0, color="#1F2937",
        fontfamily="DejaVu Sans", va="center", zorder=3
    )
    
    # 6. Colored bottom subtext
    ax.text(
        x_left + pad_x, y_bottom + height * 0.12,
        card["subtext"],
        fontsize=8.5, fontweight="bold", color=card["accent_color"],
        fontfamily="DejaVu Sans", va="center", zorder=3
    )

# ─── 1. RENDER 3-CARDS ROW (ROW 1) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 4.2), dpi=220)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

card_w = 0.30
card_h = 0.88
gap_x  = 0.035
start_x = (1.0 - (3 * card_w + 2 * gap_x)) / 2

for i, card in enumerate(DATASETS[:3]):
    x = start_x + i * (card_w + gap_x)
    draw_single_card(ax, card, x, 0.06, card_w, card_h)

plt.tight_layout()
plt.savefig("docs/dataset_acquisition_cards_row1.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig("notebooks/dataset_acquisition_cards_row1.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: docs/dataset_acquisition_cards_row1.png")

# ─── 2. RENDER 3-CARDS ROW (ROW 2) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 4.2), dpi=220)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

for i, card in enumerate(DATASETS[3:6]):
    x = start_x + i * (card_w + gap_x)
    draw_single_card(ax, card, x, 0.06, card_w, card_h)

plt.tight_layout()
plt.savefig("docs/dataset_acquisition_cards_row2.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig("notebooks/dataset_acquisition_cards_row2.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: docs/dataset_acquisition_cards_row2.png")

# ─── 3. RENDER 2x3 FULL SLIDE GRID (ALL 6 CARDS) ─────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# Title on top of slide
ax.text(0.5, 0.94, "Multi-Source Dataset Acquisition & Provenance",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#111827", fontfamily="DejaVu Sans")
ax.text(0.5, 0.89, "Granular Fusion of Administrative, GIS, Health, Weather, and Disaster Records (N = 1,879 Coastal Villages)",
        ha="center", va="center", fontsize=11, color="#6B7280", fontfamily="DejaVu Sans")

grid_w = 0.30
grid_h = 0.36
gap_x  = 0.035
start_x = (1.0 - (3 * grid_w + 2 * gap_x)) / 2

# Row 1 (y = 0.47)
for i, card in enumerate(DATASETS[:3]):
    x = start_x + i * (grid_w + gap_x)
    draw_single_card(ax, card, x, 0.47, grid_w, grid_h)

# Row 2 (y = 0.06)
for i, card in enumerate(DATASETS[3:6]):
    x = start_x + i * (grid_w + gap_x)
    draw_single_card(ax, card, x, 0.06, grid_w, grid_h)

plt.tight_layout()
plt.savefig("docs/dataset_acquisition_cards_all.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.savefig("notebooks/dataset_acquisition_cards_all.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: docs/dataset_acquisition_cards_all.png")

# ─── 4. INDIVIDUAL CARDS ─────────────────────────────────────────────────────
for card in DATASETS:
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=220)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")
    draw_single_card(ax, card, 0.05, 0.05, 0.90, 0.90)
    plt.tight_layout()
    plt.savefig(f"docs/cards/card_{card['id']}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()

print("All cards successfully created.")
