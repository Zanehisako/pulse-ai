#!/usr/bin/env python3
"""
Generate a Modèle Conceptuel de Données (MCD) entity-relationship diagram
for the PIOS blood-supply platform.

Outputs: ml-backend/simulator/figures/mcd_diagram.png
"""

import matplotlib

matplotlib.use("Agg")

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURE_W, FIGURE_H = 20, 16
DPI = 200
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "mcd_diagram.png")

# Domain colour palette
COLORS = {
    "user": "#C6DBEF",  # blue
    "comm": "#C7E9C0",  # green
    "alert": "#FDCDAC",  # orange
    "notif": "#FFF2AE",  # yellow
    "ml": "#DECBE4",  # purple
    "etl": "#E0E0E0",  # light gray
    "file": "#B2DFDB",  # teal / cyan
}

BORDER_DEFAULT = "#444444"
BORDER_FILE = "#00796B"

# ---------------------------------------------------------------------------
# Entity definitions:  (name, colour_key, [attrs], x, y, w, h, dashed_border)
# ---------------------------------------------------------------------------

ENTITIES = [
    # ---- User (top-center) ----
    (
        "User",
        "user",
        ["email (PK)", "name", "password", "is_active", "date_joined"],
        8.8,
        13.8,
        2.4,
        1.8,
        False,
    ),
    ("Group / Permission", "user", ["name", "codename"], 5.0, 14.2, 2.4, 1.0, False),
    # ---- Communication (upper-left) ----
    (
        "Conversation",
        "comm",
        ["title", "type", "created_at"],
        2.0,
        12.0,
        2.4,
        1.2,
        False,
    ),
    (
        "Message",
        "comm",
        ["content", "priority", "created_at"],
        2.0,
        9.8,
        2.4,
        1.2,
        False,
    ),
    # ---- Alerts (upper-right) ----
    (
        "AlertRule",
        "alert",
        ["name", "scope_type", "trigger_type", "severity", "conditions (JSON)"],
        15.2,
        13.8,
        2.8,
        1.8,
        False,
    ),
    (
        "AlertEvent",
        "alert",
        ["id (UUID)", "severity", "status", "title", "context (JSON)", "blood_type"],
        15.2,
        11.2,
        2.8,
        1.9,
        False,
    ),
    (
        "AlertEventHistory",
        "alert",
        ["action", "actor", "snapshot (JSON)"],
        15.2,
        8.8,
        2.8,
        1.2,
        False,
    ),
    (
        "Notification",
        "notif",
        ["id (UUID)", "title", "body", "type", "sent_at"],
        11.5,
        9.2,
        2.5,
        1.6,
        False,
    ),
    # ---- ML (center-bottom) ----
    (
        "MLModelConfig",
        "ml",
        ["model_id", "model_type", "features (JSON)", "is_active"],
        7.2,
        6.2,
        2.8,
        1.5,
        False,
    ),
    (
        "PredictionLog",
        "ml",
        ["query", "features_input (JSON)", "prediction_output (JSON)", "success"],
        7.2,
        3.8,
        2.8,
        1.5,
        False,
    ),
    (
        "OrchestratorVariant",
        "ml",
        ["variant_id", "repo_id", "is_selected"],
        11.0,
        6.2,
        2.4,
        1.1,
        False,
    ),
    (
        "ModelStats",
        "ml",
        ["model_id", "stats (JSON)", "made_at"],
        11.0,
        4.6,
        2.4,
        1.1,
        False,
    ),
    (
        "DriftReport",
        "ml",
        ["model_id", "drift_detected", "severity", "feature_details (JSON)"],
        11.0,
        2.8,
        2.5,
        1.4,
        False,
    ),
    # ---- Dashboard / ETL (lower-left) ----
    (
        "ModelResponse",
        "etl",
        ["typeModel", "jsonResponse", "region", "period"],
        1.5,
        3.5,
        2.6,
        1.4,
        False,
    ),
    (
        "DataVersion",
        "etl",
        ["key", "last_updated", "version_hash"],
        1.5,
        1.5,
        2.6,
        1.2,
        False,
    ),
    # ---- Blood-supply file-based (lower-right) ----
    (
        "Donor",
        "file",
        ["donor_id", "demographics", "eligibility", "logistics"],
        15.2,
        5.6,
        2.5,
        1.4,
        True,
    ),
    (
        "Hospital",
        "file",
        ["hospital_id", "inventory", "demand", "environment"],
        15.2,
        3.6,
        2.5,
        1.4,
        True,
    ),
    (
        "SupplyPoint",
        "file",
        ["supply_id", "stock", "lead_time", "stockout indicators"],
        15.2,
        1.4,
        2.5,
        1.4,
        True,
    ),
]

# ---------------------------------------------------------------------------
# Relationships:
#   (from_entity, to_entity, card_from, card_to, style, label)
#   style: "solid" | "dashed"
# ---------------------------------------------------------------------------

RELATIONSHIPS = [
    # User <--> Group/Permission  M:N dashed
    ("User", "Group / Permission", "M", "N", "dashed", ""),
    # User <--> Conversation  M:N dashed
    ("User", "Conversation", "M", "N", "dashed", ""),
    # Message --> Conversation  N:1
    ("Message", "Conversation", "N", "1", "solid", ""),
    # Message --> User  N:1
    ("Message", "User", "N", "1", "solid", ""),
    # AlertEvent --> AlertRule  N:1
    ("AlertEvent", "AlertRule", "N", "1", "solid", ""),
    # AlertEventHistory --> AlertEvent  N:1
    ("AlertEventHistory", "AlertEvent", "N", "1", "solid", ""),
    # Notification --- AlertEvent  dashed (soft UUID ref)
    ("Notification", "AlertEvent", "", "", "dashed", "soft ref"),
    # PredictionLog --> User  N:1
    ("PredictionLog", "User", "N", "1", "solid", ""),
    # PredictionLog --> MLModelConfig  N:1
    ("PredictionLog", "MLModelConfig", "N", "1", "solid", ""),
    # AlertRule --> User (owner, optional)
    # OrchestratorVariant is standalone but logically near MLModelConfig
    # ModelStats / DriftReport standalone
]

# ---------------------------------------------------------------------------
# Helper: find the centre of an entity box by name
# ---------------------------------------------------------------------------


def _entity_rect(name):
    """Return (x, y, w, h) for entity *name*."""
    for ent in ENTITIES:
        if ent[0] == name:
            return ent[3], ent[4], ent[5], ent[6]
    raise KeyError(name)


def _center(name):
    x, y, w, h = _entity_rect(name)
    return x + w / 2, y + h / 2


def _edge_point(name, target_xy):
    """Return the point on the border of *name*'s box closest to *target_xy*."""
    x, y, w, h = _entity_rect(name)
    cx, cy = x + w / 2, y + h / 2
    tx, ty = target_xy

    dx = tx - cx
    dy = ty - cy

    if dx == 0 and dy == 0:
        return cx, cy

    # Scale factor to reach the rectangle edge
    sx = abs((w / 2) / dx) if dx != 0 else float("inf")
    sy = abs((h / 2) / dy) if dy != 0 else float("inf")
    s = min(sx, sy)

    return cx + dx * s, cy + dy * s


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_entity(ax, name, color_key, attrs, x, y, w, h, dashed):
    """Draw one entity box with title and attribute list."""
    fill = COLORS[color_key]
    edge_color = BORDER_FILE if dashed else BORDER_DEFAULT
    ls = "--" if dashed else "-"
    lw = 1.6 if not dashed else 1.8

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.06",
        facecolor=fill,
        edgecolor=edge_color,
        linewidth=lw,
        linestyle=ls,
        zorder=2,
    )
    ax.add_patch(box)

    # Title
    title_y = y + h - 0.22
    ax.text(
        x + w / 2,
        title_y,
        name,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        family="sans-serif",
        zorder=3,
    )

    # Separator line
    sep_y = title_y - 0.16
    ax.plot(
        [x + 0.10, x + w - 0.10],
        [sep_y, sep_y],
        color=edge_color,
        lw=0.7,
        ls=ls,
        zorder=3,
    )

    # Attributes
    attr_start = sep_y - 0.18
    for i, attr in enumerate(attrs):
        ay = attr_start - i * 0.19
        if ay < y + 0.06:
            break
        prefix = "• "
        ax.text(
            x + 0.14,
            ay,
            prefix + attr,
            ha="left",
            va="center",
            fontsize=5.6,
            family="sans-serif",
            zorder=3,
        )


def draw_relationship(ax, ent_from, ent_to, card_from, card_to, style, label):
    """Draw a relationship line between two entities with cardinality labels."""
    c_to = _center(ent_to)
    c_from = _center(ent_from)
    p1 = _edge_point(ent_from, c_to)
    p2 = _edge_point(ent_to, c_from)

    ls = "--" if style == "dashed" else "-"
    color = "#666666" if style == "dashed" else "#333333"
    lw = 1.1 if style == "dashed" else 1.3

    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            lw=lw,
            linestyle=ls,
        ),
        zorder=1,
    )

    # Cardinality labels
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.hypot(dx, dy)
    if length == 0:
        return

    # unit perpendicular for offset
    ux, uy = -dy / length, dx / length
    off = 0.16  # perpendicular offset

    if card_from:
        lx = p1[0] + dx * 0.08 + ux * off
        ly = p1[1] + dy * 0.08 + uy * off
        ax.text(
            lx,
            ly,
            card_from,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            color="#B71C1C",
            zorder=4,
        )

    if card_to:
        lx = p2[0] - dx * 0.08 + ux * off
        ly = p2[1] - dy * 0.08 + uy * off
        ax.text(
            lx,
            ly,
            card_to,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            color="#B71C1C",
            zorder=4,
        )

    if label:
        mx = (p1[0] + p2[0]) / 2 + ux * off
        my = (p1[1] + p2[1]) / 2 + uy * off
        ax.text(
            mx,
            my,
            label,
            fontsize=5.5,
            fontstyle="italic",
            ha="center",
            va="center",
            color="#555555",
            zorder=4,
        )


def draw_legend(ax):
    """Draw a legend box at the bottom of the figure."""
    legend_y = 0.15
    legend_x = 1.0

    items = [
        ("-", "#333333", 1.3, "FK constraint (solid line)"),
        ("--", "#666666", 1.1, "Soft / M:N reference (dashed line)"),
    ]

    for i, (ls, col, lw, txt) in enumerate(items):
        xi = legend_x + i * 5.5
        ax.plot([xi, xi + 0.8], [legend_y, legend_y], ls=ls, color=col, lw=lw, zorder=5)
        ax.text(
            xi + 1.0,
            legend_y,
            txt,
            fontsize=6.5,
            va="center",
            family="sans-serif",
            zorder=5,
        )

    # Dashed border sample
    xi = legend_x + 2 * 5.5
    sample = FancyBboxPatch(
        (xi, legend_y - 0.14),
        0.8,
        0.28,
        boxstyle="round,pad=0.04",
        facecolor=COLORS["file"],
        edgecolor=BORDER_FILE,
        linewidth=1.4,
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(sample)
    ax.text(
        xi + 1.0,
        legend_y,
        "File-based entity (dashed border)",
        fontsize=6.5,
        va="center",
        family="sans-serif",
        zorder=5,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIGURE_W, FIGURE_H))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, FIGURE_W)
    ax.set_ylim(0, FIGURE_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        FIGURE_W / 2,
        FIGURE_H - 0.25,
        "Modèle Conceptuel de Données (MCD) — PIOS Blood-Supply Platform",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        family="sans-serif",
    )

    # Draw entities
    for ent in ENTITIES:
        name, ckey, attrs, x, y, w, h, dashed = ent
        draw_entity(ax, name, ckey, attrs, x, y, w, h, dashed)

    # Draw relationships
    for rel in RELATIONSHIPS:
        draw_relationship(ax, *rel)

    # Domain group labels (subtle)
    group_labels = [
        (3.2, 13.45, "Communication", "#2E7D32"),
        (16.6, 15.75, "Alerting", "#E65100"),
        (10.0, 7.95, "ML / Prediction", "#6A1B9A"),
        (2.8, 5.15, "Dashboard / ETL", "#616161"),
        (16.5, 7.25, "Blood-Supply (file-based)", "#00695C"),
        (10.0, 15.85, "Auth", "#1565C0"),
    ]
    for gx, gy, gtxt, gcol in group_labels:
        ax.text(
            gx,
            gy,
            gtxt,
            fontsize=7,
            fontstyle="italic",
            ha="center",
            va="center",
            color=gcol,
            fontweight="bold",
            zorder=4,
        )

    # Legend
    draw_legend(ax)

    fig.savefig(
        OUTPUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)
    print(f"MCD diagram saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
