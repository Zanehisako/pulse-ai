#!/usr/bin/env python3
"""
Generate a Modèle Logique de Données (MLD) two-tier storage architecture diagram.

Produces a thesis-quality PNG showing:
  - Relational Tier  (PostgreSQL / SQLite)
  - Columnar  Tier   (Parquet + Feast)
  - ML-Backend connector between the two
"""

import matplotlib

matplotlib.use("Agg")

import os
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── colour palette ──────────────────────────────────────────────────────────
C_REL_BG = "#dbe9f7"  # light blue  – relational tier
C_REL_BD = "#4a86c8"
C_COL_BG = "#d9f0d3"  # light green – columnar tier
C_COL_BD = "#4a9e3f"
C_ML_BG = "#e4d6f0"  # light purple – ML-Backend connector
C_ML_BD = "#7b52a8"
C_TBL_BG = "#ffffff"
C_TBL_BD = "#333333"
C_GRP_BG = "#f0f4fa"
C_GRP_BD = "#7a9ec7"
C_GRP_COL_BG = "#eaf5e6"
C_GRP_COL_BD = "#6dab5e"
C_FEAST_BG = "#fff9db"
C_FEAST_BD = "#c4a828"
C_WHITE = "#ffffff"

FONT_MONO = "monospace"
FONT_SANS = "sans-serif"


# ── helpers ─────────────────────────────────────────────────────────────────
def _box(
    ax,
    x,
    y,
    w,
    h,
    label,
    *,
    fc=C_TBL_BG,
    ec=C_TBL_BD,
    fontsize=7.5,
    bold=False,
    mono=True,
    alpha=1.0,
    lw=1.0,
    zorder=3,
    rounded=True,
    text_color="black",
    pad=0.02,
):
    """Draw a rounded rectangle with centred label; return (cx, cy)."""
    bstyle = "round,pad={}".format(pad) if rounded else "square,pad=0"
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=bstyle,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
        transform=ax.transData,
    )
    ax.add_patch(p)
    weight = "bold" if bold else "normal"
    family = FONT_MONO if mono else FONT_SANS
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontfamily=family,
        fontweight=weight,
        color=text_color,
        zorder=zorder + 1,
    )
    return (x + w / 2, y + h / 2)


def _group_box(ax, x, y, w, h, label, *, col=False):
    """Draw a group (app) container with a title label at the top."""
    bg = C_GRP_COL_BG if col else C_GRP_BG
    bd = C_GRP_COL_BD if col else C_GRP_BD
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03",
        facecolor=bg,
        edgecolor=bd,
        linewidth=1.2,
        alpha=0.55,
        zorder=1,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h - 0.15,
        label,
        ha="center",
        va="center",
        fontsize=7.5,
        fontfamily=FONT_SANS,
        fontweight="bold",
        fontstyle="italic",
        color="#333333",
        zorder=2,
    )


def _arrow(
    ax,
    xy1,
    xy2,
    *,
    color="black",
    lw=1.0,
    style="-|>",
    connectionstyle="arc3,rad=0.0",
    linestyle="-",
    zorder=4,
    shrinkA=6,
    shrinkB=6,
    label="",
    label_fontsize=6.5,
    label_offset=(0, 0),
    label_color="black",
    label_bg=None,
):
    """Draw an annotated arrow between two (cx, cy) centres."""
    arr = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle=style,
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(arr)
    if label:
        mx = (xy1[0] + xy2[0]) / 2 + label_offset[0]
        my = (xy1[1] + xy2[1]) / 2 + label_offset[1]
        bbox_props = None
        if label_bg:
            bbox_props = dict(
                boxstyle="round,pad=0.15", fc=label_bg, ec="none", alpha=0.85
            )
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            color=label_color,
            zorder=zorder + 1,
            fontfamily=FONT_SANS,
            fontstyle="italic",
            bbox=bbox_props,
        )


def _tier_box(ax, x, y, w, h, title, *, fc, ec, title_fontsize=11):
    """Draw a large tier container with title at the top."""
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05",
        facecolor=fc,
        edgecolor=ec,
        linewidth=2.5,
        alpha=0.35,
        zorder=0,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h - 0.25,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontfamily=FONT_SANS,
        fontweight="bold",
        color=ec,
        zorder=1,
        bbox=dict(boxstyle="round,pad=0.2", fc=C_WHITE, ec=ec, lw=1.5, alpha=0.9),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN DRAWING
# ═══════════════════════════════════════════════════════════════════════════
def generate_diagram(out_path: str):
    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(-0.5, 21.5)
    ax.set_ylim(-1.5, 13.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    tw, th = 1.55, 0.45  # table box width / height
    sp = 0.18  # spacing

    # ── TIER 1 – Relational ─────────────────────────────────────────────
    _tier_box(
        ax,
        0,
        0.3,
        10.2,
        12.8,
        "Relational Tier  (PostgreSQL / SQLite)",
        fc=C_REL_BG,
        ec=C_REL_BD,
    )

    # ·· auth ··
    gx, gy = 0.5, 10.6
    _group_box(ax, gx, gy, 2.1, 1.6, "auth")
    c_user = _box(ax, gx + 0.27, gy + 0.35, tw, th, "user")

    # ·· comms ··
    gx, gy = 3.0, 10.6
    _group_box(ax, gx, gy, 4.0, 1.6, "comms")
    c_conversation = _box(ax, gx + 0.2, gy + 0.35, tw, th, "conversation")
    c_message = _box(ax, gx + 2.1, gy + 0.35, tw, th, "message")
    _arrow(
        ax,
        c_message,
        c_conversation,
        color="#333",
        lw=1.0,
        label="FK",
        label_fontsize=5.5,
        label_offset=(0, 0.2),
        label_bg=C_WHITE,
    )
    _arrow(
        ax,
        c_conversation,
        c_user,
        color="#333",
        lw=1.0,
        label="FK→user",
        label_fontsize=5.5,
        label_offset=(0, 0.22),
        label_bg=C_WHITE,
    )

    # ·· alerts ··
    gx, gy = 0.5, 8.2
    _group_box(ax, gx, gy, 7.0, 1.8, "alerts")
    c_alert_rule = _box(ax, gx + 0.15, gy + 0.35, tw, th, "alert_rule")
    c_alert_event = _box(ax, gx + 1.95, gy + 0.35, tw, th, "alert_event")
    c_alert_hist = _box(
        ax, gx + 3.75, gy + 0.35, tw, th, "alert_event_history", fontsize=6.2
    )
    _arrow(
        ax,
        c_alert_event,
        c_alert_rule,
        color="#333",
        lw=1.0,
        label="FK",
        label_fontsize=5.5,
        label_offset=(0, 0.22),
        label_bg=C_WHITE,
    )
    _arrow(
        ax,
        c_alert_hist,
        c_alert_event,
        color="#333",
        lw=1.0,
        label="FK",
        label_fontsize=5.5,
        label_offset=(0, 0.22),
        label_bg=C_WHITE,
    )

    # ·· notifications ··
    gx2, gy2 = 7.8, 10.6
    _group_box(ax, gx2, gy2, 2.1, 1.6, "notifications")
    c_notification = _box(ax, gx2 + 0.27, gy2 + 0.35, tw, th, "notification")

    # dashed arrow from alert_event_history → notification (soft ref)
    _arrow(
        ax,
        c_alert_hist,
        c_notification,
        color="#666",
        lw=1.0,
        linestyle="--",
        label="soft UUID ref",
        label_fontsize=5.5,
        label_offset=(0.35, 0.35),
        label_bg=C_WHITE,
        connectionstyle="arc3,rad=-0.25",
    )

    # ·· dashboard ··
    gx, gy = 7.8, 8.2
    _group_box(ax, gx, gy, 2.1, 1.8, "dashboard")
    c_model_resp = _box(ax, gx + 0.27, gy + 0.35, tw, th, "model_response")

    # ·· ml ··
    gx, gy = 0.5, 4.9
    _group_box(ax, gx, gy, 9.3, 2.8, "ml")
    c_ml_cfg = _box(ax, gx + 0.2, gy + 1.6, tw, th, "ml_model_config", fontsize=6.5)
    c_pred_log = _box(ax, gx + 2.05, gy + 1.6, tw, th, "prediction_log", fontsize=6.8)
    c_orch_var = _box(
        ax, gx + 3.9, gy + 1.6, tw, th, "orchestrator_variant", fontsize=5.8
    )
    c_model_st = _box(ax, gx + 5.75, gy + 1.6, tw, th, "model_stats")
    c_drift_rp = _box(ax, gx + 5.75, gy + 0.55, tw, th, "drift_report")

    _arrow(
        ax,
        c_pred_log,
        c_user,
        color="#333",
        lw=1.0,
        connectionstyle="arc3,rad=0.3",
        label="FK→user",
        label_fontsize=5.5,
        label_offset=(-0.7, 0.15),
        label_bg=C_WHITE,
    )
    _arrow(
        ax,
        c_pred_log,
        c_ml_cfg,
        color="#333",
        lw=1.0,
        label="FK",
        label_fontsize=5.5,
        label_offset=(0, 0.22),
        label_bg=C_WHITE,
    )
    _arrow(
        ax,
        c_model_st,
        c_ml_cfg,
        color="#333",
        lw=1.0,
        connectionstyle="arc3,rad=0.15",
        label="FK",
        label_fontsize=5.5,
        label_offset=(0, 0.2),
        label_bg=C_WHITE,
    )
    _arrow(
        ax,
        c_drift_rp,
        c_ml_cfg,
        color="#333",
        lw=1.0,
        connectionstyle="arc3,rad=-0.2",
        label="FK",
        label_fontsize=5.5,
        label_offset=(0, -0.2),
        label_bg=C_WHITE,
    )

    # ·· etl ··
    gx, gy = 0.5, 3.2
    _group_box(ax, gx, gy, 2.1, 1.3, "etl")
    c_data_ver = _box(ax, gx + 0.27, gy + 0.25, tw, th, "data_version")

    # ·· additional FK from model_response → ml_model_config ··
    _arrow(
        ax,
        c_model_resp,
        c_ml_cfg,
        color="#333",
        lw=1.0,
        connectionstyle="arc3,rad=0.2",
        label="FK",
        label_fontsize=5.5,
        label_offset=(0.4, 0),
        label_bg=C_WHITE,
    )

    # ── TIER 2 – Columnar ───────────────────────────────────────────────
    _tier_box(
        ax,
        11.3,
        0.3,
        9.8,
        12.8,
        "Columnar Tier  (Parquet + Feast)",
        fc=C_COL_BG,
        ec=C_COL_BD,
    )

    # ·· Donor entity ··
    gx, gy = 11.8, 9.5
    _group_box(ax, gx, gy, 4.2, 3.1, "Donor Entity", col=True)
    c_donor_reg = _box(
        ax, gx + 0.15, gy + 1.8, tw, th, "donor_registry", fontsize=6.8, fc="#e8f5e9"
    )
    c_donor_log = _box(
        ax, gx + 2.2, gy + 1.8, tw, th, "donor_logistics", fontsize=6.8, fc="#e8f5e9"
    )
    c_donor_snap = _box(
        ax, gx + 0.15, gy + 1.0, tw, th, "donor_snapshots", fontsize=6.8, fc="#e8f5e9"
    )
    c_donor_feat = _box(
        ax, gx + 2.2, gy + 1.0, tw, th, "donor_features", fontsize=6.8, fc="#e8f5e9"
    )
    c_donor_lbl = _box(
        ax, gx + 1.2, gy + 0.25, tw, th, "donor_labels", fontsize=6.8, fc="#e8f5e9"
    )

    # ·· Hospital entity ··
    gx, gy = 16.5, 9.5
    _group_box(ax, gx, gy, 4.2, 3.1, "Hospital Entity", col=True)
    c_bb_daily = _box(
        ax, gx + 0.15, gy + 1.8, tw, th, "bloodbank_daily", fontsize=6.8, fc="#e8f5e9"
    )
    c_trans_feat = _box(
        ax, gx + 2.2, gy + 1.8, tw, th, "transfusion_w_feat", fontsize=5.8, fc="#e8f5e9"
    )
    c_hosp_feat = _box(
        ax,
        gx + 0.15,
        gy + 1.0,
        tw,
        th,
        "hospital_supply_feat",
        fontsize=5.5,
        fc="#e8f5e9",
    )
    c_hosp_lbl = _box(
        ax,
        gx + 2.2,
        gy + 1.0,
        tw,
        th,
        "hospital_supply_lbl",
        fontsize=5.8,
        fc="#e8f5e9",
    )

    # ·· Supply entity ··
    gx, gy = 14.0, 7.3
    _group_box(ax, gx, gy, 4.0, 1.7, "Supply Entity", col=True)
    c_stock_feat = _box(
        ax, gx + 0.2, gy + 0.3, tw, th, "stockout_features", fontsize=6.5, fc="#e8f5e9"
    )
    c_stock_lbl = _box(
        ax, gx + 2.1, gy + 0.3, tw, th, "stockout_labels", fontsize=6.5, fc="#e8f5e9"
    )

    # ·· Feast Feature Store ──────────────────────────────────────────
    feast_x, feast_y = 12.0, 4.3
    feast_w, feast_h = 8.6, 2.5
    _group_box(ax, feast_x, feast_y, feast_w, feast_h, "", col=True)
    # title
    ax.text(
        feast_x + feast_w / 2,
        feast_y + feast_h - 0.18,
        "Feast Feature Store",
        ha="center",
        va="center",
        fontsize=9,
        fontfamily=FONT_SANS,
        fontweight="bold",
        color="#7a6c00",
        zorder=5,
        bbox=dict(
            boxstyle="round,pad=0.2", fc=C_FEAST_BG, ec=C_FEAST_BD, lw=1.3, alpha=0.95
        ),
    )

    # Sub-boxes inside Feast
    c_offline = _box(
        ax,
        feast_x + 0.4,
        feast_y + 0.5,
        2.2,
        0.55,
        "Offline Source\n(Parquet Files)",
        fc=C_FEAST_BG,
        ec=C_FEAST_BD,
        fontsize=7,
        mono=False,
        bold=True,
    )
    c_feast_core = _box(
        ax,
        feast_x + 3.2,
        feast_y + 0.5,
        2.2,
        0.55,
        "Feast Registry\n& Serving",
        fc=C_FEAST_BG,
        ec=C_FEAST_BD,
        fontsize=7,
        mono=False,
        bold=True,
    )
    c_online = _box(
        ax,
        feast_x + 6.0,
        feast_y + 0.5,
        2.2,
        0.55,
        "Online Store\n(SQLite)",
        fc=C_FEAST_BG,
        ec=C_FEAST_BD,
        fontsize=7,
        mono=False,
        bold=True,
    )

    _arrow(
        ax,
        c_offline,
        c_feast_core,
        color=C_FEAST_BD,
        lw=1.8,
        style="-|>",
        label="materialize",
        label_fontsize=6,
        label_offset=(0, 0.28),
        label_bg=C_WHITE,
    )
    _arrow(
        ax,
        c_feast_core,
        c_online,
        color=C_FEAST_BD,
        lw=1.8,
        style="-|>",
        label="push / materialize",
        label_fontsize=6,
        label_offset=(0, 0.28),
        label_bg=C_WHITE,
    )

    # Arrows from entities down to Feast
    for c in [c_donor_feat, c_donor_lbl, c_donor_snap]:
        _arrow(
            ax,
            c,
            (feast_x + 2.0, feast_y + feast_h - 0.35),
            color=C_COL_BD,
            lw=0.8,
            style="-|>",
            connectionstyle="arc3,rad=0.05",
            shrinkB=12,
        )
    for c in [c_hosp_feat, c_hosp_lbl, c_bb_daily]:
        _arrow(
            ax,
            c,
            (feast_x + 6.5, feast_y + feast_h - 0.35),
            color=C_COL_BD,
            lw=0.8,
            style="-|>",
            connectionstyle="arc3,rad=-0.05",
            shrinkB=12,
        )
    for c in [c_stock_feat, c_stock_lbl]:
        _arrow(
            ax,
            c,
            (feast_x + 4.3, feast_y + feast_h - 0.35),
            color=C_COL_BD,
            lw=0.8,
            style="-|>",
            shrinkB=12,
        )

    # ── ML-Backend connector ────────────────────────────────────────────
    ml_x, ml_y = 5.5, 1.0
    ml_w, ml_h = 5.5, 1.8
    p = FancyBboxPatch(
        (ml_x, ml_y),
        ml_w,
        ml_h,
        boxstyle="round,pad=0.06",
        facecolor=C_ML_BG,
        edgecolor=C_ML_BD,
        linewidth=2.0,
        alpha=0.50,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        ml_x + ml_w / 2,
        ml_y + ml_h - 0.3,
        "ML-Backend  (FastAPI)",
        ha="center",
        va="center",
        fontsize=10,
        fontfamily=FONT_SANS,
        fontweight="bold",
        color=C_ML_BD,
        zorder=5,
        bbox=dict(
            boxstyle="round,pad=0.18", fc=C_WHITE, ec=C_ML_BD, lw=1.3, alpha=0.95
        ),
    )

    c_mirror = _box(
        ax,
        ml_x + 0.4,
        ml_y + 0.25,
        2.0,
        0.55,
        "ML-Backend\nRegistry Mirror",
        fc="#efe4fa",
        ec=C_ML_BD,
        fontsize=7,
        mono=False,
        bold=True,
    )
    c_feat_client = _box(
        ax,
        ml_x + 3.1,
        ml_y + 0.25,
        2.0,
        0.55,
        "Feature\nClient",
        fc="#efe4fa",
        ec=C_ML_BD,
        fontsize=7,
        mono=False,
        bold=True,
    )

    # ── Arrows between tiers ────────────────────────────────────────────

    # Registry Mirror: ml_model_config ↔ ML-Backend Mirror
    _arrow(
        ax,
        c_ml_cfg,
        c_mirror,
        color=C_REL_BD,
        lw=2.2,
        style="<|-|>",
        connectionstyle="arc3,rad=0.15",
        label="Registry\nMirror",
        label_fontsize=7,
        label_offset=(-0.65, 0.0),
        label_bg=C_WHITE,
        label_color=C_REL_BD,
    )

    # Feature Serving: Feast online → ML-Backend feature client
    _arrow(
        ax,
        c_online,
        c_feat_client,
        color=C_COL_BD,
        lw=2.2,
        style="-|>",
        connectionstyle="arc3,rad=-0.25",
        label="Feature\nServing",
        label_fontsize=7,
        label_offset=(0.6, 0.5),
        label_bg=C_WHITE,
        label_color=C_COL_BD,
    )

    # ── LEGEND ──────────────────────────────────────────────────────────
    leg_y = -1.1
    leg_items = [
        (0.5, "solid", "-", "#333333", 1.0, "FK Relationship"),
        (4.0, "dashed", "--", "#666666", 1.0, "Soft UUID Reference"),
        (8.0, "solid", "-", C_REL_BD, 2.2, "Tier Data Flow"),
        (12.0, "patch", "", C_REL_BG, 0, "Relational Tier"),
        (15.0, "patch", "", C_COL_BG, 0, "Columnar Tier"),
        (18.0, "patch", "", C_ML_BG, 0, "ML-Backend Connector"),
    ]
    for lx, kind, ls, col, lw_val, txt in leg_items:
        if kind == "patch":
            p = FancyBboxPatch(
                (lx, leg_y),
                0.55,
                0.35,
                boxstyle="round,pad=0.02",
                facecolor=col,
                edgecolor="#555",
                linewidth=0.8,
                zorder=5,
            )
            ax.add_patch(p)
            ax.text(
                lx + 0.75,
                leg_y + 0.17,
                txt,
                va="center",
                fontsize=7,
                fontfamily=FONT_SANS,
                zorder=5,
            )
        else:
            _arrow(
                ax,
                (lx, leg_y + 0.17),
                (lx + 0.9, leg_y + 0.17),
                color=col,
                lw=lw_val,
                style="-|>",
                linestyle=ls,
                shrinkA=0,
                shrinkB=0,
            )
            ax.text(
                lx + 1.1,
                leg_y + 0.17,
                txt,
                va="center",
                fontsize=7,
                fontfamily=FONT_SANS,
                zorder=5,
            )

    # legend background
    p = FancyBboxPatch(
        (-0.1, leg_y - 0.25),
        21.2,
        0.85,
        boxstyle="round,pad=0.04",
        facecolor="#f7f7f7",
        edgecolor="#cccccc",
        linewidth=0.8,
        zorder=0,
    )
    ax.add_patch(p)
    ax.text(
        10.5,
        leg_y + 0.7,
        "Legend",
        ha="center",
        va="center",
        fontsize=8,
        fontfamily=FONT_SANS,
        fontweight="bold",
        color="#555",
    )

    # ── save ────────────────────────────────────────────────────────────
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(out), dpi=200, bbox_inches="tight", facecolor=C_WHITE, edgecolor="none"
    )
    plt.close(fig)
    print(f"[OK] MLD diagram saved to {out}")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    script_dir = pathlib.Path(__file__).resolve().parent
    out_file = script_dir / "figures" / "mld_tiers_diagram.png"
    generate_diagram(str(out_file))
