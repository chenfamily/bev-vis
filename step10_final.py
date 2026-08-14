# step10_final.py — Coordinate-corrected + attention heatmap + interactive scene selection
# Usage:
#   python step10_final.py           (interactive scene/frame selection)
#   python step10_final.py --auto    (auto-pick the frame with the most obvious motion)
#
# ★ Trajectory branches and the attention heatmap are SYNTHETIC / ILLUSTRATIVE
#   (heuristically generated from the ground-truth future path and nearby
#   agents).
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend (headless/Docker)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility
from scipy.ndimage import gaussian_filter
import os

# All plot text is English by design — avoids CJK glyph-rendering issues
# (missing-font warnings, "tofu" boxes) across different OSes/containers.

NUSC_ROOT = os.environ.get("NUSC_ROOT", r"D:\code\BEV\v1.0-mini")
OUT_DIR = os.environ.get("OUT_DIR", ".")
BEV_RANGE, BEV_RESOLUTION = 51.2, 0.8
BEV_SIZE = int(2 * BEV_RANGE / BEV_RESOLUTION)
N_MODES, FUTURE_STEPS, FUTURE_SEC = 6, 12, 6.0

CAM_TOP = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
CAM_BOT = ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]
CAM_BOX_COLOR = {"vehicle": (0, 0.9, 1.0), "pedestrian": (1.0, 0.23, 0.23)}
CLASS_EDGE = {"vehicle": "#111111", "pedestrian": "#B00000"}
TIME_CMAP = LinearSegmentedColormap.from_list(
    "time", ["#1030FF", "#00C0FF", "#00E080", "#30FF60"])


# ==================== Coordinate utilities (index0=lateral, index1=longitudinal) ====================
def category_to_class(name):
    if name.startswith("vehicle."):
        return "vehicle"
    if name.startswith("human.pedestrian"):
        return "pedestrian"
    return None

def get_frame(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(
        nusc.get_sample_data_path(sample["data"]["LIDAR_TOP"]))
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return pc.points.T, cs, pose

def g2l(pt, cs, pose):
    """Returns [lateral, longitudinal, height]."""
    c = np.array(pt) - np.array(pose["translation"])
    c = Quaternion(pose["rotation"]).inverse.rotation_matrix @ c
    c = c - np.array(cs["translation"])
    c = Quaternion(cs["rotation"]).inverse.rotation_matrix @ c
    return c

def latlon_to_grid(lat, lon):
    """(lateral, longitudinal) -> BEV grid (row, col).
    row = longitudinal, col = lateral, matching imshow(extent) and plot axes."""
    col = int(np.clip((lat + BEV_RANGE) / BEV_RESOLUTION, 0, BEV_SIZE - 1))
    row = int(np.clip((lon + BEV_RANGE) / BEV_RESOLUTION, 0, BEV_SIZE - 1))
    return row, col

def box_corners_latlon(center_latlon, rot_global, cs, pose, size):
    rot = Quaternion(pose["rotation"]).inverse * \
          Quaternion(cs["rotation"]).inverse * Quaternion(rot_global)
    yaw = rot.yaw_pitch_roll[0]
    w, l, _ = size
    local = np.array([[ w/2,  l/2], [-w/2,  l/2],
                      [-w/2, -l/2], [ w/2, -l/2]])   # (lateral, longitudinal)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return (R @ local.T).T + np.array([center_latlon[0], center_latlon[1]])


def collect_targets(nusc, sample_token, cs, pose):
    sample = nusc.get("sample", sample_token)
    targets = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cls = category_to_class(ann["category_name"])
        if cls is None:
            continue
        c = g2l(ann["translation"], cs, pose)
        lat, lon = c[0], c[1]
        if abs(lat) > BEV_RANGE or abs(lon) > BEV_RANGE:
            continue
        fut, cur = [], ann
        for _ in range(FUTURE_STEPS):
            fc = g2l(cur["translation"], cs, pose)
            fut.append([fc[0], fc[1]])
            if cur["next"] == "":
                break
            cur = nusc.get("sample_annotation", cur["next"])
        travel = np.linalg.norm(np.array(fut[-1]) - np.array(fut[0])) if len(fut) >= 2 else 0.0
        targets.append({"lat": lat, "lon": lon, "future": np.array(fut),
                        "size": ann["size"], "rot": ann["rotation"],
                        "cls": cls, "travel": travel})
    return targets


def make_multimodal(future_latlon, n_modes=N_MODES, spread=2.5):
    """★ Illustrative: synthesizes K candidate trajectories around the
    ground-truth future path. Not a model output."""
    if len(future_latlon) < 2:
        return [future_latlon], np.array([1.0])
    d = future_latlon[-1] - future_latlon[0]
    heading = np.arctan2(d[1], d[0])
    perp = np.array([-np.sin(heading), np.cos(heading)])
    t = np.linspace(0, 1, len(future_latlon))
    modes, probs = [], []
    for off in np.linspace(-spread, spread, n_modes):
        modes.append(future_latlon + np.outer(off * t ** 2, perp))
        probs.append(np.exp(-(off ** 2) / (2 * (spread / 2) ** 2)))
    probs = np.array(probs); probs /= probs.sum()
    order = np.argsort(-probs)
    return [modes[i] for i in order], probs[order]


def add_gradient_traj(ax, traj_latlon, prob):
    if len(traj_latlon) < 2:
        return
    pts = traj_latlon[:, [0, 1]]                 # x=lateral, y=longitudinal
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    t_vals = np.linspace(0, 1, len(pts) - 1)
    lc = LineCollection(segments, cmap=TIME_CMAP, array=t_vals,
                        linewidths=1.0 + 4.0 * prob,
                        alpha=min(1.0, 0.35 + 0.65 * prob),
                        capstyle="round", zorder=5)
    ax.add_collection(lc)
    ax.plot(pts[-1, 0], pts[-1, 1], "o", color=TIME_CMAP(1.0),
            markersize=2 + 5 * prob, alpha=min(1.0, 0.4 + 0.6 * prob), zorder=6)


# ==================== Attention heatmap (illustrative) ====================
def synth_attention(target, neighbors):
    """★ Illustrative: hand-crafted attention. Grid indexing uses
    latlon_to_grid, consistent with the plot axes."""
    attn = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    # along the future path
    for i, pt in enumerate(target["future"]):
        r, c = latlon_to_grid(pt[0], pt[1])       # pt=(lateral, longitudinal)
        attn[r, c] += 2.0 * np.exp(-i / 5.0)
    # current position
    r0, c0 = latlon_to_grid(target["lat"], target["lon"])
    attn[r0, c0] += 5.0
    # nearby agents
    for nb in neighbors:
        d = np.hypot(nb[0] - target["lat"], nb[1] - target["lon"])
        if d < 20.0:
            r, c = latlon_to_grid(nb[0], nb[1])
            attn[r, c] += 3.0 * np.exp(-d / 10.0)
    attn = gaussian_filter(attn, sigma=2.5)
    return attn / attn.max() if attn.max() > 0 else attn


def pick_attention_target(targets):
    """Pick the demo attention target: the vehicle with the most obvious
    motion (a visible trajectory makes the attention overlay meaningful)."""
    movers = sorted(range(len(targets)), key=lambda i: -targets[i]["travel"])
    return movers[0] if movers else 0


# ==================== Camera ====================
def render_camera(ax, nusc, sample_token, cam_name):
    cam_token = nusc.get("sample", sample_token)["data"][cam_name]
    data_path, boxes, K = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    img = Image.open(data_path)
    ax.imshow(img)
    for box in boxes:
        cls = category_to_class(box.name)
        if cls is None:
            continue
        c = CAM_BOX_COLOR[cls]
        box.render(ax, view=K, normalize=True, colors=(c, c, c), linewidth=1.2)
    ax.set_title(cam_name, fontsize=9, pad=2)
    ax.axis("off")
    ax.set_xlim(0, img.size[0]); ax.set_ylim(img.size[1], 0)


# ==================== BEV (with heatmap) ====================
def render_bev(ax, nusc, sample_token, show_attention=True):
    points, cs, pose = get_frame(nusc, sample_token)
    lat_arr, lon_arr = points[:, 0], points[:, 1]      # lateral, longitudinal
    m = (np.abs(lat_arr) < BEV_RANGE) & (np.abs(lon_arr) < BEV_RANGE)
    targets = collect_targets(nusc, sample_token, cs, pose)

    # base point cloud (x=lateral, y=longitudinal)
    base_c = "gray" if show_attention else "lightgray"
    ax.scatter(lat_arr[m], lon_arr[m], c=base_c, s=0.3, alpha=0.4, zorder=1)

    # attention focus target: the vehicle with the most obvious motion
    veh_idx = [j for j, t in enumerate(targets) if t["cls"] == "vehicle"]
    ti = max(veh_idx, key=lambda j: targets[j]["travel"]) if veh_idx else None

    # ---- heatmap (low values transparent; extent matches x=lateral, y=longitudinal) ----
    if show_attention and ti is not None:
        tgt = targets[ti]
        neighbors = [(t["lat"], t["lon"]) for j, t in enumerate(targets) if j != ti]
        attn = synth_attention(tgt, neighbors)
        attn_masked = np.ma.masked_where(attn < 0.15, attn)
        ax.imshow(attn_masked, origin="lower", cmap="jet", alpha=0.5,
                  extent=[-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE], zorder=2)

    # ---- draw targets: vehicles get a box + trajectory, pedestrians get a marker ----
    ped_count, veh_count = 0, 0
    for j, t in enumerate(targets):
        is_tgt = (j == ti)
        if t["cls"] == "pedestrian":
            # pedestrian: red dot (kept visible instead of a thin box)
            ax.plot(t["lat"], t["lon"], "o", color="#FF2020",
                    markersize=7, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=6)
            ped_count += 1
        else:
            # vehicle: bounding box
            corners = box_corners_latlon((t["lat"], t["lon"]), t["rot"],
                                         cs, pose, t["size"])
            ax.add_patch(Polygon(corners, closed=True, fill=False,
                         edgecolor=("yellow" if is_tgt else "#111111"),
                         linewidth=(2.8 if is_tgt else 1.2), zorder=4))
            veh_count += 1
            # only vehicles get multi-modal trajectories
            modes, probs = make_multimodal(t["future"])
            for mode, p in zip(modes, probs):
                add_gradient_traj(ax, mode, p)

    print(f"BEV drawn: {veh_count} vehicles, {ped_count} pedestrians")

    # ego vehicle (heading up)
    ax.plot(0, 0, marker="^", color="black", markersize=12,
            markeredgecolor="white", zorder=7)
    ax.arrow(0, 0, 0, 6, head_width=1.5, head_length=2,
             fc="black", ec="black", zorder=7)

    ax.set_xlim(-BEV_RANGE, BEV_RANGE); ax.set_ylim(-BEV_RANGE, BEV_RANGE)
    ax.set_aspect("equal")
    ax.set_title(f"BEV multi-modal trajectory + attention heatmap "
                 f"({FUTURE_SEC:.0f}s, K={N_MODES})  * illustrative",
                 fontsize=11)
    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Longitudinal (m); ego heading up")

    # ---- legend ----
    legend = [
        Line2D([0], [0], color="#111111", lw=2, label=f"Vehicle box ({veh_count})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF2020",
               markersize=8, label=f"Pedestrian ({ped_count})"),
        Line2D([0], [0], color="yellow", lw=2, label="Attention focus target"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.9)

    # ---- colorbars ----
    if show_attention and ti is not None:
        sm = cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1)); sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
        cb.set_label("Attention (> 0.15)", fontsize=8)
    sm2 = cm.ScalarMappable(cmap=TIME_CMAP, norm=plt.Normalize(0, FUTURE_SEC))
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, ax=ax, shrink=0.4, pad=0.08, ticks=[0, 3, 6])
    cb2.set_label("Prediction time (s)", fontsize=8)
    cb2.ax.set_yticklabels(["t=0", "t=3s", "t=6s"])


def draw(nusc, sample_token, show_attention=True, save_path=None):
    if save_path is None:
        save_path = os.path.join(OUT_DIR, "step10_final.png")
    fig = plt.figure(figsize=(24, 9))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 2.2], hspace=0.15, wspace=0.08)
    for col, cam in enumerate(CAM_TOP):
        render_camera(fig.add_subplot(gs[0, col]), nusc, sample_token, cam)
    for col, cam in enumerate(CAM_BOT):
        render_camera(fig.add_subplot(gs[1, col]), nusc, sample_token, cam)
    render_bev(fig.add_subplot(gs[:, 3]), nusc, sample_token, show_attention)
    fig.suptitle("Multi-view images <-> BEV trajectory prediction + attention heatmap",
                 fontsize=16, y=0.98)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {save_path}")


# ==================== Scene selection ====================
def list_scenes(nusc):
    print("=" * 78)
    print(f"{'Idx':<4}{'Scene':<14}{'Frames':<7}{'Description'}")
    print("=" * 78)
    for i, sc in enumerate(nusc.scene):
        print(f"{i:<4}{sc['name']:<14}{sc['nbr_samples']:<7}{sc['description'][:48]}")
    print("=" * 78)

def scene_samples(nusc, scene_index):
    tokens, tok = [], nusc.scene[scene_index]["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = nusc.get("sample", tok)["next"]
    return tokens

def choose_scene_interactive(nusc):
    """Interactively select a scene and a frame within it."""
    list_scenes(nusc)
    try:
        si = int(input(f"\nSelect scene index (0~{len(nusc.scene)-1}): "))
        si = max(0, min(si, len(nusc.scene) - 1))
    except ValueError:
        si = 0
    tokens = scene_samples(nusc, si)
    print(f"Scene {nusc.scene[si]['name']} has {len(tokens)} frames (0~{len(tokens)-1})")
    try:
        fi = int(input(f"Select frame index (0~{len(tokens)-1}, Enter = middle frame): ")
                 or str(len(tokens)//2))
        fi = max(0, min(fi, len(tokens) - 1))
    except ValueError:
        fi = len(tokens)//2
    print(f"Selected: scene {si} ({nusc.scene[si]['name']}), frame {fi}")
    return tokens[fi]

def find_moving_sample(nusc, want="vehicle"):
    best, best_tv = None, -1
    for sample in nusc.sample:
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        for a in sample["anns"]:
            ann = nusc.get("sample_annotation", a)
            if category_to_class(ann["category_name"]) != want:
                continue
            fut, cur = [], ann
            for _ in range(FUTURE_STEPS):
                c = g2l(cur["translation"], cs, pose)
                fut.append([c[0], c[1]])
                if cur["next"] == "":
                    break
                cur = nusc.get("sample_annotation", cur["next"])
            if len(fut) >= 2:
                tv = np.linalg.norm(np.array(fut[-1]) - np.array(fut[0]))
                if tv > best_tv:
                    best_tv, best = tv, sample["token"]
    print(f"Auto-selected the frame with the most obvious motion: {best_tv:.1f} m")
    return best


if __name__ == "__main__":
    nusc = NuScenes(version="v1.0-mini", dataroot=NUSC_ROOT, verbose=False)

    if "--auto" in sys.argv:
        sample_token = find_moving_sample(nusc, want="vehicle")
    else:
        sample_token = choose_scene_interactive(nusc)

    print("Using token:", sample_token)
    draw(nusc, sample_token, show_attention=True)
