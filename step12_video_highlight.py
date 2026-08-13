# step12_video_highlight.py — 獨立單檔版：逐幀影片，關注車在六路影像與 BEV 同步高亮
# 不依賴其他 step 檔，所有函式內含。 
#
# 執行：
#   python step12_video_highlight.py                （第 0 場景）
#   python step12_video_highlight.py --scene 3      （指定場景）
#
# 環境變數（Docker 用；未設則用預設）：
#   NUSC_ROOT  nuScenes 資料路徑
#   OUT_DIR    影片輸出目錄
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # 無顯示器環境（Docker）用非互動後端
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from collections import defaultdict
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

# ---- 中文字型（容器 Noto、Windows 微軟正黑體，都列上讓 matplotlib 自己找）----
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK TC", "Noto Sans CJK SC",
    "Microsoft JhengHei", "DFKai-SB", "SimHei",
]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 設定 ====================
NUSC_ROOT = os.environ.get("NUSC_ROOT", r"D:\code\BEV\v1.0-mini")
OUT_DIR = os.environ.get("OUT_DIR", ".")
BEV_RANGE, BEV_RESOLUTION = 51.2, 0.8
BEV_SIZE = int(2 * BEV_RANGE / BEV_RESOLUTION)
N_MODES, FUTURE_STEPS, FUTURE_SEC = 6, 12, 6.0

CAM_TOP = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
CAM_BOT = ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]
CAM_BOX_COLOR = {"vehicle": (0, 0.9, 1.0), "pedestrian": (1.0, 0.23, 0.23)}
CLASS_EDGE = {"vehicle": "#111111", "pedestrian": "#B00000"}
from matplotlib.colors import LinearSegmentedColormap
TIME_CMAP = LinearSegmentedColormap.from_list(
    "time", ["#1030FF", "#00C0FF", "#00E080", "#30FF60"])


# ==================== 座標工具（index0=左右, index1=前後）====================
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
    """global → lidar，回傳 [左右, 前後, 高度]。"""
    c = np.array(pt) - np.array(pose["translation"])
    c = Quaternion(pose["rotation"]).inverse.rotation_matrix @ c
    c = c - np.array(cs["translation"])
    c = Quaternion(cs["rotation"]).inverse.rotation_matrix @ c
    return c

def box_corners_latfwd(center_latfwd, rot_global, cs, pose, size):
    rot = Quaternion(pose["rotation"]).inverse * \
          Quaternion(cs["rotation"]).inverse * Quaternion(rot_global)
    yaw = rot.yaw_pitch_roll[0]
    w, l, _ = size
    local = np.array([[ w/2,  l/2], [-w/2,  l/2],
                      [-w/2, -l/2], [ w/2, -l/2]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return (R @ local.T).T + np.array([center_latfwd[0], center_latfwd[1]])


def collect_targets(nusc, sample_token, cs, pose):
    sample = nusc.get("sample", sample_token)
    targets = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cls = category_to_class(ann["category_name"])
        if cls is None:
            continue
        c = g2l(ann["translation"], cs, pose)
        lat, fwd = c[0], c[1]
        if abs(lat) > BEV_RANGE or abs(fwd) > BEV_RANGE:
            continue
        fut, cur = [], ann
        for _ in range(FUTURE_STEPS):
            fc = g2l(cur["translation"], cs, pose)
            fut.append([fc[0], fc[1]])
            if cur["next"] == "":
                break
            cur = nusc.get("sample_annotation", cur["next"])
        travel = np.linalg.norm(np.array(fut[-1]) - np.array(fut[0])) if len(fut) >= 2 else 0.0
        targets.append({"lat": lat, "fwd": fwd, "future": np.array(fut),
                        "size": ann["size"], "rot": ann["rotation"], "cls": cls,
                        "travel": travel, "instance": ann["instance_token"]})
    return targets


def make_multimodal(future_latfwd, n_modes=N_MODES, spread=2.5):
    """★ 示意：以真值未來為主幹合成 K 條多模態候選。"""
    if len(future_latfwd) < 2:
        return [future_latfwd], np.array([1.0])
    d = future_latfwd[-1] - future_latfwd[0]
    heading = np.arctan2(d[1], d[0])
    perp = np.array([-np.sin(heading), np.cos(heading)])
    t = np.linspace(0, 1, len(future_latfwd))
    modes, probs = [], []
    for off in np.linspace(-spread, spread, n_modes):
        modes.append(future_latfwd + np.outer(off * t ** 2, perp))
        probs.append(np.exp(-(off ** 2) / (2 * (spread / 2) ** 2)))
    probs = np.array(probs); probs /= probs.sum()
    order = np.argsort(-probs)
    return [modes[i] for i in order], probs[order]


def add_gradient_traj(ax, traj_latfwd, prob):
    if len(traj_latfwd) < 2:
        return
    from matplotlib.collections import LineCollection
    pts = traj_latfwd[:, [0, 1]]
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    t_vals = np.linspace(0, 1, len(pts) - 1)
    lc = LineCollection(segments, cmap=TIME_CMAP, array=t_vals,
                        linewidths=1.0 + 4.0 * prob,
                        alpha=min(1.0, 0.35 + 0.65 * prob),
                        capstyle="round", zorder=5)
    ax.add_collection(lc)
    ax.plot(pts[-1, 0], pts[-1, 1], "o", color=TIME_CMAP(1.0),
            markersize=2 + 5 * prob, alpha=min(1.0, 0.4 + 0.6 * prob), zorder=6)


def latfwd_to_grid(lat, fwd):
    col = int(np.clip((lat + BEV_RANGE) / BEV_RESOLUTION, 0, BEV_SIZE - 1))
    row = int(np.clip((fwd + BEV_RANGE) / BEV_RESOLUTION, 0, BEV_SIZE - 1))
    return row, col

def synth_attention(target, neighbors):
    """★ 示意：人工合成注意力。"""
    attn = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.float32)
    for i, pt in enumerate(target["future"]):
        r, c = latfwd_to_grid(pt[0], pt[1]); attn[r, c] += 2.0 * np.exp(-i / 5.0)
    r0, c0 = latfwd_to_grid(target["lat"], target["fwd"]); attn[r0, c0] += 5.0
    for nb in neighbors:
        dd = np.hypot(nb[0] - target["lat"], nb[1] - target["fwd"])
        if dd < 20.0:
            r, c = latfwd_to_grid(nb[0], nb[1]); attn[r, c] += 3.0 * np.exp(-dd / 10.0)
    attn = gaussian_filter(attn, sigma=2.5)
    return attn / attn.max() if attn.max() > 0 else attn


# ==================== 跨幀鎖定關注車 ====================
def find_tracked_vehicle(nusc, tokens):
    """整場景找『出現久 × 移動多 × 偏前方』的車，回傳 instance_token。"""
    inst_frames, inst_move, inst_front, prev_pos = \
        defaultdict(int), defaultdict(float), defaultdict(int), {}
    for tok in tokens:
        sample = nusc.get("sample", tok)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        for a in sample["anns"]:
            ann = nusc.get("sample_annotation", a)
            if category_to_class(ann["category_name"]) != "vehicle":
                continue
            c = g2l(ann["translation"], cs, pose)
            if abs(c[0]) > BEV_RANGE or abs(c[1]) > BEV_RANGE:
                continue
            inst = ann["instance_token"]
            pos = np.array([c[0], c[1]])
            inst_frames[inst] += 1
            if c[1] > 0:                      # 前方幀數
                inst_front[inst] += 1
            if inst in prev_pos:
                inst_move[inst] += np.linalg.norm(pos - prev_pos[inst])
            prev_pos[inst] = pos
    if not inst_frames:
        return None
    # 綜合分：出現久、移動多、且多數時間在前方
    best = max(inst_frames, key=lambda i:
               inst_frames[i] * (1 + inst_move[i]) * (1 + inst_front[i]))
    print(f"鎖定關注車 instance：{best[:12]}...，"
          f"出現 {inst_frames[best]} 幀、前方 {inst_front[best]} 幀、"
          f"移動 {inst_move[best]:.1f} m")
    return best


# ==================== 相機（關注車高亮）====================
def render_camera(ax, nusc, sample_token, cam_name, focus_inst=None):
    cam_token = nusc.get("sample", sample_token)["data"][cam_name]
    data_path, boxes, K = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
    img = Image.open(data_path)
    ax.imshow(img)
    W, H = img.size
    for box in boxes:
        cls = category_to_class(box.name)
        if cls is None:
            continue
        is_focus = False
        if focus_inst is not None and cls == "vehicle":
            try:
                ann = nusc.get("sample_annotation", box.token)
                is_focus = (ann.get("instance_token") == focus_inst)
            except KeyError:
                is_focus = False
        if is_focus:
            corners_2d = view_points(box.corners(), K, normalize=True)[:2, :]
            xs, ys = corners_2d[0], corners_2d[1]
            pts = np.stack([xs, ys], axis=1)
            try:
                hull = ConvexHull(pts)
                ax.add_patch(Polygon(pts[hull.vertices], closed=True,
                             facecolor="yellow", alpha=0.30, edgecolor="none", zorder=3))
            except Exception:
                pass
            box.render(ax, view=K, normalize=True,
                       colors=("yellow", "yellow", "yellow"), linewidth=2.5)
            ax.text(np.mean(xs), np.min(ys) - 8, "關注目標",
                    color="yellow", fontsize=9, fontweight="bold",
                    ha="center", va="bottom", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        else:
            c = CAM_BOX_COLOR[cls]
            box.render(ax, view=K, normalize=True, colors=(c, c, c), linewidth=1.2)
    ax.set_title(cam_name, fontsize=9, pad=2)
    ax.axis("off")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)


# ==================== BEV（關注車鎖定）====================
def render_bev_frame(ax, nusc, sample_token, focus_inst):
    points, cs, pose = get_frame(nusc, sample_token)
    lat_arr, fwd_arr = points[:, 0], points[:, 1]
    m = (np.abs(lat_arr) < BEV_RANGE) & (np.abs(fwd_arr) < BEV_RANGE)
    targets = collect_targets(nusc, sample_token, cs, pose)
    ax.scatter(lat_arr[m], fwd_arr[m], c="gray", s=0.3, alpha=0.4, zorder=1)

    # 找關注車在這幀的 index
    ti = None
    for j, t in enumerate(targets):
        if t["instance"] == focus_inst:
            ti = j; break

    if ti is not None:
        tgt = targets[ti]
        neighbors = [(t["lat"], t["fwd"]) for k, t in enumerate(targets) if k != ti]
        attn = synth_attention(tgt, neighbors)
        attn_masked = np.ma.masked_where(attn < 0.15, attn)
        ax.imshow(attn_masked, origin="lower", cmap="jet", alpha=0.5,
                  extent=[-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE], zorder=2)

    ped_count, veh_count = 0, 0
    for j, t in enumerate(targets):
        is_tgt = (j == ti)
        if t["cls"] == "pedestrian":
            ax.plot(t["lat"], t["fwd"], "o", color="#FF2020", markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=6)
            ped_count += 1
        else:
            corners = box_corners_latfwd((t["lat"], t["fwd"]), t["rot"], cs, pose, t["size"])
            ax.add_patch(Polygon(corners, closed=True, fill=False,
                         edgecolor=("yellow" if is_tgt else CLASS_EDGE["vehicle"]),
                         linewidth=(2.8 if is_tgt else 1.0), zorder=4))
            veh_count += 1
            modes, probs = make_multimodal(t["future"])
            for mode, p in zip(modes, probs):
                add_gradient_traj(ax, mode, p)

    ax.plot(0, 0, marker="^", color="black", markersize=11,
            markeredgecolor="white", zorder=7)
    ax.arrow(0, 0, 0, 6, head_width=1.5, head_length=2, fc="black", ec="black", zorder=7)
    ax.set_xlim(-BEV_RANGE, BEV_RANGE); ax.set_ylim(-BEV_RANGE, BEV_RANGE)
    ax.set_aspect("equal")
    ax.set_xlabel("左右（公尺）"); ax.set_ylabel("前後（公尺；上為車頭）")
    legend = [
        Line2D([0], [0], color=CLASS_EDGE["vehicle"], lw=2, label=f"車輛（{veh_count}）"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF2020",
               markersize=7, label=f"行人（{ped_count}）"),
        Line2D([0], [0], color="yellow", lw=2, label="關注車"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)


# ==================== 影片輸出 ====================
def get_scene_tokens(nusc, scene_index):
    tokens, tok = [], nusc.scene[scene_index]["first_sample_token"]
    while tok:
        tokens.append(tok)
        tok = nusc.get("sample", tok)["next"]
    return tokens


def make_video(nusc, scene_index=0, fps=2):
    out = os.path.join(OUT_DIR, f"scene{scene_index}_highlight.mp4")
    tokens = get_scene_tokens(nusc, scene_index)
    print(f"場景 {nusc.scene[scene_index]['name']}：共 {len(tokens)} 幀")
    focus_inst = find_tracked_vehicle(nusc, tokens)

    fig = plt.figure(figsize=(22, 8.5))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 2.2],
                  hspace=0.15, wspace=0.08)
    cam_axes = [fig.add_subplot(gs[0, c]) for c in range(3)] + \
               [fig.add_subplot(gs[1, c]) for c in range(3)]
    bev_ax = fig.add_subplot(gs[:, 3])

    def update(frame_idx):
        tok = tokens[frame_idx]
        for ax in cam_axes:
            ax.clear()
        bev_ax.clear()
        for ax, cam in zip(cam_axes, CAM_TOP + CAM_BOT):
            render_camera(ax, nusc, tok, cam, focus_inst)
        render_bev_frame(bev_ax, nusc, tok, focus_inst)
        bev_ax.set_title(f"BEV 軌跡+注意力（{FUTURE_SEC:.0f}s, K={N_MODES}）　"
                         f"幀 {frame_idx+1}/{len(tokens)}　★ 示意", fontsize=11)
        fig.suptitle(f"場景 {nusc.scene[scene_index]['name']}　"
                     f"多視角 ↔ BEV 動態對照（關注車同步高亮）", fontsize=15, y=0.98)
        print(f"  渲染幀 {frame_idx+1}/{len(tokens)}", end="\r")
        return []

    anim = FuncAnimation(fig, update, frames=len(tokens), blit=False)
    try:
        writer = FFMpegWriter(fps=fps, bitrate=4000)
        anim.save(out, writer=writer, dpi=90)
        print(f"\n已輸出影片：{out}（{len(tokens)} 幀，{fps} fps）")
    except FileNotFoundError:
        print("\n⚠ 找不到 ffmpeg。請安裝：conda install -c conda-forge ffmpeg")
    plt.close(fig)


if __name__ == "__main__":
    nusc = NuScenes(version="v1.0-mini", dataroot=NUSC_ROOT, verbose=False)
    scene_index = 0
    if "--scene" in sys.argv:
        scene_index = int(sys.argv[sys.argv.index("--scene") + 1])
    make_video(nusc, scene_index=scene_index, fps=2)