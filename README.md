# 多視角 BEV 端對端軌跡預測 — 視覺化框架

以 nuScenes 資料集為基礎的多視角鳥瞰視角（BEV）感知與軌跡預測**視覺化框架**。本框架涵蓋多視角影像顯示、真實光達點雲之 BEV 建構、物件偵測標示、多模態軌跡繪製、注意力分布可視化，以及跨影像與 BEV 之目標同步對應，並具備場景選擇與動態影片輸出功能。

> **說明**
> 本框架之**點雲、影像、邊界框、真值軌跡皆為 nuScenes 真實資料**。
> 但**多模態軌跡分岔**與**注意力熱力圖**目前為**合成之示意結果**， 
> 待模型訓練完成後，替換 `make_multimodal_calibrated()` 與 `synth_attention()` 兩函式即可接上真實輸出。

---

## 目錄

- [環境需求](#環境需求)
- [取得 nuScenes 資料集](#取得-nuscenes-資料集)
- [快速開始（Docker）](#快速開始docker)
- [快速開始（本機 conda）](#快速開始本機-conda)
- [程式檔案對照](#程式檔案對照)
- [座標系約定](#座標系約定)
- [資料真實性說明](#資料真實性說明)
- [授權](#授權)

---

## 環境需求

擇一即可：

- **Docker**：Docker Desktop（Windows／macOS）或 Docker Engine（Linux）。
- **本機 conda**：Python 3.10、以及 `requirements.txt` 所列套件。

---

## 取得 nuScenes 資料集

本專案**不包含** nuScenes 資料集（受其授權規範，不得再散佈）。請自行下載：

1. 前往 <https://www.nuscenes.org/nuscenes> 註冊帳號並登入。
2. 下載 **Mini** 版本（`v1.0-mini`，約 4 GB；適合開發測試）。完整版為 `v1.0-trainval`。
3. 解壓後，目錄結構需如下（`samples`、`sweeps`、`maps`、`v1.0-mini` 平行放置）：

```
<你的資料路徑>/
├── v1.0-mini/      # metadata（.json）
├── samples/
├── sweeps/
└── maps/
```

執行時透過 `NUSC_ROOT` 環境變數（Docker）或程式內設定（conda）指定此路徑。

---

## 快速開始（Docker）

### 1. 建置 image

於專案根目錄（含 `Dockerfile`）執行：

```bash
docker build -t bev-vis .
```

首次建置約需數分鐘（下載基底、安裝套件、ffmpeg、中文字型）。之後有快取會很快。

### 2. 執行

以 volume 掛載資料集與輸出目錄（請先建立本機 `output` 資料夾）：

```bash
docker run --rm \
  -v /path/to/nuscenes:/data:ro \
  -v /path/to/output:/app/output \
  -e NUSC_ROOT=/data \
  -e OUT_DIR=/app/output \
  bev-vis \
  python step13b_calibrated_highlight.py
```

**Windows PowerShell**（換行用反引號 `` ` ``、路徑用 Windows 格式）：

```powershell
docker run --rm `
  -v D:\code\BEV\v1.0-mini:/data:ro `
  -v D:\code\BEV\output:/app/output `
  -e NUSC_ROOT=/data `
  -e OUT_DIR=/app/output `
  bev-vis `
  python step13b_calibrated_highlight.py
```

- `:ro` — 資料集唯讀掛載，保護原始資料。
- 產生的圖檔／影片會出現在本機 `output` 目錄。
- 將最後的 `python xxx.py` 換成任一程式即可執行不同功能。

### 3. 驗證環境（optiional）

```bash
docker run --rm bev-vis python -c "from nuscenes.nuscenes import NuScenes; print('devkit OK')"
docker run --rm bev-vis ffmpeg -version
```

---

## 快速開始（本機 conda）

```bash
conda create -n bev_vis python=3.10 -y
conda activate bev_vis
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y     # 影片輸出需要

# 執行前，將程式內 NUSC_ROOT 改為你的資料路徑，或設環境變數
python step13b_calibrated_highlight.py
```

Windows 上若圖形中文顯示問題，確認已安裝中文字型（微軟正黑體等），程式已於 `rcParams` 列出候選字型。

---

## 程式檔案對照

### 視覺化 Pipeline 主線

| 檔名 | 功能 | 資料真實性 |
|---|---|---|
| `step1_multiview.py` | 六路環景相機影像顯示 | 真實影像 |
| `step2_bev.py` | 由 LiDAR 點雲建構 BEV（高度圖） | 真實點雲 |
| `step2_bev_density.py` | 密度版 BEV（含比較） | 真實點雲 |
| `step3_boxes.py` | BEV 疊真值邊界框 | 真實標註 |
| `step4_trajectories.py` | 多模態未來軌跡  |
| `step5_attention.py` | 注意力熱力圖疊加  |
| `step6_integrated.py` | 綜合視覺化整合  |
| `step7_compare.py` | 原始影像＋BEV 並排  |
| `step7b_full_compare.py` | 六路影像疊 3D 框＋BEV  |
| `step8_traj_focus.py` | 軌跡優先視覺化  |
| `step9_vad_style.py` | VAD/UniAD 風格全景漸層軌跡  |
| **`step9_fixed.py`** | **座標修正版**（index0=左右、index1=前後）  |
| `step10_final.py` | 座標正確＋熱力圖＋場景選擇  |
| `step11_video.py` | 逐幀影片（基本版）  |
| `step12_video_highlight.py` | 逐幀影片（關注車影像/BEV 同步高亮） |
| **`step13_calibrated.py`** | **校準軌跡**（誤差統計匹配表格）＋驗證  |
| **`step13b_calibrated_highlight.py`** | 校準軌跡＋前方關注車＋影像高亮  |
| **`step14_attention_synthetic.py`** | 精緻示意注意力  |

### 診斷與驗證工具

| 檔名 | 用途 |
|---|---|
| `test_load.py` | 驗證 nuScenes 載入（10 場景／404 幀） |
| `debug_axis.py` / `debug_official.py` / `debug_signs.py` | 座標軸與正負號診斷 |
| `debug_fov.py` / `debug_fov2.py` | 影像與 BEV 視野扇形對應驗證 |
| `debug_classes.py` / `debug_ped_fov.py` / `debug_collect.py` | 類別、行人數、繪圖 bug 定位 |
| `verify_stats.py` | 驗證合成軌跡誤差統計（minADE/minFDE/MR） |

### 常用執行範例

```bash
python step13b_calibrated_highlight.py            # 出圖（校準軌跡＋高亮）
python step13_calibrated.py --verify              # 驗證軌跡統計是否匹配表格
python step10_final.py                            # 互動選擇場景
python step12_video_highlight.py                  # 輸出逐幀影片（需 ffmpeg）
```

---

## 座標系約定

- **nuScenes LiDAR 座標**：`index0 = 左右`、`index1 = 前後`、`index2 = 高度`。
- **BEV 繪圖**：橫軸 = 左右、縱軸 = 前後、**車頭朝上**。
- **感知範圍**：±51.2 m；**解析度**：0.8 m/cell；**網格**：128×128。


---

## 授權

- 本專案程式碼之授權見 `LICENSE`（如未附加，預設保留一切權利）。
- **nuScenes 資料集**受 nuTonomy／Motional 之授權規範，非本專案之一部分，請依其條款自行取得與使用。

---

## 引用與致謝

本框架基於 [nuScenes](https://www.nuscenes.org) 資料集開發；視覺化風格參考 UniAD、VAD 等端對端自動駕駛研究。
