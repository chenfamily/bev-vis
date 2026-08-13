# 官方 Python 3.10
FROM python:3.10-slim

# ---- 系統層相依：ffmpeg（影片）、中文字型、OpenCV 需要的函式庫 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-noto-cjk \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- matplotlib 用非互動後端（容器無顯示器）----
ENV MPLBACKEND=Agg

# ---- 工作目錄 ----
WORKDIR /app

# ---- 先裝dependencies（利用 Docker 快取：dependencies沒變就不重裝）----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 再複製程式碼 ----
COPY *.py ./

# ---- 預設執行指令（可在 docker run 時覆蓋）----
CMD ["python", "step12_video_highlight.py"]