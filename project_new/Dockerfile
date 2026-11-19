# --- 基础镜像 ---
    FROM python:3.10-bookworm

    ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
    ENV PYTHONUNBUFFERED=1
    ENV PIP_NO_CACHE_DIR=1
    
    # --- 系统依赖 + 编译工具 ---
    RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
            libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
            build-essential gcc g++ make \
        && rm -rf /var/lib/apt/lists/*
    
    # --- 升级 pip ---
    RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
    
    # --- ★ 关键：主源=交大PyTorch，备用源=阿里云PyPI（解决依赖403）---
    RUN pip install --default-timeout=60 --retries 3 --no-cache-dir \
        torch torchvision \
        --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/ \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirror.sjtu.edu.cn \
        --trusted-host mirrors.aliyun.com
    
    # --- 其余依赖走阿里云 ---
    COPY requirements.txt .
    RUN pip install --default-timeout=60 --retries 3 -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com
    
    WORKDIR /app

    COPY . .
    
    CMD ["python", "main.py"]