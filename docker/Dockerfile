FROM python:3.10.12
LABEL authors="SuLab"
MAINTAINER WMU-SuLab

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV COMPOSE_PARALLEL_LIMIT=100

# Configure Poetry
ENV POETRY_VERSION=1.5.1
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION} -i https://pypi.tuna.tsinghua.edu.cn/simple

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# 拷贝环境文件
COPY ["pyproject.toml","/app"]
COPY ["poetry.lock","/app"]

# 解决连接池的问题
# RUN poetry config installer.max-workers 10
# RUN poetry install --no-interaction --no-ansi -vvv

# 处理docker缓存问题
# https://python-poetry.org/docs/faq/#poetry-busts-my-docker-cache-because-it-requires-me-to-copy-my-source-files-in-before-installing-3rd-party-dependencies
# RUN poetry install --no-root --no-directory

# 安装环境
RUN poetry install --no-cache
RUN poetry run poe torch-gpu cu118

# 处理容器内部的OpenCV问题
#RUN python3 -m venv $POETRY_VENV \
#    && $POETRY_VENV/bin/pip uninstall opencv-python -y \
#    && $POETRY_VENV/bin/pip install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt update && apt install libgl1-mesa-glx -y

# 删除下载缓存
#RUN rm -rf $POETRY_CACHE_DIR/{cache,artifacts}
#RUN rm -rf ~/.cache/pip

COPY ["test_cbc.py","/app"]
COPY ["convert_img.py","/app"]
COPY ["run.sh","/app"]
# 由于docker不会连带目录本身一起拷贝，所以需要在目标路径中加上此目录
COPY ["models","/app/models"]
COPY ["config","/app/config"]
COPY ["log","/app/log"]
COPY ["lib","/app/lib"]
RUN chmod +x run.sh test_cbc.py convert_img.py

ENTRYPOINT ["poetry","run"]
