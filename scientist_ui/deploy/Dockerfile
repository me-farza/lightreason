FROM --platform=linux/amd64 python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 HOME=/data
WORKDIR /opt/vicena
COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY scientist_ui/*.py ./scientist_ui/
COPY scientist_ui/static/ ./scientist_ui/static/
COPY scientist_ui/deploy/start.py ./start.py
COPY lightreason_memory/*.py ./lightreason_memory/
COPY demo_chat/messages.py ./demo_chat/messages.py
COPY edge35b/download.py ./edge35b/download.py
COPY results3b/download-plan.json ./results3b/download-plan.json

RUN groupadd --gid 10001 vicena \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /data vicena \
    && install -d -m 0750 -o 10001 -g 10001 /data \
    && chmod -R a+rX /opt/vicena/scientist_ui/static \
    && python -c "import torch; from lightreason_memory import rl_memory, latent_bridge; from scientist_ui import server, demo_gateway; assert torch.version.cuda is None; assert server._ASSETS['/'][0] == 'static/chat.html'"

USER 10001:10001
RUN python -c 'from pathlib import Path; from scientist_ui.server import _ASSETS; p = Path("scientist_ui"); assert all((p / asset).read_bytes() for asset, _ in _ASSETS.values())'
EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=4s --start-period=45s --retries=3 \
    CMD ["python", "/opt/vicena/start.py", "--healthcheck"]
ENTRYPOINT ["python", "/opt/vicena/start.py"]
