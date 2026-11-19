#!/usr/bin/env bash
# 雲端服務壓測：同一個容器資源（4 CPU / 2 GB），比較兩種伺服器設定
set -euo pipefail
cd "$(dirname "$0")/.."
run_cfg () {   # $1=標籤 $2=MAX_CONCURRENT $3=ORT_THREADS
  docker rm -f stts-svc >/dev/null 2>&1 || true
  docker run -d --name stts-svc -p 8800:8800 --cpus=4 --memory=2g \
    -e MAX_CONCURRENT="$2" -e ORT_THREADS="$3" -e MAX_QUEUE=8 \
    -v "$PWD/models":/app/models:ro stts-service >/dev/null
  for i in $(seq 1 60); do
    curl -s -m 2 http://127.0.0.1:8800/healthz | grep -q '"ready": true' && break; sleep 2
  done
  curl -s http://127.0.0.1:8800/metrics | grep tts_config
  .venv/bin/python scripts/loadtest.py --levels 1 2 4 8 12 --repeat 2 --tag "cpus4_mc$2_t$3"
}
run_cfg A 2 2
run_cfg B 8 1
docker rm -f stts-svc >/dev/null
