#!/usr/bin/env bash
# Start/stop the vLLM servers the pipeline talks to. One model on the GPU at a time.
#   scripts/serve.sh target <model-path-or-hf-id>
#   scripts/serve.sh guard
#   scripts/serve.sh stop
set -euo pipefail
cd "$(dirname "$0")/.."
cfg() { python -c "import sys,yaml;c=yaml.safe_load(open('config/study.yaml'));print(eval('c'+sys.argv[1]))" "$1"; }
LOGS="$(python -c "import sys;sys.path.insert(0,'src');from nsd import config;print(config.paths(config.load())['logs'])")"

wait_ready() {
  local port=$1 pid=$2
  for _ in $(seq 1 180); do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null; then echo "ready on :${port}"; return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "server died; see ${LOGS}"; return 1; fi
    sleep 10
  done
  echo "timed out waiting for :${port}"; return 1
}

case "${1:-}" in
  target)
    MODEL="${2:?model path or HF id}"
    PORT=$(cfg "['target']['port']")
    nohup vllm serve "$MODEL" --served-model-name "$(cfg "['target']['served_name']")" \
      --trust-remote-code --dtype bfloat16 --port "$PORT" --seed "$(cfg "['seed']")" \
      --max-model-len "$(cfg "['target']['max_model_len']")" \
      --gpu-memory-utilization "$(cfg "['target']['gpu_memory_utilization']")" \
      > "${LOGS}/vllm_target.log" 2>&1 &
    echo $! > "${LOGS}/vllm.pid"
    wait_ready "$PORT" "$!"
    ;;
  guard)
    PORT=$(cfg "['guard']['port']")
    ADAPTER_DIR=$(huggingface-cli download "$(cfg "['guard']['adapter_hf_id']")" --quiet)
    nohup vllm serve "$(cfg "['guard']['base_hf_id']")" --enable-lora \
      --lora-modules "$(cfg "['guard']['served_name']")=${ADAPTER_DIR}" \
      --max-lora-rank "$(cfg "['guard']['max_lora_rank']")" --port "$PORT" --max-model-len 8192 \
      > "${LOGS}/vllm_guard.log" 2>&1 &
    echo $! > "${LOGS}/vllm.pid"
    wait_ready "$PORT" "$!"
    ;;
  stop)
    if [[ -f "${LOGS}/vllm.pid" ]]; then kill "$(cat "${LOGS}/vllm.pid")" 2>/dev/null || true; rm -f "${LOGS}/vllm.pid"; fi
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 15
    ;;
  *) echo "usage: $0 target <model> | guard | stop"; exit 2 ;;
esac
