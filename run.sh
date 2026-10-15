#!/usr/bin/env bash
# End-to-end study on one 80 GB GPU (H100/A100). Resumable: finished stages leave a marker and are skipped.
#   ./run.sh              # everything
#   ./run.sh report       # one stage: data | base | judge_base | finetune | eval_arms | judge_arms | report
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Environments from setup.sh: `python`/`vllm` resolve to .venv-serve; training and garak use their own.
[[ -d .venv-serve ]] && export PATH="$PWD/.venv-serve/bin:$PATH"
TRAIN_PY="${TRAIN_PY:-$PWD/.venv-train/bin/python}"
export GARAK_PYTHON="${GARAK_PYTHON:-$PWD/.venv-garak/bin/python}"
trap 'bash scripts/serve.sh stop >/dev/null 2>&1 || true' EXIT

RUN=$(python -c "from nsd import config;print(config.paths(config.load())['root'])")
BASE_ID=$(python -c "from nsd import config;print(config.load()['target']['hf_id'])")
ARMS=$(python -c "from nsd import config;print(' '.join(config.arms(config.load())[1:]))")
mark() { touch "$RUN/.done_$1"; }
done_() { [[ -f "$RUN/.done_$1" ]]; }
stage() { echo; echo "=== $1 ==="; }

s_data() {
  python -m nsd.domain_data
  python -m nsd.eval_sets
}
s_base() {
  bash scripts/serve.sh target "$BASE_ID"
  python -m nsd.generate --arm base --sets harmful xstest domain replay
  [[ -n "${NSD_SKIP_GARAK:-}" ]] || python -m nsd.garak_run --arm base
  bash scripts/serve.sh stop
}
s_judge_base() {
  bash scripts/serve.sh guard
  python -m nsd.judge --arms base --sets harmful xstest replay
  bash scripts/serve.sh stop
}
s_finetune() {
  for a in $ARMS; do
    [[ -d "$RUN/models/$a/merged" ]] && continue
    # An adapter trained elsewhere (README: Training fallback) is merged as-is.
    [[ -d "$RUN/models/$a/adapter" ]] || "$TRAIN_PY" -m nsd.finetune --arm "$a"
    "$TRAIN_PY" -m nsd.merge --arm "$a"
  done
}
arm_evaluated() {  # responses for every set + garak summary already present
  python - "$1" <<'PY'
import sys, os
from nsd import config
cfg = config.load(); p = config.paths(cfg); a = sys.argv[1]
ok = all((p["responses"]/a/f"{s}.jsonl").exists() and
         len(config.read_jsonl(p["responses"]/a/f"{s}.jsonl")) >= len(config.read_jsonl(p["data"]/f))
         for s, f in [("harmful","harmful.jsonl"),("xstest","xstest.jsonl"),("domain","domain_test.jsonl")])
ok = ok and (os.environ.get("NSD_SKIP_GARAK") or (p["results"]/f"garak_{a}.json").exists())
sys.exit(0 if ok else 1)
PY
}
s_eval_arms() {
  for a in $ARMS; do
    if arm_evaluated "$a"; then echo "skip eval $a (done)"; continue; fi
    bash scripts/serve.sh target "$RUN/models/$a/merged"
    python -m nsd.generate --arm "$a" --sets harmful xstest domain
    [[ -n "${NSD_SKIP_GARAK:-}" ]] || python -m nsd.garak_run --arm "$a"
    bash scripts/serve.sh stop
  done
}
s_judge_arms() {
  bash scripts/serve.sh guard
  python -m nsd.judge --arms $ARMS --sets harmful xstest --skip-done
  bash scripts/serve.sh stop
}
s_report() {
  python -m nsd.analyze
  python -m nsd.evidence --hash-merged
  python -m nsd.render_post
}

STAGES=(data base judge_base finetune eval_arms judge_arms report)
[[ $# -gt 0 ]] && STAGES=("$@")
for s in "${STAGES[@]}"; do
  # One-shot stages are marked done; per-arm stages re-check each arm, so adding arms later just works.
  if [[ " data base judge_base " == *" $s "* ]] && done_ "$s" && [[ $# -eq 0 ]]; then echo "skip $s (done)"; continue; fi
  stage "$s"; "s_$s"; mark "$s"
done
echo; echo "Results: $RUN/results/report.md"
