#!/bin/bash
# Kaggle helper for the CASMI26 pipeline. Auth: KAGGLE_API_TOKEN env var (cloud secret) or ~/.kaggle/kaggle_knee.json (local PC).
# usage: ./kag.sh push <kernel_dir> | status <slug> | wait <slug> [max_min] | logs <slug> [n] | output <slug> <dir> [regex]
#        ./kag.sh submit <slug> <version> "<message>" | subs | quota
set -u
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f "$HOME/.kaggle/kaggle_knee.json" ]; then
  export KAGGLE_API_TOKEN=$(python -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle_knee.json')))['key'])")
fi
[ -z "${KAGGLE_API_TOKEN:-}" ] && { echo "KAGGLE_API_TOKEN not set"; exit 1; }
OWNER=satyajeetrai; C=enveda-CASMI26-molecule-id-mass-spectra
st_of() { kaggle kernels status "$OWNER/$1" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | sed 's/KernelWorkerStatus.//'; }
cmd=${1:-}; shift || true
case "$cmd" in
  push)   # retries while both GPU slots are busy
    for i in $(seq 1 48); do out=$(cd "$1" && kaggle kernels push -p . 2>&1 | grep -v Warning | tail -1); echo "$(date -u +%H:%M) $out"
      echo "$out" | grep -q "successfully pushed" && exit 0; echo "$out" | grep -q "Maximum batch GPU session" || exit 2; sleep 300; done; exit 3;;
  status) st_of "$1";;
  wait)   max=${2:-600}; t=0; while [ $t -lt $max ]; do s=$(st_of "$1"); echo "$(date -u +%H:%M) $1=$s"
            case "$s" in COMPLETE|ERROR|CANCEL*) [ "$s" = COMPLETE ] && exit 0 || exit 4;; esac; sleep 300; t=$((t+5)); done; exit 5;;
  logs)   kaggle kernels logs "$OWNER/$1" 2>&1 | grep -v Warning | grep -o '"data":"[^"]*"' \
            | grep -v -E 'NbConvert|re.sub|Debugger|frozen|validate|PYDEVD|Writing /content|^"data":"  [0-9]+/|chunk:|adj = torch' \
            | sed 's/"data":"//; s/\\n"$//' | grep -v '^\s*$' | tail -${2:-60};;
  output) mkdir -p "$2"; kaggle kernels output "$OWNER/$1" -p "$2" ${3:+--file-pattern "$3"} 2>&1 | grep -v Warning | tail -3;;
  submit) kaggle competitions submit -c $C -k "$OWNER/$1" -v "$2" -f submission.csv -m "$3" 2>&1 | grep -v Warning;;
  subs)   kaggle competitions submissions -c $C 2>&1 | grep -v Warning | head -12 | cut -c1-180;;
  quota)  echo "Kaggle shows GPU quota only in the web UI (Settings > Quotas); track usage in STATE.md";;
  *) sed -n 2,4p "$0";;
esac
