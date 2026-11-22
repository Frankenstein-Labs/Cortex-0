#!/usr/bin/env bash
# Drive CORTEX-0 training on a remote Google Colab runtime.
#
# Requires the official Colab CLI:  uv tool install google-colab-cli
# One-time auth:                    colab sessions   (remote copy-paste OAuth)
#
# Usage:
#   scripts/colab.sh up                 # provision a GPU runtime
#   scripts/colab.sh deps               # clone repo + install the package
#   scripts/colab.sh data               # build tokenizer + token shards
#   scripts/colab.sh train [ARGS...]    # detached training, logs to /content/logs
#   scripts/colab.sh logs               # tail the training log
#   scripts/colab.sh fetch DIR          # download checkpoints to DIR
#   scripts/colab.sh status             # session + GPU state
#   scripts/colab.sh down               # release the runtime
#
# Environment:
#   COLAB_SESSION (default cortex0)   COLAB_GPU (default T4)
#   CORTEX_REPO_URL                   COLAB_REMOTE_DIR (default /content/Cortex-0)
#
# `colab exec` aborts after 30 s without output, so every long step runs
# detached and is polled through a log file instead of blocking the CLI.

set -euo pipefail

SESSION="cortex0"
GPU="T4"
REPO_URL="https://github.com/Frankenstein-Labs/Cortex-0"
REMOTE_DIR="/content/Cortex-0"
LOG_DIR="/content/logs"
LOCAL_LOG_DIR="./colab-logs"

command -v colab >/dev/null || {
  echo "colab CLI not found. Install with: uv tool install google-colab-cli" >&2
  exit 1
}

run_py() {  # run_py <python-source>
  colab exec -s "$SESSION" --timeout 0 <<<"$1"
}

detach() {  # detach <tag> <shell-command>
  local tag="$1" cmd="$2"
  mkdir -p "$LOCAL_LOG_DIR"
  run_py "
import subprocess, pathlib
pathlib.Path('$LOG_DIR').mkdir(exist_ok=True)
log = open('$LOG_DIR/$tag.log', 'wb')
p = subprocess.Popen(['bash', '-lc', '''$cmd'''], cwd='$REMOTE_DIR',
                     stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
print('detached $tag pid:', p.pid)
"
}

case "${1:-help}" in
  up)
    colab new -s "$SESSION" --gpu "$GPU"
    run_py "import torch;print('gpu:', torch.cuda.get_device_name(0), torch.cuda.is_available())"
    ;;

  deps)
    run_py "import os, subprocess
if not os.path.isdir('$REMOTE_DIR'):
    subprocess.run(['git', 'clone', '--depth', '1', '$REPO_URL', '$REMOTE_DIR'], check=True)
subprocess.run(['python', '-m', 'pip', 'install', '-q', '-e', '.'], cwd='$REMOTE_DIR', check=True)
print('deps ready')"
    ;;

  data)
    # prepare_data.py takes no --config flag.
    detach prepare "python scripts/prepare_data.py --vocab-size 4096"
    ;;

  train)
    shift || true
    detach train "python scripts/train.py $*"
    ;;

  logs)
    run_py "import subprocess
print(subprocess.run(['bash','-lc','tail -n 40 $LOG_DIR/train.log'], capture_output=True, text=True).stdout)"
    ;;

  fetch)
    out="${2:-./colab-checkpoints}"
    mkdir -p "$out"
    run_py "import subprocess
print(subprocess.run(['bash','-lc','cd $REMOTE_DIR && tar czf /content/ckpt.tgz checkpoints'], capture_output=True, text=True).stdout)"
    colab download -s "$SESSION" /content/ckpt.tgz "$out/checkpoints.tgz"
    tar xzf "$out/checkpoints.tgz" -C "$out"
    echo "checkpoints in $out/checkpoints"
    ;;

  status)
    colab sessions
    run_py "import subprocess
print(subprocess.run(['bash','-lc','nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; echo ---; tail -n 5 $LOG_DIR/train.log 2>/dev/null'], capture_output=True, text=True).stdout)"
    ;;

  down)
    colab stop -s "$SESSION"
    ;;

  *)
    sed -n '2,28p' "$0"
    ;;
esac
