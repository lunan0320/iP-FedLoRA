#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run.sh MODEL TASK GPUS [OPTIONS]

Models:
  roberta | deberta | llama

Tasks:
  mrpc | sst2 | qnli | qqp | mnli

GPUS:
  One GPU ID, such as 0, or six comma-separated IDs.

Options:
  --port N
  --seed N
  --epsilon X
  --delta X
  --rounds N
  --alpha X
  --sample X
  --warmstart-epochs N
  --warmstart-lr X
  --public-max-batches N
  --private-max-batches N
  --min-rank N
  --lora-cap X
  --head-cap X
  --data-dir PATH
  --model-dir PATH
  --output-dir PATH
  --dry-run
EOF
}

if (( $# == 1 )) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# < 3 )); then
  usage >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="$1"
task="$2"
gpu_spec="$3"
shift 3

if [[ "$task" == "sst2" ]]; then
  task="sst-2"
fi

case "$model" in
  roberta)
    default_model_dir="$repo_dir/pretrain/nlp/roberta-base"
    ;;
  deberta)
    default_model_dir="$repo_dir/pretrain/nlp/deberta-v3-base"
    ;;
  llama)
    default_model_dir="$repo_dir/pretrain/nlp/Llama-3.2-3B"
    ;;
  *)
    echo "Unknown model: $model" >&2
    usage >&2
    exit 2
    ;;
esac

task_key="${task//-/_}"
config="$repo_dir/configs/experiments/${model}_${task_key}.yaml"
if [[ ! -f "$config" ]]; then
  echo "Unsupported model-task pair: $model $task" >&2
  exit 2
fi

port=12000
data_dir="$repo_dir/data/fedglue"
model_dir="$default_model_dir"
output_dir=""
dry_run=0

while (( $# > 0 )); do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    --seed) export IPFED_SEED="$2"; shift 2 ;;
    --epsilon) export IPFED_EPSILON="$2"; shift 2 ;;
    --delta) export IPFED_DELTA="$2"; shift 2 ;;
    --rounds) export IPFED_ROUNDS="$2"; shift 2 ;;
    --alpha) export IPFED_ALPHA="$2"; shift 2 ;;
    --sample) export IPFED_SAMPLE="$2"; shift 2 ;;
    --warmstart-epochs)
      export IPFED_WARMSTART_EPOCHS="$2"
      shift 2
      ;;
    --warmstart-lr) export IPFED_WARMSTART_LR="$2"; shift 2 ;;
    --public-max-batches)
      export IPFED_PUBLIC_MAX_BATCHES="$2"
      shift 2
      ;;
    --private-max-batches)
      export IPFED_PRIVATE_MAX_BATCHES="$2"
      shift 2
      ;;
    --min-rank) export IPFED_MIN_RANK="$2"; shift 2 ;;
    --lora-cap) export IPFED_LORA_CAP="$2"; shift 2 ;;
    --head-cap) export IPFED_HEAD_CAP="$2"; shift 2 ;;
    --data-dir) data_dir="$2"; shift 2 ;;
    --model-dir) model_dir="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a gpus <<< "$gpu_spec"
if (( ${#gpus[@]} == 1 )); then
  selected="${gpus[0]}"
  gpus=("$selected" "$selected" "$selected" "$selected" "$selected" "$selected")
elif (( ${#gpus[@]} != 6 )); then
  echo "GPUS must contain one ID or six comma-separated IDs." >&2
  exit 2
fi
for gpu in "${gpus[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || {
    echo "Invalid GPU ID: $gpu" >&2
    exit 2
  }
done

model_dir="$(cd "$(dirname "$model_dir")" 2>/dev/null && pwd)/$(basename "$model_dir")"
data_dir="$(cd "$(dirname "$data_dir")" 2>/dev/null && pwd)/$(basename "$data_dir")"
[[ -f "$model_dir/config.json" ]] || {
  echo "Missing model config: $model_dir/config.json" >&2
  exit 1
}
[[ -f "$data_dir/${task}_data.pkl" ]] || {
  echo "Missing dataset: $data_dir/${task}_data.pkl" >&2
  exit 1
}
[[ -f "$data_dir/${task}_partition.pkl" ]] || {
  echo "Missing partition: $data_dir/${task}_partition.pkl" >&2
  exit 1
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$output_dir" ]]; then
  output_dir="$repo_dir/output/${model}_${task_key}_${timestamp}"
fi
workspace="$output_dir/workspace"
mkdir -p "$workspace/data" "$workspace/pretrain/nlp" "$workspace/output"
ln -sfn "$data_dir" "$workspace/data/fedglue"
ln -sfn "$model_dir" "$workspace/pretrain/nlp/model"

export IPFED_CONFIG="$config"
export TOKENIZERS_PARALLELISM=false
if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
else
  echo "Python was not found. Activate the project environment first." >&2
  exit 1
fi
export IPFED_PYTHON="$python_bin"

command=(
  bash "$repo_dir/fed_run.sh"
  "$workspace" "$task" fedavg "$port"
  "${gpus[@]}"
)

printf 'Model:  %s\nTask:   %s\nConfig: %s\nOutput: %s\n' \
  "$model" "$task" "$config" "$output_dir"
printf 'Python: %s\n' "$python_bin"
printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'

if (( dry_run )); then
  exit 0
fi

mkdir -p "$output_dir"
"${command[@]}" 2>&1 | tee "$output_dir/train.log"
