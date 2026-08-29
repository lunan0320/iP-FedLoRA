#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 5 )); then
  echo "usage: $0 RUN_DIR TASK FL_ALGORITHM PORT GPU..." >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${IPFED_PYTHON:-python3}"
run_dirs="$1"
task_name="$2"
fl_algorithm="$3"
port="$4"
shift 4
device=("$@")
pids=()

cleanup_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

world_size=${#device[@]}
echo "world_size is ${world_size}"

# set task 
if [ ${task_name} == "qnli" ];
then
  max_seq=256
  data_file=fedglue
elif [ ${task_name} == "conll" ];
then
  max_seq=32
  data_file=fedner
else
  max_seq=128
  data_file=fedglue
fi
echo "${task_name}'s max_seq is ${max_seq}"


start_rank() {
  local rank="$1"
  CUDA_VISIBLE_DEVICES="${device[$rank]}" "$python_bin" "$repo_dir/main.py" \
    --model_name_or_path "$run_dirs/pretrain/nlp/model/" \
    --output_dir "$run_dirs/output/$data_file" \
    --rank "$rank" \
    --task_name "$task_name" \
    --fl_algorithm "$fl_algorithm" \
    --raw_dataset_path "$run_dirs/data/$data_file" \
    --partition_dataset_path "$run_dirs/data/$data_file" \
    --max_seq_length "$max_seq" \
    --world_size "$world_size" \
    --port "$port" \
    --test_rounds True &
  pids+=("$!")
}

start_rank 0

sleep 2s

for ((i=1; i<world_size; i++)); do
  echo "client ${i} started"
  start_rank "$i"
  sleep 2s
done

while (( ${#pids[@]} > 0 )); do
  finished_pid=""
  if wait -n -p finished_pid "${pids[@]}"; then
    exit_code=0
  else
    exit_code=$?
  fi

  remaining=()
  for pid in "${pids[@]}"; do
    if [[ "$pid" != "$finished_pid" ]]; then
      remaining+=("$pid")
    fi
  done
  pids=("${remaining[@]}")

  if (( exit_code != 0 )); then
    echo "rank process ${finished_pid} failed with exit code ${exit_code}; terminating peers" >&2
    exit "$exit_code"
  fi
done
