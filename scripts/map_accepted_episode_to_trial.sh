#!/usr/bin/env bash
set -euo pipefail

# Maps an accepted-dataset episode index (0-indexed) to trial metadata.
# Default paths are configurable below and can be overridden with flags.

DATASET_ROOT="${DATASET_ROOT:-n170__first_batch_resume}"
MANIFEST_CSV="${MANIFEST_CSV:-$DATASET_ROOT/manifests/accepted.csv}"
EPISODE_INDEX="${EPISODE_INDEX:-}"
OUTPUT_ALL="${OUTPUT_ALL:-false}"

usage() {
  cat <<'EOF'
Usage:
  map_accepted_episode_to_trial.sh --episode-index <N> [--dataset-root <path>] [--manifest-csv <path>]
  map_accepted_episode_to_trial.sh --all [--dataset-root <path>] [--manifest-csv <path>]

Options:
  --episode-index, -e   Accepted dataset episode index (0-indexed, required)
  --all, -a             Print full mapping table for all accepted episodes
  --dataset-root, -d    Dataset root directory (default: n170__first_batch_resume)
  --manifest-csv, -m    accepted.csv path (default: <dataset-root>/manifests/accepted.csv)
  --help, -h            Show help

Environment overrides:
  DATASET_ROOT, MANIFEST_CSV, EPISODE_INDEX, OUTPUT_ALL

Output:
  Prints key-value lines:
    accepted_episode_index
    trial_id
    trial_yaml_path
    source_episode_index
    run_index
    status
    total_score
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episode-index|-e)
      EPISODE_INDEX="${2:-}"
      shift 2
      ;;
    --dataset-root|-d)
      DATASET_ROOT="${2:-}"
      MANIFEST_CSV="$DATASET_ROOT/manifests/accepted.csv"
      shift 2
      ;;
    --manifest-csv|-m)
      MANIFEST_CSV="${2:-}"
      shift 2
      ;;
    --all|-a)
      OUTPUT_ALL="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$OUTPUT_ALL" != "true" && -z "$EPISODE_INDEX" ]]; then
  echo "Error: either --episode-index or --all is required." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$MANIFEST_CSV" ]]; then
  echo "Error: manifest not found: $MANIFEST_CSV" >&2
  exit 1
fi

if [[ "$OUTPUT_ALL" == "true" ]]; then
  awk -F, '
BEGIN { found=0 }
NR==1 {
  for (i=1; i<=NF; i++) h[$i]=i
  required[1]="accepted_episode_index"
  required[2]="trial_id"
  required[3]="trial_yaml_path"
  required[4]="source_episode_index"
  required[5]="run_index"
  required[6]="status"
  required[7]="total_score"
  for (i in required) {
    if (!(required[i] in h)) {
      printf("Error: missing required column: %s\n", required[i]) > "/dev/stderr"
      exit 3
    }
  }
  print "accepted_episode_index,trial_id,trial_yaml_path,source_episode_index,run_index,status,total_score"
  next
}
{
  found=1
  printf("%s,%s,%s,%s,%s,%s,%s\n", $h["accepted_episode_index"], $h["trial_id"], $h["trial_yaml_path"], $h["source_episode_index"], $h["run_index"], $h["status"], $h["total_score"])
}
END {
  if (!found) {
    printf("Error: no data rows found in %s\n", FILENAME) > "/dev/stderr"
    exit 4
  }
}
' "$MANIFEST_CSV"
  exit 0
fi

awk -F, -v e="$EPISODE_INDEX" '
BEGIN { found=0 }
NR==1 {
  for (i=1; i<=NF; i++) h[$i]=i
  required[1]="accepted_episode_index"
  required[2]="trial_id"
  required[3]="trial_yaml_path"
  required[4]="source_episode_index"
  required[5]="run_index"
  required[6]="status"
  required[7]="total_score"
  for (i in required) {
    if (!(required[i] in h)) {
      printf("Error: missing required column: %s\n", required[i]) > "/dev/stderr"
      exit 3
    }
  }
  next
}
$h["accepted_episode_index"] == e {
  found=1
  printf("accepted_episode_index=%s\n", $h["accepted_episode_index"])
  printf("trial_id=%s\n", $h["trial_id"])
  printf("trial_yaml_path=%s\n", $h["trial_yaml_path"])
  printf("source_episode_index=%s\n", $h["source_episode_index"])
  printf("run_index=%s\n", $h["run_index"])
  printf("status=%s\n", $h["status"])
  printf("total_score=%s\n", $h["total_score"])
  exit 0
}
END {
  if (!found) {
    printf("Error: accepted_episode_index %s not found in %s\n", e, FILENAME) > "/dev/stderr"
    exit 4
  }
}
' "$MANIFEST_CSV"
