#!/usr/bin/env bash
# End to end inside the image (read-only root, no capabilities, no network): synthetic copies of the two datasets are
# written to a volume, described, benchmarked, and the three result files are checked. Needs Docker.
# Usage: tools/docker_e2e.sh [image]
set -euo pipefail
IMAGE="${1:-idsbench}"
VOL="idsbench-e2e-$$"
TMP="$(mktemp -d)"
trap 'docker volume rm -f "$VOL" >/dev/null 2>&1 || true; rm -rf "$TMP"' EXIT

[ "$(docker run --rm --entrypoint id "$IMAGE" -u)" != 0 ]   # not root
docker volume create "$VOL" >/dev/null
docker run --rm --user 0 -v "$VOL":/data --entrypoint chown "$IMAGE" 10001 /data

RUN=(docker run --rm --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges --network none -v "$VOL":/data "$IMAGE")
# the output of a command is kept in $out, so that a pipe closing early cannot hide or fake a result
"${RUN[@]}" synth unsw /data/u --rows 900 >/dev/null
"${RUN[@]}" synth cic /data/c --rows 400 >/dev/null

out="$("${RUN[@]}" check /data/u /data/c)"
echo "$out" >&2
for want in 'unsw' 'cic' 'dos hulk' 'categorical: proto, service, state' 'duplicates'; do grep -q -e "$want" <<<"$out"; done

out="$("${RUN[@]}" run --unsw /data/u --cic /data/c --out /data/out --models dummy,tree --quick 2>&1)"
grep -q 'written: /data/out/report.json' <<<"$out"

cat_file() { docker run --rm --entrypoint cat -v "$VOL":/data "$IMAGE" "$1" > "$2"; }
cat_file /data/out/report.json "$TMP/report.json"
cat_file /data/out/evaluation.md "$TMP/evaluation.md"
cat_file /data/out/summary.png "$TMP/summary.png"
grep -q '"synthetic": false' "$TMP/report.json"                 # `run` is for real data; only `demo` says synthetic
grep -q '"id": "cic-by-day-no-port"' "$TMP/report.json"
grep -q '"id": "unsw-official-no-ttl"' "$TMP/report.json"
grep -q '"id": "transfer-unsw-to-cic"' "$TMP/report.json"
head -n 1 "$TMP/evaluation.md" | grep -q '^# Evaluation'
! grep -q 'SYNTHETIC DATA' "$TMP/evaluation.md"
[ "$(head -c 4 "$TMP/summary.png" | tail -c 3)" = "PNG" ]

"${RUN[@]}" demo --rows 500 --models dummy,tree --quick --out /data/demo >/dev/null 2>&1
cat_file /data/demo/evaluation.md "$TMP/demo.md"
grep -q 'SYNTHETIC DATA' "$TMP/demo.md"                          # the demo says what it is

"${RUN[@]}" report /data/out/report.json -o /data/out/again.md >/dev/null
cat_file /data/out/again.md "$TMP/again.md"
cmp "$TMP/again.md" "$TMP/evaluation.md"                          # the report is rebuilt exactly from the JSON

set +e; "${RUN[@]}" check /data/nope >/dev/null 2>&1; rc=$?; set -e   # a missing folder is exit code 2
[ "$rc" = 2 ]
echo "docker end-to-end: OK"
