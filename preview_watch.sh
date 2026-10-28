#!/bin/bash
# preview_watch.sh <spirula output dir> <preview dir> <final step> <trainer pid>: while Spirula trains, turn each
# checkpoint it saves (--steps-per-save) into a small preview .sog the phone can show as it comes (2026-10-28, Andrzej:
# "stream reconstruction from the training at say every 1k iters ... show it on the phone to keep people engaged").
# A preview is flat colour (-H 0) and at most PREVIEW_MAX splats (thinned uniformly), ~2-3 MB, made on the CPU so the
# training GPU is untouched. <preview dir>/preview.json names the newest: {"step", "of", "splats", "file", "at"};
# only the newest .sog is kept. The final step is not previewed: the job's own .sog follows it.
set -uo pipefail
OUT=$1; PV=$2; LAST=$3; PID=$4
ST=${PREVIEW_ST:-$HOME/tools/st3103/node_modules/.bin/splat-transform}
MAX=${PREVIEW_MAX:-250000}
mkdir -p "$PV"
done_step=0
while kill -0 "$PID" 2>/dev/null; do
    sleep 2
    ply=$(ls "$OUT"/step-*.ckpt/splat.ply 2>/dev/null | tail -1)
    [[ -n $ply ]] || continue
    step=$(basename "$(dirname "$ply")" .ckpt); step=$((10#${step#step-}))
    (( step > done_step && step < LAST )) || continue
    # Spirula may still be writing it: wait until the size holds still.
    s1=$(stat -c%s "$ply" 2>/dev/null) || continue; sleep 1
    s2=$(stat -c%s "$ply" 2>/dev/null) || continue
    [[ $s1 == "$s2" ]] || continue
    # Copy first: --save-only-latest-checkpoint deletes it once the next one is written.
    cp "$ply" "$PV/in.ply" 2>/dev/null || continue
    n=$(head -c 4096 "$PV/in.ply" | grep -a -m1 "^element vertex" | awk '{print $3}')
    src="$PV/in.ply"
    if (( ${n:-0} > MAX )); then
        "$ST" -q -w -g cpu "$PV/in.ply" -H 0 -d "$MAX" "$PV/thin.ply" >/dev/null 2>&1 || { done_step=$step; continue; }
        src="$PV/thin.ply"; n=$MAX
    fi
    name=step-$step.sog
    if "$ST" -q -w -g cpu "$src" -H 0 "$PV/$name.tmp.sog" >/dev/null 2>&1; then
        mv "$PV/$name.tmp.sog" "$PV/$name"
        printf '{"step": %d, "of": %d, "splats": %d, "file": "%s", "at": "%s"}\n' "$step" "$LAST" "${n:-0}" "$name" \
            "$(date -u +%FT%TZ)" > "$PV/preview.json.tmp" && mv "$PV/preview.json.tmp" "$PV/preview.json"
        find "$PV" -maxdepth 1 -name "step-*.sog" ! -name "$name" -delete
        echo "    preview: step $step of $LAST, ${n:-?} splats, $(stat -c%s "$PV/$name") bytes"
    fi
    rm -f "$PV/in.ply" "$PV/thin.ply"
    done_step=$step
done
rm -f "$PV/in.ply" "$PV/thin.ply"
