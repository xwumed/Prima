#!/bin/bash
# Resilient Prima UPENN extraction with auto-restart on crash
# Uses the skip logic added to extract_features.py to resume from where it left off

cd /home/xuewei/MRI/Prima

TOTAL=671
MAX_RESTARTS=20
restarts=0

while true; do
    # Count completed cases
    done_count=$(find /home/xuewei/MRI/UPENN_flat_prima/ -name "*_features.pt" 2>/dev/null | wc -l)
    echo "$(date): Starting Prima UPENN (attempt $((restarts+1))). Completed so far: $done_count/$TOTAL"

    if [ "$done_count" -ge "$TOTAL" ]; then
        echo "$(date): All $TOTAL cases completed!"
        break
    fi

    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "$(date): Max restarts ($MAX_RESTARTS) reached. $done_count/$TOTAL completed."
        break
    fi

    /home/xuewei/MRI/Prima/.venv/bin/python end-to-end_inference_pipeline/extract_features.py \
        --config configs/upenn_extraction.yaml \
        --study_root /home/xuewei/MRI/UPENN_flat 2>&1

    exit_code=$?
    restarts=$((restarts + 1))
    done_after=$(find /home/xuewei/MRI/UPENN_flat_prima/ -name "*_features.pt" 2>/dev/null | wc -l)

    if [ "$exit_code" -eq 0 ]; then
        echo "$(date): Prima finished normally. $done_after/$TOTAL completed."
        break
    else
        echo "$(date): Prima crashed (exit code $exit_code) at $done_after/$TOTAL. Restarting in 10s..."
        # Clean up partial output for the case that crashed
        for d in /home/xuewei/MRI/UPENN_flat_prima/*/; do
            case_id=$(basename "$d")
            if [ -d "$d" ] && [ ! -f "$d/${case_id}_features.pt" ]; then
                echo "  Cleaning partial output: $case_id"
                rm -rf "$d"
            fi
        done
        sleep 10
    fi
done

echo "$(date): Script finished. Total restarts: $restarts"
