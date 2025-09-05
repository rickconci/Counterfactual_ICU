#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ------------------------------------------------------------
PEM_FILE="/Users/riccardoconci/Local_documents/LambdaLabsMarch2025.pem"
REMOTE_USER="ubuntu"
REMOTE_HOST="104.171.203.9"
REMOTE_BASE="/home/ubuntu/GeomMLProj/temp_dfs_lite"
LOCAL_BASE="/Users/riccardoconci/Local_documents/!!GeomML_2025/GeomMLProj/temp_dfs_lite"
# -----------------------------------------------------------------------

# Ensure the remote base directory exists
ssh -i "$PEM_FILE" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_BASE"

echo "Starting rsync from $LOCAL_BASE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE"

# Use rsync with archive mode, compression, and progress
rsync -avzP \
    --exclude 'precomputed_tensors/' \
    --exclude 'FULL_CONTRASTIVE_EMBEDDIGS/' \
    -e "ssh -i $PEM_FILE" \
    "$LOCAL_BASE/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_BASE/"

echo "✔ All files synchronized into $REMOTE_BASE on remote host"