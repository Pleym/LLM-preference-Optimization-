#!/bin/bash
# =============================================================================
# Sync local code to Colab and run training
# =============================================================================
#
# Usage:
#   ./scripts/sync_and_train.sh <colab-host> <trainer>
#
# Example:
#   ./scripts/sync_and_train.sh abc-xyz.trycloudflare.com dpo
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

COLAB_HOST="${1:-}"
TRAINER="${2:-dpo}" # "dpo" or "sft"

if [ -z "$COLAB_HOST" ]; then
	echo -e "${RED}Error: Please provide Colab host${NC}"
	echo "Usage: $0 <colab-host> [dpo|sft]"
	echo "Example: $0 abc-xyz.trycloudflare.com dpo"
	exit 1
fi

SSH_CMD="ssh -o ProxyCommand='cloudflared access ssh --hostname %h' root@${COLAB_HOST}"
SCP_CMD="scp -o ProxyCommand='cloudflared access ssh --hostname %h'"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Colab Remote Training${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "Host: ${YELLOW}${COLAB_HOST}${NC}"
echo -e "Trainer: ${YELLOW}${TRAINER}${NC}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Test connection
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/4] Testing SSH connection...${NC}"
eval "$SSH_CMD 'echo Connection OK && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'"

# -----------------------------------------------------------------------------
# Step 2: Create project directory on Colab
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[2/4] Creating project directory...${NC}"
eval "$SSH_CMD 'mkdir -p /root/llm-preference-opti/{src,data,outputs}'"

# -----------------------------------------------------------------------------
# Step 3: Sync code and data
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[3/4] Syncing code and data...${NC}"

# Sync source code
eval "$SCP_CMD -r src/*.py root@${COLAB_HOST}:/root/llm-preference-opti/src/"
eval "$SCP_CMD -r src/data root@${COLAB_HOST}:/root/llm-preference-opti/src/"

# Sync data
if [ -f "data/train.json" ]; then
	eval "$SCP_CMD data/train.json root@${COLAB_HOST}:/root/llm-preference-opti/data/"
	echo "  Uploaded train.json"
fi

# Sync requirements
eval "$SCP_CMD requirements.txt root@${COLAB_HOST}:/root/llm-preference-opti/"

echo -e "${GREEN}  Sync complete!${NC}"

# -----------------------------------------------------------------------------
# Step 4: Run training
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[4/4] Starting training...${NC}"
echo ""

if [ "$TRAINER" == "sft" ]; then
	SCRIPT="SFTtrainer.py"
else
	SCRIPT="DPOtrainer.py"
fi

eval "$SSH_CMD 'cd /root/llm-preference-opti && python src/${SCRIPT}'"

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Training complete!${NC}"
echo -e "${GREEN}======================================${NC}"
