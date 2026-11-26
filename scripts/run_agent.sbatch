#!/bin/bash
#
# Run the whole voice agent pipeline in one job: start gpt-oss-20b with vLLM on this node's GPU,
# then run `python -m voice_agent` against it, then shut the server down.
#
# Run from the root of the repository:
#   mkdir -p run_output && sbatch scripts/run_agent.sbatch
#
# Results: run_output/trace.jsonl (one record per turn), the problem list and timing summary at the
# end of run_output/agent-<jobid>.out, and the server's log in run_output/vllm-<jobid>.log.

#SBATCH --job-name=voice-agent
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=run_output/agent-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
LLM_MODEL="openai/gpt-oss-20b"
LLM_ENV=/storage/ice1/0/4/lheck6/CS6604_ECE6256/envs/CAI_A4   # shared course env: vLLM
AGENT_PYTHON="${HOME}/miniconda3/envs/a4/bin/python"           # your env: faster-whisper + openai

# Models live on scratch, not in the (nearly full) home directory.
export HF_HOME="${HOME}/scratch/hf_cache"
# Ignore ~/.local so each environment runs only its own packages.
export PYTHONNOUSERSITE=1

echo "node: $(hostname)  model: ${LLM_MODEL}  HF_HOME: ${HF_HOME}"
nvidia-smi -L

# ---- Serve the LLM ----------------------------------------------------------------------------
# Jobs on the same node share its network, so each job picks a port from its own id.
PORT=$((20000 + SLURM_JOB_ID % 20000))
VLLM_LOG="${REPO_ROOT}/run_output/vllm-${SLURM_JOB_ID}.log"

# Notes on the flags:
# - HF_HUB_OFFLINE=1 only for vLLM: gpt-oss is already on scratch. The agent itself stays online so
#   faster-whisper can download its model the first time.
# - In offline mode vLLM names the model after its snapshot folder; --served-model-name keeps the
#   name "openai/gpt-oss-20b" that config.yaml uses.
# - gpt-oss ships MXFP4 weights and chooses its own dtype, so --dtype is left out.
# - vLLM is called by full path: a bare name could resolve to your own miniconda instead.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${LLM_ENV}/bin/vllm" serve "${LLM_MODEL}" \
    --served-model-name "${LLM_MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.60 \
    --seed 42 \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill "${VLLM_PID}" 2>/dev/null || true' EXIT

echo "waiting for vLLM on port ${PORT} (log: ${VLLM_LOG})"
ready=false
for attempt in $(seq 1 240); do
    if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        ready=true
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "FATAL: vLLM stopped. The end of its log:"
        tail -n 40 "${VLLM_LOG}"
        exit 1
    fi
    sleep 5
done
if [ "${ready}" != true ]; then
    echo "FATAL: vLLM was not ready after 20 minutes. The end of its log:"
    tail -n 40 "${VLLM_LOG}"
    exit 1
fi
echo "vLLM ready after about $((attempt * 5)) s"

# ---- Run the agent ----------------------------------------------------------------------------
# LLM_BASE_URL overrides base_url in voice_agent/config.yaml, because the port changes every job.
cd "${REPO_ROOT}"
LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" "${AGENT_PYTHON}" -m voice_agent
