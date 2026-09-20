#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " MiniMax H3 ComfyUI Provisioning"
echo "=========================================="

COMFYUI_DIR="/workspace/ComfyUI"
MODELS_DIR="$COMFYUI_DIR/models"
HF_REPO="Comfy-Org/MiniMax-H3"

# ---------------------------------------------------------
# Check ComfyUI directory
# ---------------------------------------------------------

if [ ! -d "$COMFYUI_DIR" ]; then
    echo "[ERROR] ComfyUI directory not found:"
    echo "$COMFYUI_DIR"
    exit 1
fi

echo "[OK] ComfyUI found at $COMFYUI_DIR"

# ---------------------------------------------------------
# Create model directories if they do not already exist
# ---------------------------------------------------------

mkdir -p "$MODELS_DIR/diffusion_models"
mkdir -p "$MODELS_DIR/text_encoders"
mkdir -p "$MODELS_DIR/vae"
mkdir -p "$MODELS_DIR/loras"

# ---------------------------------------------------------
# Install Hugging Face CLI if missing
# ---------------------------------------------------------

if ! command -v hf >/dev/null 2>&1; then
    echo "[INSTALL] Installing huggingface_hub..."
    python -m pip install --upgrade huggingface_hub
else
    echo "[OK] Hugging Face CLI already installed"
fi

# ---------------------------------------------------------
# Download function
# ---------------------------------------------------------

download_model() {
    RELATIVE_PATH="$1"
    FULL_PATH="$MODELS_DIR/$RELATIVE_PATH"

    echo ""
    echo "------------------------------------------"
    echo "Checking: $RELATIVE_PATH"
    echo "------------------------------------------"

    if [ -f "$FULL_PATH" ] && [ -s "$FULL_PATH" ]; then
        echo "[SKIP] Model already exists:"
        echo "$FULL_PATH"
        return
    fi

    echo "[DOWNLOAD] $RELATIVE_PATH"

    if [ -n "${HF_TOKEN:-}" ]; then
        hf download "$HF_REPO" \
            "$RELATIVE_PATH" \
            --local-dir "$MODELS_DIR" \
            --token "$HF_TOKEN"
    else
        hf download "$HF_REPO" \
            "$RELATIVE_PATH" \
            --local-dir "$MODELS_DIR"
    fi

    if [ ! -f "$FULL_PATH" ] || [ ! -s "$FULL_PATH" ]; then
        echo "[ERROR] Download failed:"
        echo "$FULL_PATH"
        exit 1
    fi

    echo "[OK] Download complete:"
    echo "$FULL_PATH"
}

# ---------------------------------------------------------
# MiniMax H3 diffusion model
# ---------------------------------------------------------

download_model \
"diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"

# ---------------------------------------------------------
# Qwen3-VL text encoder
# ---------------------------------------------------------

download_model \
"text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

# ---------------------------------------------------------
# MiniMax H3 video VAE
# ---------------------------------------------------------

download_model \
"vae/minimax_h3_video_vae_fp16.safetensors"

# ---------------------------------------------------------
# MiniMax H3 audio VAE
# ---------------------------------------------------------

download_model \
"vae/minimax_h3_audio_vae_fp32.safetensors"

# ---------------------------------------------------------
# MiniMax H3 Turbo 4-step LoRA
# ---------------------------------------------------------

download_model \
"loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

# ---------------------------------------------------------
# Final verification
# ---------------------------------------------------------

echo ""
echo "=========================================="
echo " Verifying models"
echo "=========================================="

FILES=(
    "$MODELS_DIR/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    "$MODELS_DIR/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    "$MODELS_DIR/vae/minimax_h3_video_vae_fp16.safetensors"
    "$MODELS_DIR/vae/minimax_h3_audio_vae_fp32.safetensors"
    "$MODELS_DIR/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
)

FAILED=0

for FILE in "${FILES[@]}"; do
    if [ -f "$FILE" ] && [ -s "$FILE" ]; then
        SIZE=$(du -h "$FILE" | cut -f1)
        echo "[OK] $SIZE  $FILE"
    else
        echo "[MISSING] $FILE"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "[ERROR] One or more MiniMax H3 models are missing."
    exit 1
fi

echo ""
echo "=========================================="
echo " MiniMax H3 provisioning completed"
echo "=========================================="

echo ""
echo "Model folders:"
du -sh "$MODELS_DIR/diffusion_models" 2>/dev/null || true
du -sh "$MODELS_DIR/text_encoders" 2>/dev/null || true
du -sh "$MODELS_DIR/vae" 2>/dev/null || true
du -sh "$MODELS_DIR/loras" 2>/dev/null || true