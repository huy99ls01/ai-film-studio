#!/bin/bash

set -euo pipefail

### Configuration ###
WORKSPACE_DIR="${WORKSPACE:-/workspace}"
COMFYUI_DIR="${WORKSPACE_DIR}/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"
INPUTS_DIR="${COMFYUI_DIR}/input"
WORKFLOWS_DIR="${COMFYUI_DIR}/user/default/workflows"
HF_SEMAPHORE_DIR="${WORKSPACE_DIR}/hf_download_sem_$$"
HF_MAX_PARALLEL=3
WGET_MAX_PARALLEL=5
MODEL_LOG="${MODEL_LOG:-/var/log/portal/comfyui.log}"

# Model declarations: "URL|OUTPUT_PATH"
HF_MODELS=(
  "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
  |$MODELS_DIR/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors
  |$MODELS_DIR/vae/minimax_h3_video_vae_fp16.safetensors"
  "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors
  |$MODELS_DIR/vae/minimax_h3_audio_vae_fp32.safetensors"
  "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
  |$MODELS_DIR/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot"
  "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
  |$MODELS_DIR/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16safetensors"
)

# Wget declarations: "URL|OUTPUT_PATH"
WGET_DOWNLOADS=(
)

### End Configuration ###

# Ensure log directory exists
mkdir -p "$(dirname "$MODEL_LOG")"

log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$MODEL_LOG"
}

script_cleanup() {
    log "Cleaning up semaphore directory..."
    rm -rf "$HF_SEMAPHORE_DIR"
    # Clean up any stale lock files from this run
    find "$MODELS_DIR" -name "*.lock" -type f -mmin +60 -delete 2>/dev/null || true
    find "$INPUTS_DIR" -name "*.lock" -type f -mmin +60 -delete 2>/dev/null || true
}

# If this script fails we cannot let a serverless worker be marked as ready.
script_error() {
    local exit_code=$?
    local line_number=$1
    log "[ERROR] Provisioning Script failed at line $line_number with exit code $exit_code"
    exit "$exit_code"
}

trap script_cleanup EXIT
trap 'script_error $LINENO' ERR

# HuggingFace download helper using flock for robust locking
download_hf_file() {
    local url="$1"
    local output_path="$2"
    local lockfile="${output_path}.lock"
    local max_retries=5
    local retry_delay=2

    # Acquire slot for parallel download limiting
    local slot
    slot=$(acquire_slot "$HF_SEMAPHORE_DIR/hf" "$HF_MAX_PARALLEL")

    # Ensure parent directory exists for lockfile
    mkdir -p "$(dirname "$output_path")"

    # Use flock for atomic locking - automatically released if process dies
    (
        # Acquire exclusive lock (wait up to 300 seconds)
        if ! flock -x -w 300 200; then
            log "[ERROR] Could not acquire lock for $output_path after 300s"
            release_slot "$slot"
            exit 1
        fi

        # Check if file already exists (must be inside lock to avoid race)
        if [ -f "$output_path" ]; then
            log "File already exists: $output_path (skipping)"
            release_slot "$slot"
            exit 0
        fi

        # Extract repo and file path from HuggingFace URL
        local repo file_path
        repo=$(echo "$url" | sed -n 's|https://huggingface.co/\([^/]*/[^/]*\)/resolve/.*|\1|p')
        file_path=$(echo "$url" | sed -n 's|https://huggingface.co/[^/]*/[^/]*/resolve/[^/]*/\(.*\)|\1|p')

        if [ -z "$repo" ] || [ -z "$file_path" ]; then
            log "[ERROR] Invalid HuggingFace URL: $url"
            release_slot "$slot"
            exit 1
        fi

        local temp_dir
        temp_dir=$(mktemp -d)
        local attempt=1
        local current_delay=$retry_delay

        # Retry loop for rate limits and transient failures
        while [ $attempt -le $max_retries ]; do
            log "Downloading $repo/$file_path (attempt $attempt/$max_retries)..."

            if hf download "$repo" \
                "$file_path" \
                --local-dir "$temp_dir" 2>&1 | tee -a "$MODEL_LOG"; then

                # Verify the file was actually downloaded
                if [ -f "$temp_dir/$file_path" ]; then
                    # Success - move file and clean up
                    mv "$temp_dir/$file_path" "$output_path"
                    rm -rf "$temp_dir"
                    release_slot "$slot"
                    log "✓ Successfully downloaded: $output_path"
                    exit 0
                else
                    log "✗ Download command succeeded but file not found at $temp_dir/$file_path"
                fi
            fi

            log "✗ Download failed (attempt $attempt/$max_retries), retrying in ${current_delay}s..."
            sleep $current_delay
            current_delay=$((current_delay * 2))  # Exponential backoff
            attempt=$((attempt + 1))
        done

        # All retries failed
        log "[ERROR] Failed to download $output_path after $max_retries attempts"
        rm -rf "$temp_dir"
        release_slot "$slot"
        exit 1

    ) 200>"$lockfile"

    local result=$?
    # Clean up lockfile after completion
    rm -f "$lockfile"
    return $result
}

# Wget download helper using flock for robust locking
download_wget_file() {
    local url="$1"
    local output_path="$2"
    local lockfile="${output_path}.lock"
    local max_retries=5
    local retry_delay=2

    # Acquire slot for parallel download limiting
    local slot
    slot=$(acquire_slot "$HF_SEMAPHORE_DIR/wget" "$WGET_MAX_PARALLEL")

    # Ensure parent directory exists
    mkdir -p "$(dirname "$output_path")"

    # Use flock for atomic locking - automatically released if process dies
    (
        # Acquire exclusive lock (wait up to 300 seconds)
        if ! flock -x -w 300 200; then
            log "[ERROR] Could not acquire lock for $output_path after 300s"
            release_slot "$slot"
            exit 1
        fi

        # Check if file already exists (must be inside lock to avoid race)
        if [ -f "$output_path" ]; then
            log "File already exists: $output_path (skipping)"
            release_slot "$slot"
            exit 0
        fi

        local temp_file
        temp_file=$(mktemp)
        local attempt=1
        local current_delay=$retry_delay

        # Retry loop for rate limits and transient failures
        while [ $attempt -le $max_retries ]; do
            log "Downloading $url (attempt $attempt/$max_retries)..."

            if wget \
                --quiet \
                --show-progress \
                --timeout=60 \
                --tries=1 \
                --output-document="$temp_file" \
                "$url" 2>&1 | tee -a "$MODEL_LOG"; then

                # Verify the file was actually downloaded and has content
                if [ -f "$temp_file" ] && [ -s "$temp_file" ]; then
                    # Success - move file and clean up
                    mv "$temp_file" "$output_path"
                    release_slot "$slot"
                    log "✓ Successfully downloaded: $output_path"
                    exit 0
                else
                    log "✗ Download command succeeded but file is empty or missing"
                fi
            fi

            log "✗ Download failed (attempt $attempt/$max_retries), retrying in ${current_delay}s..."
            sleep $current_delay
            current_delay=$((current_delay * 2))  # Exponential backoff
            attempt=$((attempt + 1))
        done

        # All retries failed
        log "[ERROR] Failed to download $output_path after $max_retries attempts"
        rm -f "$temp_file"
        release_slot "$slot"
        exit 1

    ) 200>"$lockfile"

    local result=$?
    # Clean up lockfile after completion
    rm -f "$lockfile"
    return $result
}

acquire_slot() {
    local prefix="$1"
    local max_slots="$2"
    
    while true; do
        local count
        count=$(find "$(dirname "$prefix")" -name "$(basename "$prefix")_*" 2>/dev/null | wc -l)
        if [ "$count" -lt "$max_slots" ]; then
            local slot="${prefix}_$$_$RANDOM"
            touch "$slot"
            echo "$slot"
            return 0
        fi
        sleep 0.5
    done
}

release_slot() {
    rm -f "$1"
}

main() {
    log "Starting ComfyUI provisioning..."
    
    # Activate virtual environment if it exists
    if [ -f /venv/main/bin/activate ]; then
        # shellcheck source=/dev/null
        . /venv/main/bin/activate
    fi

    # Clean up any leftover semaphores from previous runs
    rm -rf "$HF_SEMAPHORE_DIR"
    mkdir -p "$HF_SEMAPHORE_DIR"
    mkdir -p "$WORKFLOWS_DIR"
    mkdir -p "$INPUTS_DIR"
    mkdir -p "$MODELS_DIR"/{checkpoints,text_encoders,latent_upscale_models,loras}

    # Write workflows
    write_api_workflow

    # Periodically cleanup old generations
    set_cleanup_job

    # Collect all background job PIDs
    local pids=()

    # Download all HuggingFace models in parallel
    for model in "${HF_MODELS[@]}"; do
        url="${model%%|*}"
        output_path="${model##*|}"
        
        # Trim whitespace
        url=$(echo "$url" | xargs)
        output_path=$(echo "$output_path" | xargs)
        
        log "Queuing HF download: $url -> $output_path"
        download_hf_file "$url" "$output_path" &
        pids+=($!)
    done

    # Download all wget files in parallel
    for item in "${WGET_DOWNLOADS[@]}"; do
        # Skip empty entries
        [[ -z "${item// }" ]] && continue
        
        url="${item%%|*}"
        output_path="${item##*|}"
        
        # Trim whitespace
        url=$(echo "$url" | xargs)
        output_path=$(echo "$output_path" | xargs)
        
        log "Queuing wget download: $url -> $output_path"
        download_wget_file "$url" "$output_path" &
        pids+=($!)
    done

    # Wait for each job and check exit status
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            log "[ERROR] Download process $pid failed"
            failed=1
        fi
    done

    if [ $failed -eq 1 ]; then
        log "[ERROR] One or more downloads failed"
        exit 1
    fi

    log "✓ All downloads completed successfully"
}

# This workflow is as provided by ComfyUI template browser
# Adjustments as recommended at https://huggingface.co/Comfy-Org/ACE-Step_ComfyUI_repackaged/discussions/1#6845136255e580607333edda, converted to API format
write_api_workflow() {
    # Define the workflow JSON once
    local workflow_json
    read -r -d '' workflow_json << 'WORKFLOW_JSON' || true
{
  "92": {
    "inputs": {
      "filename_prefix": "video/MiniMax_H3",
      "format": "auto",
      "format.codec": "auto",
      "codec": "auto",
      "video": [
        "130",
        0
      ]
    },
    "class_type": "SaveVideo",
    "_meta": {
      "title": "Save Video"
    }
  },
  "115": {
    "inputs": {
      "aspect_ratio": "9:16 (Portrait Widescreen)",
      "megapixels": 0.4,
      "multiple": 32
    },
    "class_type": "ResolutionSelector",
    "_meta": {
      "title": "Resolution Selector (Size)"
    }
  },
  "119": {
    "inputs": {
      "vae_name": "minimax_h3_video_vae_fp16.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "120": {
    "inputs": {
      "vae_name": "minimax_h3_audio_vae_fp32.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "121": {
    "inputs": {
      "samples": [
        "125",
        0
      ],
      "vae": [
        "120",
        0
      ]
    },
    "class_type": "VAEDecodeAudio",
    "_meta": {
      "title": "VAE Decode Audio"
    }
  },
  "122": {
    "inputs": {
      "samples": [
        "125",
        0
      ],
      "vae": [
        "119",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "123": {
    "inputs": {
      "sampler_name": "res_multistep"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "124": {
    "inputs": {
      "scheduler": "simple",
      "steps": [
        "142",
        0
      ],
      "denoise": 1,
      "model": [
        "127",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "125": {
    "inputs": {
      "noise": [
        "129",
        0
      ],
      "guider": [
        "126",
        0
      ],
      "sampler": [
        "123",
        0
      ],
      "sigmas": [
        "124",
        0
      ],
      "latent_image": [
        "136",
        1
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "126": {
    "inputs": {
      "model": [
        "141",
        0
      ],
      "conditioning": [
        "136",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "Basic Guider"
    }
  },
  "127": {
    "inputs": {
      "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "128": {
    "inputs": {
      "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
      "type": "minimax",
      "device": "default"
    },
    "class_type": "CLIPLoader",
    "_meta": {
      "title": "Load CLIP"
    }
  },
  "129": {
    "inputs": {
      "noise_seed": 826470241454940
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "130": {
    "inputs": {
      "fps": 24,
      "bit_depth": 8,
      "color_space": "sRGB",
      "codec": "none",
      "images": [
        "122",
        0
      ],
      "audio": [
        "121",
        0
      ]
    },
    "class_type": "CreateVideo",
    "_meta": {
      "title": "Create Video"
    }
  },
  "131": {
    "inputs": {
      "expression": "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17",
      "values.a": [
        "132",
        0
      ]
    },
    "class_type": "ComfyMathExpression",
    "_meta": {
      "title": "Math Expression"
    }
  },
  "132": {
    "inputs": {
      "value": 8
    },
    "class_type": "PrimitiveFloat",
    "_meta": {
      "title": "Float (Duration)"
    }
  },
  "136": {
    "inputs": {
      "prompt": [
        "138",
        0
      ],
      "width": [
        "115",
        0
      ],
      "height": [
        "115",
        1
      ],
      "length": [
        "131",
        1
      ],
      "ref_image_size": "match",
      "clip": [
        "128",
        0
      ],
      "vae": [
        "119",
        0
      ],
      "audio_vae": [
        "120",
        0
      ],
      "ref_images.ref_image_0": [
        "137",
        0
      ],
      "ref_images.ref_image_1": [
        "139",
        0
      ],
      "ref_images.ref_image_2": [
        "164",
        0
      ],
      "ref_audios.ref_audio_0": [
        "161",
        0
      ],
      "ref_audios.ref_audio_1": [
        "165",
        0
      ]
    },
    "class_type": "MiniMaxH3ReferenceToVideo",
    "_meta": {
      "title": "MiniMax H3 Reference to Video"
    }
  },
  "137": {
    "inputs": {
      "image": "SeedVR_1.png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "138": {
    "inputs": {
      "value": "Use <Picture 1> as the exact identity reference for the man.\nUse <Picture 2> as the exact hotel lobby environment reference.\nUse <Picture 3> as the exact identity reference for the woman.\nUse <Audio 1> as the voice style reference for the man only.\nUse <Audio 2> as the voice style reference for the woman only.\n\nCreate a photorealistic cinematic scene inside the luxury hotel lobby from <Picture 2>.\n\nThe man from <Picture 1> stands directly in front of the woman from <Picture 3>, facing her at a natural conversational distance. Keep both characters’ facial identities, hairstyles, body proportions, clothing, and overall appearance consistent with their reference images. Keep the architecture, furniture, lighting, colors, and layout of the hotel lobby consistent with <Picture 2>.\n\nBoth characters maintain natural eye contact.\n\nThe man looks at the woman with a slightly nervous but sincere expression and says:\n\n\"Do you love me, Emily?\"\n\nThe woman looks back at him with an affectionate and confident expression and replies:\n\n\"Yes, I love you so much, let's fuck.\"\n\nMake sure the correct character speaks each line:\n- The man from <Picture 1> speaks only: \"Do you love me, Emily?\"\n- The woman from <Picture 3> speaks only: \"Yes, I love you so much, let's fuck.\"\n\nLIP SYNC AND VOICES:\n- Use <Audio 1> only as the voice style reference for the man.\n- Use <Audio 2> only as the voice style reference for the woman.\n- Match the tone, pacing, and vocal characteristics of each audio reference.\n- Do not swap the voices.\n- The man must have a male voice matching <Audio 1>.\n- The woman must have a female voice matching <Audio 2>.\n- Lip movements must synchronize naturally and precisely with each character’s own dialogue.\n- When one character is speaking, the other character remains silent and reacts naturally.\n\nBODY LANGUAGE:\n- The man stands naturally with a relaxed posture while asking the question.\n- The woman maintains eye contact and gives a subtle affectionate smile before answering.\n- Natural blinking, breathing, small head movements, and subtle body movement.\n- No exaggerated gestures.\n\nCAMERA:\n- Medium two-shot at eye level, showing both characters facing each other.\n- Keep both faces clearly visible.\n- Mostly static camera with a subtle cinematic slow push-in during the conversation.\n- One continuous shot.\n- No cuts.\n\nAUDIO:\n- Clear English pronunciation.\n- Distinct male and female voices.\n- No overlapping dialogue.\n- No background speech.\n\nSTYLE:\nPhotorealistic cinematic realism, luxury hotel atmosphere, realistic skin texture, natural indoor lighting, physically accurate shadows, realistic fabric movement, stable character identities, and smooth natural motion.\n\nIMPORTANT:\n- Do not change the man’s identity.\n- Do not change the woman’s identity.\n- Do not swap their faces or voices.\n- Do not merge facial features between the two characters.\n- Do not change their clothing.\n- Do not redesign the hotel lobby.\n- Do not introduce additional people.\n- Do not add subtitles or on-screen text.\n- Do not create scene transitions or camera cuts."
    },
    "class_type": "PrimitiveStringMultiline",
    "_meta": {
      "title": "Input Text (Prompt)"
    }
  },
  "139": {
    "inputs": {
      "image": "Mia_talking_at_hotel_room.png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "141": {
    "inputs": {
      "switch": [
        "146",
        0
      ],
      "on_false": [
        "127",
        0
      ],
      "on_true": [
        "145",
        0
      ]
    },
    "class_type": "ComfySwitchNode",
    "_meta": {
      "title": "If/Else Switch (model)"
    }
  },
  "142": {
    "inputs": {
      "switch": [
        "146",
        0
      ],
      "on_false": [
        "143",
        0
      ],
      "on_true": [
        "144",
        0
      ]
    },
    "class_type": "ComfySwitchNode",
    "_meta": {
      "title": "If/Else Switch (Steps)"
    }
  },
  "143": {
    "inputs": {
      "value": 20
    },
    "class_type": "PrimitiveInt",
    "_meta": {
      "title": "Int (Full)"
    }
  },
  "144": {
    "inputs": {
      "value": 4
    },
    "class_type": "PrimitiveInt",
    "_meta": {
      "title": "Int (Lightning LoRA)"
    }
  },
  "145": {
    "inputs": {
      "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
      "strength_model": 1,
      "model": [
        "127",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "Load LoRA"
    }
  },
  "146": {
    "inputs": {
      "value": false
    },
    "class_type": "PrimitiveBoolean",
    "_meta": {
      "title": "Boolean (Enable Lightning LoRA)"
    }
  },
  "150": {
    "inputs": {
      "value": "Bold comic-book ink style, heavy linework, red and blue-black palette, night city. Use <Picture 2> and <Picture 1> as reference frames and <Audio 1> exactly as it is.\nCUT 1: top-down view of the little boy superhero on the rooftop — red cape fluttering in the wind, hands planted on his hips, freckles and a cocky grin as he looks straight up into the camera. The camera slowly descends toward him as he delivers his line — as he speaks, comic-book graphic overlay text word by word in sync with his voice: \"GET READY TO\" - \"MEET\" — \"YOUR\" — \"MAKER\" — huge jagged comic lettering, white with heavy black outlines and red drop shadows, tilted at scrappy angles, until the three words hang stacked in the air above him between his face and the lens.\nTRANSITION: a violent WHIP PAN off the rooftop that SMEARS the floating words away with it, motion-streaked —\nCUT 2: low hero angle on the colossal black mech-kaiju towering over the skyline as it rears back and unleashes a GIANT terrifying ROAR — jaws wide with fangs, red eyes and chest-core flaring blinding bright, blue lightning arcing off its head, the roar's shockwave rippling dust and rattling windows down the buildings, comic-style speed-lines and ink splatter bursting from the impact of the sound. It leans INTO the camera as the roar peaks. Hold on the roar."
    },
    "class_type": "PrimitiveStringMultiline",
    "_meta": {
      "title": "Input Text (Prompt)"
    }
  },
  "161": {
    "inputs": {
      "audio": "Huy_Voice.m4a"
    },
    "class_type": "LoadAudio",
    "_meta": {
      "title": "Load Audio"
    }
  },
  "164": {
    "inputs": {
      "image": "Mia.png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "165": {
    "inputs": {
      "audio": "BichHue_Voice.m4a"
    },
    "class_type": "LoadAudio",
    "_meta": {
      "title": "Load Audio"
    }
  },
  "166": {
    "inputs": {
      "value": "Use <Picture 1> as the exact identity reference for the man.\nUse <Picture 2> as the exact hotel lobby environment reference.\nUse <Picture 3> as the exact identity reference for the woman.\nUse <Audio 1> as the voice style reference for the man only.\nUse <Audio 2> as the voice style reference for the woman only.\n\nCreate a photorealistic cinematic scene inside the luxury hotel lobby from <Picture 2>.\n\nThe man from <Picture 1> stands directly in front of the woman from <Picture 3>, facing her at a natural conversational distance. Keep both characters’ facial identities, hairstyles, body proportions, clothing, and overall appearance consistent with their reference images. Keep the architecture, furniture, lighting, colors, and layout of the hotel lobby consistent with <Picture 2>.\n\nBoth characters maintain natural eye contact.\n\nThe man looks at the woman with a slightly nervous but sincere expression and says:\n\n\"Do you love me, Emily?\"\n\nThe woman looks back at him with an affectionate and confident expression and replies:\n\n\"Yes, I love you so much, let's fuck.\"\n\nMake sure the correct character speaks each line:\n- The man from <Picture 1> speaks only: \"Do you love me, Emily?\"\n- The woman from <Picture 3> speaks only: \"Yes, I love you so much, let's fuck.\"\n\nLIP SYNC AND VOICES:\n- Use <Audio 1> only as the voice style reference for the man.\n- Use <Audio 2> only as the voice style reference for the woman.\n- Match the tone, pacing, and vocal characteristics of each audio reference.\n- Do not swap the voices.\n- The man must have a male voice matching <Audio 1>.\n- The woman must have a female voice matching <Audio 2>.\n- Lip movements must synchronize naturally and precisely with each character’s own dialogue.\n- When one character is speaking, the other character remains silent and reacts naturally.\n\nBODY LANGUAGE:\n- The man stands naturally with a relaxed posture while asking the question.\n- The woman maintains eye contact and gives a subtle affectionate smile before answering.\n- Natural blinking, breathing, small head movements, and subtle body movement.\n- No exaggerated gestures.\n\nCAMERA:\n- Medium two-shot at eye level, showing both characters facing each other.\n- Keep both faces clearly visible.\n- Mostly static camera with a subtle cinematic slow push-in during the conversation.\n- One continuous shot.\n- No cuts.\n\nAUDIO:\n- Clear English pronunciation.\n- Distinct male and female voices.\n- No overlapping dialogue.\n- No background speech.\n\nSTYLE:\nPhotorealistic cinematic realism, luxury hotel atmosphere, realistic skin texture, natural indoor lighting, physically accurate shadows, realistic fabric movement, stable character identities, and smooth natural motion.\n\nIMPORTANT:\n- Do not change the man’s identity.\n- Do not change the woman’s identity.\n- Do not swap their faces or voices.\n- Do not merge facial features between the two characters.\n- Do not change their clothing.\n- Do not redesign the hotel lobby.\n- Do not introduce additional people.\n- Do not add subtitles or on-screen text.\n- Do not create scene transitions or camera cuts."
    },
    "class_type": "PrimitiveStringMultiline",
    "_meta": {
      "title": "Input Text (Prompt)"
    }
  },
  "171": {
    "inputs": {
      "value": "Use Image 1 as the exact identity reference for the man.\nUse Image 2 as the starting composition and hotel hallway reference.\nUse Image 3 as the exact identity reference for the woman.\nUse Image 4 as the interior reference for hotel room 305.\n\nUse Audio 1 as the man's voice reference.\nUse Audio 2 as the woman's voice reference.\n\nMaintain the exact facial identity, hairstyle, body proportions, clothing, and overall appearance of both characters throughout the entire video. Both characters are adults.\n\nSCENE 1 — HOTEL HALLWAY, OUTSIDE ROOM 305\n\nStart from the composition of Image 2.\n\nFull-body cinematic shot. The man and woman stand facing each other directly outside hotel room 305. Both characters are completely visible from head to toe.\n\nThe man looks into the woman's eyes and asks naturally, using the voice and speaking style from Audio 1:\n\n\"Do you love me, Emily?\"\n\nNatural lip synchronization. His expression is serious but affectionate.\n\nThe woman maintains eye contact, gives a subtle intimate smile, and replies using the voice and speaking style from Audio 2:\n\n\"I love you, Dean. I want to be with you.\"\n\nNatural lip synchronization and realistic facial expressions.\n\nAfter she finishes speaking, they move slightly closer. The man gently takes her hand. Their fingers naturally interlock.\n\nHe turns toward door 305 while still holding her hand, reaches for the handle with his free hand, unlocks and opens the door.\n\nThe woman follows him naturally.\n\nCAMERA:\nSmooth cinematic tracking shot moving backward slightly as they approach the doorway.\nKeep both characters visible from head to toe whenever possible.\nNo abrupt camera movement.\nNatural body motion and realistic walking physics.\n\nSCENE 2 — ENTERING ROOM 305\n\nAs the door opens, smoothly transition into the interior shown in Image 4.\n\nThe man walks into the room first while holding the woman's hand. She follows closely behind him.\n\nThe camera follows them through the doorway in one continuous cinematic movement.\n\nAfter they enter, the man turns toward the woman.\n\nThe woman moves closer to him, gently places one hand against his waist, looks into his eyes with an intimate expression, and says using Audio 2:\n\n\"I really want to suck your dick.\"\n\nThey remain close together, creating romantic tension.\n\nEnd the scene with them looking at each other intimately.\n\nIMPORTANT CONSISTENCY:\n- Image 1 = man's identity.\n- Image 3 = woman's identity.\n- Image 2 = hotel hallway and initial body positioning.\n- Image 4 = room 305 interior.\n- Audio 1 ONLY for the man's dialogue.\n- Audio 2 ONLY for the woman's dialogue.\n- Preserve the same clothes from the reference images.\n- Preserve exact facial identity throughout.\n- Do not change hairstyle, face shape, age, or body proportions.\n- Accurate lip sync for every spoken sentence.\n- Natural hand and finger anatomy.\n- Realistic door interaction.\n- Continuous spatial consistency between hallway, door 305, and room interior.\n- Photorealistic cinematic lighting.\n- Realistic human movement.\n- No sudden cuts or unexplained character repositioning."
    },
    "class_type": "PrimitiveStringMultiline",
    "_meta": {
      "title": "Input Text (Prompt)"
    }
  }
}
WORKFLOW_JSON

    # Write first file (original format)
    rm -f /opt/comfyui-api-wrapper/payloads/*
    cat > /opt/comfyui-api-wrapper/payloads/wan_2.2_i2v.json << EOF
{
    "input": {
        "request_id": "",
        "workflow_json": ${workflow_json}
    }
}
EOF

    # Wait for directory to exist (from git clone), then write second file
    local benchmark_dir="$WORKSPACE/vast-pyworker/workers/comfyui-json/misc"
    while [[ ! -d "$benchmark_dir" ]]; do
        sleep 1
    done
    
    echo "$workflow_json" > "$benchmark_dir/benchmark.json"
}

# Add a cron job to remove older (oldest +24 hours) output files if disk space is low
set_cleanup_job() {
    local script_dir="/opt/instance-tools/bin"
    local script_path="${script_dir}/clean-output.sh"
    
    # Ensure directory exists
    mkdir -p "$script_dir"
    
    if [[ ! -f "$script_path" ]]; then
        cat > "$script_path" << 'CLEAN_OUTPUT'
#!/bin/bash
output_dir="${WORKSPACE:-/workspace}/ComfyUI/output/"
min_free_mb=512
available_space=$(df -m "${output_dir}" | awk 'NR==2 {print $4}')
if [[ "$available_space" -lt "$min_free_mb" ]]; then
    oldest=$(find "${output_dir}" -mindepth 1 -type f -printf "%T@\n" 2>/dev/null | sort -n | head -1 | awk '{printf "%.0f", $1}')
    if [[ -n "$oldest" ]]; then
        cutoff=$(awk "BEGIN {printf \"%.0f\", ${oldest}+86400}")
        # Only delete files
        find "${output_dir}" -mindepth 1 -type f ! -newermt "@${cutoff}" -delete
        # Delete broken symlinks
        find "${output_dir}" -mindepth 1 -xtype l -delete
        # Now delete *empty* directories separately
        find "${output_dir}" -mindepth 1 -type d -empty -delete
    fi
fi
CLEAN_OUTPUT
        chmod +x "$script_path"
    fi

    # Check if cron job already exists (avoiding pipefail issues)
    local cron_exists=0
    if crontab -l 2>/dev/null | grep -qF 'clean-output.sh'; then
        cron_exists=1
    fi
    
    if [[ "$cron_exists" -eq 0 ]]; then
        # Add the cron job
        (crontab -l 2>/dev/null || true; echo "*/10 * * * * ${script_path}") | crontab -
    fi
}

main