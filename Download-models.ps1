[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$ModelsDir,
    [switch]$IncludeOptionalLoras,
    [switch]$UseConvV1Upscaler,
    [switch]$AiFilmStudio
)

$ErrorActionPreference = 'Stop'

if (-not $ModelsDir) {
    $configJson = docker compose --project-directory $PSScriptRoot -f (Join-Path $PSScriptRoot 'compose.yaml') config --format json
    if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the Compose configuration.' }

    $config = $configJson | ConvertFrom-Json
    $mount = $config.services.comfyui.volumes | Where-Object target -eq '/ComfyUI/models'
    if ($mount.type -ne 'bind' -or -not $mount.source) {
        throw 'Compose does not define a host model folder. Specify -ModelsDir.'
    }
    $ModelsDir = $mount.source
}
$ModelsDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ModelsDir)
Write-Host "Model folder: $ModelsDir"

$files = [ordered]@{
    'diffusion_models/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors' = 'smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models/resolve/main/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors'
    'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors' = 'Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
    'vae/minimax_h3_video_vae_int8_convrot.safetensors' = 'Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_video_vae_int8_convrot.safetensors'
    'vae/minimax_h3_audio_vae_fp32.safetensors' = 'Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors'
    'loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors' = 'lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'
}

if ($AiFilmStudio) {
    $files.Remove('diffusion_models/minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors')
    $files.Remove('vae/minimax_h3_video_vae_int8_convrot.safetensors')
    $files.Remove('loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors')
    $files['diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors'] = 'Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors'
    $files['vae/minimax_h3_video_vae_fp16.safetensors'] = 'Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors'
    $files['loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors'] = 'lightx2v/Minimax-h3-Turbo/resolve/a29e57d195f80ead07c89f89e55ecf2911e893c5/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors'
}
elseif ($UseConvV1Upscaler) {
    $files['latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors'] = 'LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors'
} else {
    $files['latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors'] = 'LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/27236837051793bc7cccf628e5936c6110361dab/minimax_h3_latent_upscaler_3d_fp16.safetensors'
}

if ($IncludeOptionalLoras) {
    $files['loras/h3-realism-people-t2v-i2v-r2v.safetensors'] = 'fal/MiniMax-H3-Realism-People-LoRA/resolve/main/h3-realism-people-t2v-i2v-r2v.safetensors'
    $files['loras/minimax_h3_turbo_v4_step600_ema.safetensors'] = 'larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600_ema.safetensors'
}

foreach ($file in $files.GetEnumerator()) {
    $destination = Join-Path $ModelsDir $file.Key
    $existing = Get-Item -LiteralPath $destination -ErrorAction SilentlyContinue
    if ($existing) {
        if ($existing.PSIsContainer -or $existing.Length -eq 0) {
            throw "Expected a nonempty model file at $destination. Move the existing item aside and retry."
        }
        Write-Host "Already present: $($file.Key)"
        continue
    }

    if ($PSCmdlet.ShouldProcess($destination, 'Download from Hugging Face')) {
        $curl = Get-Command curl.exe -CommandType Application -ErrorAction Stop
        [void][IO.Directory]::CreateDirectory((Split-Path -Parent $destination))
        $partial = "$destination.$([guid]::NewGuid().ToString('N')).part"
        try {
            $url = "https://huggingface.co/$($file.Value)"
            Write-Host "Downloading: $($file.Key)"
            & $curl.Source --fail --location --retry 3 --connect-timeout 30 --progress-bar --output $partial $url
            if ($LASTEXITCODE -ne 0 -or (Get-Item -LiteralPath $partial).Length -eq 0) {
                throw "Download failed for $($file.Key). Run the script again to retry."
            }
            [IO.File]::Move($partial, $destination)
        } finally {
            if (Test-Path -LiteralPath $partial) {
                Remove-Item -LiteralPath $partial -Force -WhatIf:$false
            }
        }
    }
}
