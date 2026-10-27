$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WhisperTarget = Join-Path $ProjectRoot "models\whisper\small"
$MediaPipeTarget = Join-Path $ProjectRoot "frontend\public\models\mediapipe"
$MediaPipeSource = Join-Path $ProjectRoot "frontend\node_modules\@mediapipe\tasks-vision\wasm"

if (-not (Test-Path $MediaPipeSource)) {
    throw "Frontend packages are missing. Run 'npm install' in the frontend folder first."
}

New-Item -ItemType Directory -Force -Path (Join-Path $MediaPipeTarget "wasm") | Out-Null
Copy-Item -Path (Join-Path $MediaPipeSource "*") -Destination (Join-Path $MediaPipeTarget "wasm") -Force

$FaceModel = Join-Path $MediaPipeTarget "face_landmarker.task"
if (-not (Test-Path $FaceModel)) {
    Invoke-WebRequest `
        -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
        -OutFile $FaceModel
}

New-Item -ItemType Directory -Force -Path $WhisperTarget | Out-Null
@'
from huggingface_hub import snapshot_download
import sys
snapshot_download(
    repo_id="Systran/faster-whisper-small",
    local_dir=sys.argv[1],
    allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"],
)
'@ | python - $WhisperTarget

Write-Host "Offline models are ready. You can now run the application without internet access."
