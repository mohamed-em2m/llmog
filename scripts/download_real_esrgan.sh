#!/usr/bin/env bash
# Download Real-ESRGAN weights (default: realesr-general-x4v3).
#
# Model table mirrors llmog/esr/registry.py (single source of truth beside
# it -- keep the two in sync):
#   general-x4v3      v0.2.5.0  realesr-general-x4v3.pth      (~17MB, default)
#   general-wdn-x4v3  v0.2.5.0  realesr-general-wdn-x4v3.pth  (~17MB, denoise companion)
#   animevideov3      v0.2.5.0  realesr-animevideov3.pth       (~17MB, animation)
#   x4plus            v0.1.0    RealESRGAN_x4plus.pth          (~64MB, best still quality)
#   x4plus-anime-6B   v0.2.2.4  RealESRGAN_x4plus_anime_6B.pth (~18MB, anime stills)
#   x2plus            v0.2.1    RealESRGAN_x2plus.pth          (~64MB, native 2x)
#
# Usage:
#   ./scripts/download_real_esrgan.sh [--model KEY] [--out-dir DIR] [--sha256 HEX]
#
# Idempotent: skips download when the file already exists with a non-zero size.
set -euo pipefail

MODEL="general-x4v3"
OUT_DIR="."
SHA256=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --sha256)  SHA256="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

GH="https://github.com/xinntao/Real-ESRGAN/releases/download"
case "$MODEL" in
  general-x4v3)      FILE="realesr-general-x4v3.pth";      URL="$GH/v0.2.5.0/$FILE" ;;
  general-wdn-x4v3)  FILE="realesr-general-wdn-x4v3.pth";  URL="$GH/v0.2.5.0/$FILE" ;;
  animevideov3)      FILE="realesr-animevideov3.pth";      URL="$GH/v0.2.5.0/$FILE" ;;
  x4plus)            FILE="RealESRGAN_x4plus.pth";         URL="$GH/v0.1.0/$FILE" ;;
  x4plus-anime-6B)   FILE="RealESRGAN_x4plus_anime_6B.pth"; URL="$GH/v0.2.2.4/$FILE" ;;
  x2plus)            FILE="RealESRGAN_x2plus.pth";         URL="$GH/v0.2.1/$FILE" ;;
  *)
    echo "Unknown --model '$MODEL' (expected one of: general-x4v3, general-wdn-x4v3, animevideov3, x4plus, x4plus-anime-6B, x2plus)" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"
DEST="$OUT_DIR/$FILE"

if [[ -s "$DEST" ]]; then
  echo "Already present: $DEST (skipping download)"
else
  echo "Downloading $URL -> $DEST"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$DEST" "$URL"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL -o "$DEST" "$URL"
  else
    echo "Need wget or curl to download weights" >&2
    exit 1
  fi
fi

if [[ -n "$SHA256" ]]; then
  echo "Verifying sha256..."
  echo "$SHA256  $DEST" | sha256sum -c -
fi

echo "OK: $DEST"
