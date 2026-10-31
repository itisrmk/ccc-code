#!/bin/sh
# Downloads optional local models to ~/.ccc/models and checks them against pinned hashes.
# Usage: scripts/fetch-models.sh classifier-small | classifier | ner
set -eu

fetch() { # repository file sha256 destination
    mkdir -p "$(dirname "$4")"
    if [ ! -f "$4" ] || [ "$(shasum -a 256 "$4" | cut -d' ' -f1)" != "$3" ]; then
        curl -fL --progress-bar -o "$4.part" "https://huggingface.co/$1/resolve/main/$2"
        mv "$4.part" "$4"
    fi
    actual="$(shasum -a 256 "$4" | cut -d' ' -f1)"
    if [ "$actual" != "$3" ]; then
        echo "hash mismatch for $4: expected $3, got $actual" >&2
        rm -f "$4"
        exit 1
    fi
    echo "ok: $4"
}

models="$HOME/.ccc/models"
case "${1:-}" in
    classifier-small)
        fetch Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf \
            9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031 \
            "$models/Qwen3-0.6B-Q8_0.gguf"
        ;;
    classifier)
        fetch Qwen/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf \
            7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5 \
            "$models/Qwen3-4B-Q4_K_M.gguf"
        ;;
    ner)
        fetch onnx-community/gliner_small-v2.1 onnx/model_quantized.onnx \
            c76c90920547fd937aaf505e7f2de5ec73168bf1c25abbb55a298104cb061400 \
            "$models/gliner/model_quantized.onnx"
        fetch onnx-community/gliner_small-v2.1 tokenizer.json \
            677203884d026e721115cf0daccf70ec4239545a13d6619e3e66d7151e0c9ce3 \
            "$models/gliner/tokenizer.json"
        ;;
    *)
        echo "usage: $0 classifier-small | classifier | ner" >&2
        exit 2
        ;;
esac
