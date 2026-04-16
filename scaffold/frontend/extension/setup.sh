#!/bin/bash
# Copy model files and detection module into extension directory for loading unpacked.
# Run from the extension/ directory.

set -e
cd "$(dirname "$0")"

echo "Copying detection module..."
rm -rf ./detection/
cp -r ../detection/ ./detection/

echo "Patching detection module for MV3 CSP (CDN -> vendored imports)..."
sed -i \
  's|import { AutoTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";|import { AutoTokenizer } from "../vendor/transformers.min.js";|' \
  ./detection/slop-inference.js
sed -i \
  's|import \* as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.all.min.mjs";|import * as ort from "../vendor/ort.all.min.mjs";|' \
  ./detection/slop-inference.js
sed -i \
  's|from_pretrained(TOKENIZER_ID)|from_pretrained("./model-slop/")|' \
  ./detection/slop-inference.js
echo "  Patched slop-inference.js imports + tokenizer path"

echo "Copying Habermas model..."
rm -rf ./model/
mkdir -p model
cp ../model/model.onnx ../model/tokenizer.json ../model/tokenizer_config.json ../model/config.json ./model/

echo "Copying Erscheinung model..."
rm -rf ./model-slop/
mkdir -p model-slop
cp ../model-slop/model.onnx ../model-slop/token_remap.json ./model-slop/

echo "Copying Erscheinung tokenizer from demo page..."
cp ../model-slop/tokenizer.json ../model-slop/tokenizer_config.json ./model-slop/
if [ -f ../model-slop/spm.model ]; then
  cp ../model-slop/spm.model ./model-slop/
fi

echo "Copying Explore mode..."
cp ../explore.html ../explore.js ../explore.css ./

echo "Downloading vendor libraries (if needed)..."
mkdir -p vendor
if [ ! -f vendor/ort.all.min.mjs ]; then
  echo "  Downloading onnxruntime-web@1.24.3..."
  curl -sL "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort.all.min.mjs" -o vendor/ort.all.min.mjs
  curl -sL "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort-wasm-simd-threaded.wasm" -o vendor/ort-wasm-simd-threaded.wasm
  curl -sL "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort-wasm-simd-threaded.jsep.mjs" -o vendor/ort-wasm-simd-threaded.jsep.mjs
  curl -sL "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort-wasm-simd-threaded.jsep.wasm" -o vendor/ort-wasm-simd-threaded.jsep.wasm
  curl -sL "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/ort-wasm-simd.wasm" -o vendor/ort-wasm-simd.wasm
fi

if [ ! -f vendor/transformers.min.js ]; then
  echo "  Downloading @huggingface/transformers@3.8.1..."
  curl -sL "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1" -o vendor/transformers.min.js
fi

echo ""
echo "Done. Load unpacked extension from: $(pwd)"
echo "Total size: $(du -sh . | cut -f1)"
