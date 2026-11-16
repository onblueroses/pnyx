# Pnyx

Pnyx is a Chrome extension and a demo page that make listening visible on social media feeds. Pnyx was built at the Agora Hackathon with TUM.ai E-Lab in April 2026.

## What it does

The Chrome extension runs two AI models on-device. No post data leaves the browser for scoring. It adds four layers to a feed:

| Layer | What you see |
|-------|--------------|
| SEE | Badges show the claims in a post and whether the post gives reasons. |
| PAUSE | When you click Reply, an overlay shows the claims as checkboxes. You select the claims that you respond to. |
| SHOW | Your reply carries a tag: "Responded to: [claim]". |
| EXPLORE | Branching deliberation on a claim, through DeepSeek V3 on OpenRouter with your own key. |

The Habermas model scores claim risk and argument quality. Its reported F1 is 0.974, but this is agreement with model-generated labels, not discourse-detection accuracy. The Erscheinung model and 85 heuristic signals detect AI-generated text. [Pnyx in detail](docs/details.md) explains the theory, the architecture, the training pipeline and what the F1 measures.

## How a post is scored

```text
post text -> Habermas + Erscheinung (heuristics and ONNX) -> claim extractor -> badges, pause, tags
```

If the heuristic score is 4.0 or more, Erscheinung skips its ML inference. Habermas scores separately.

## Model files

The ONNX model files are not in this repo because they are too large. Download them from Hugging Face:

| Model | File | Size | Hugging Face |
|-------|------|------|--------------|
| Habermas v3 | `scaffold/frontend/model/model.onnx` | 270.3 MiB | [onblueroses/pnyx-habermas](https://huggingface.co/onblueroses/pnyx-habermas) |
| Erscheinung v0.7 | `scaffold/frontend/model-slop/model.onnx` | 125.7 MiB | [onblueroses/pnyx-erscheinung](https://huggingface.co/onblueroses/pnyx-erscheinung) |

The two ONNX files total 415,253,911 bytes (396.0 MiB). This counts model files only. Tokenizers and runtime files add to downloads. Browser memory use is unmeasured. [Size evidence](docs/details.md#model-file-sizes) gives exact bytes and source revisions.

The repo includes the tokenizer files and configs. If the model files are missing, the demo uses only heuristic scoring.

## Run the demo page

```bash
cd scaffold/frontend
python -m http.server 8080
# Open http://localhost:8080/index.html
```

The demo loads Habermas and uses heuristics for Erscheinung. Download traffic depends on runtime files and the browser cache. Seven pre-scored posts show all four layers.

## Install the Chrome extension

```bash
cd scaffold/frontend/extension
bash setup.sh
```

`setup.sh` copies the model files, detection modules and vendor libraries into the extension directory. Then in Chrome:

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click "Load unpacked".
4. Select the `scaffold/frontend/extension/` directory.

The extension adds Pnyx to LinkedIn and X/Twitter feeds.

## Layout

```text
scaffold/frontend/   demo page, inference worker, claim extractor, layers, extension
scripts/             data generation and training pipeline
data/                demo datasets and source credibility
docs/                model contracts and details
kb/                  research notes (problem space, prior art)
pitch/               presentation slides and demo recording
```

The design system is in [DESIGN-LANGUAGE.md](DESIGN-LANGUAGE.md).

## License

MIT. See [LICENSE](LICENSE).
