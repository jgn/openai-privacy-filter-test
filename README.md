# Privacy Filter Demo

A demo of [openai/privacy-filter](https://huggingface.co/openai/privacy-filter), a small (1.5B
param, 50M active) token-classification model that detects and redacts personally identifiable
information (PII) in text. It recognizes 8 categories of PII:

- **private_person** -- names
- **private_email** -- email addresses
- **private_phone** -- phone numbers
- **private_address** -- physical addresses
- **private_date** -- dates of birth and other private dates
- **private_url** -- URLs containing private information
- **account_number** -- account numbers, SSNs, etc.
- **secret** -- passwords, API keys, etc.

This repo contains two interfaces to the model:

- `redact.py` -- Python CLI tool for batch processing text files
- `index.html` -- Browser-based demo that runs entirely client-side

## Python CLI

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers
```

### Download the model

The first run will download the model from Hugging Face (~2.6 GB). To avoid re-downloading on every
run, you can pre-download into a local `model/` directory:

```bash
mkdir -p model
# Download config, tokenizer, and weights
for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -L -o "model/$f" "https://huggingface.co/openai/privacy-filter/resolve/main/$f"
done
```

The script automatically uses `./model` if present, otherwise falls back to the Hugging Face hub.

### Usage

```bash
# Redact a file
python redact.py sample.txt

# Read from stdin
cat sample.txt | python redact.py -

# Only redact high-confidence detections
python redact.py --threshold 0.9 sample.txt

# Use a specific model path or HF model ID
python redact.py --model openai/privacy-filter sample.txt
```

Output goes to stdout with PII replaced by `[REDACTED:category]` tags. Model loading messages go to
stderr so you can pipe the output cleanly.

## Browser demo

### How it works

The browser demo runs the model entirely client-side -- no data leaves your machine. It uses:

- **ONNX Runtime Web** (WASM backend) for inference
- **Transformers.js** `AutoTokenizer` for tokenization
- The **q4f16 quantized ONNX** variant of the model (~772 MB)

### Running it

Serve this directory over HTTP (needed for ES module imports):

```bash
python3 -m http.server 8080
```

Then open http://localhost:8080 in Chrome or Edge. On first load, the model weights (~772 MB) are
downloaded from Hugging Face and cached by the browser. Subsequent loads are fast.

Paste or type text into the input box and click **Redact**. Detected PII appears as black pills;
hover to see the original text and category.

## Development notes

### Why ONNX for the browser?

The model uses a custom architecture (`openai_privacy_filter`) with sparse mixture-of-experts and
banded attention. As of April 2025,
[Transformers.js](https://github.com/huggingface/transformers.js) does not support this model type,
so the `pipeline("token-classification", "openai/privacy-filter")` approach that works in Python
fails in the browser with:

> Unsupported model type: openai_privacy_filter

The workaround is to use ONNX Runtime Web directly with the pre-exported ONNX model files that
OpenAI provides in the repo. The Transformers.js `AutoTokenizer` still works fine for tokenization
since the tokenizer itself is a standard BPE tokenizer -- only the model class is unsupported.

### Offset mapping

The Transformers.js tokenizer does not return `offset_mapping` (character-level start/end positions
for each token) the way the Python `transformers` library does. The browser demo computes offsets
manually by decoding each token back to text and finding its position in the original string. This
is slightly fragile but works reliably for the BPE tokenizer used by this model.

### BIOES decoding

The model outputs 33 classes: a background label (`O`) plus 8 PII categories x 4 BIOES tags (Begin,
Inside, End, Single). The Python CLI relies on the `transformers` library's built-in
`aggregation_strategy="simple"` to merge sub-token predictions into spans. The browser demo
implements BIOES span decoding manually, merging adjacent spans of the same category.

### Model variants

The Hugging Face repo includes several ONNX variants:

| Variant                | Size    | Notes                        |
| ---------------------- | ------- | ---------------------------- |
| `model.onnx`           | ~5.6 GB | Full precision (f32)         |
| `model_fp16.onnx`      | ~2.8 GB | Half precision               |
| `model_q4.onnx`        | ~917 MB | 4-bit quantized              |
| `model_q4f16.onnx`     | ~772 MB | 4-bit quantized, fp16 scales |
| `model_quantized.onnx` | ~1.6 GB | 8-bit quantized              |

The browser demo uses `model_q4f16.onnx` for the smallest download. The Python CLI uses the full
safetensors weights via PyTorch.

### Limitations

- English-focused; accuracy drops on non-English text and non-Latin scripts
- May over-redact public entities (e.g., company names, city names)
- May miss uncommon names, regional conventions, or domain-specific identifiers
- Not a substitute for a comprehensive privacy review -- use as one layer in a defense-in-depth
  approach
