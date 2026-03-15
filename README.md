# 🍊 The Orange Problem — Multimodal SLM Fine-Tuning

Fine-tuning **Qwen2-VL-2B-Instruct** on **ChartQA** for visual question answering on charts.

---

## Team — NLPee-ers

| Name | SRN |
|---|---|
| Shreyas S Mallappa | PES1UG23AM296 |
| Sanat Shirwaicar | PES1UG23AM266 |
| Siddharth M | PES1UG23AM303 |
| Shreyansh Subham | PES1UG23AM295 |

---

## Overview

| | |
|---|---|
| **Base Model** | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| **Dataset** | [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) |
| **Task** | Visual Question Answering on Charts |
| **Fine-tuning** | LoRA (r=16, alpha=32) with 8-bit quantization |
| **Hardware** | T4 GPU (Kaggle) |
| **HuggingFace Model** | 🔗 [Devildarker6789/qwen2vl-chartqa](https://huggingface.co/YOUR_HF_USERNAME/qwen2vl-chartqa) |

---

## Why These Choices?

### Model: Qwen2-VL-2B-Instruct
- 2B parameters — fits on a T4 (16GB VRAM) with 8-bit quantization
- Native multimodal support (image + text) with dynamic resolution handling via `image_grid_thw`
- Strong baseline performance on visual reasoning and chart understanding tasks
- Well-supported by HuggingFace `transformers` with `Qwen2VLForConditionalGeneration`

### Dataset: ChartQA
- Rich multimodal dataset: chart images + natural language questions + answers
- Tests genuine visual reasoning on structured data — not just OCR
- ~28,000 training examples; we use a 2,000-sample subset for T4 time budget

### Fine-tuning: LoRA
- Full fine-tuning of 2B parameters requires too much VRAM on T4
- LoRA freezes the base model and adds small trainable rank-decomposition matrices (~1-2% of params)
- Targets q/k/v/o projections — the most impactful layers for visual reasoning
- Result: 90%+ of full fine-tuning benefit at a fraction of the memory cost

### Training Loop: Custom PyTorch
- Used a custom training loop instead of HuggingFace `Trainer`
- Allows explicit passing of `image_grid_thw` to the model forward pass
- `image_grid_thw` tells Qwen2-VL exactly how many visual tokens each image occupies, preventing shape mismatch errors
- Full control over checkpointing, gradient accumulation, and loss logging

---

## Repository Structure

```
├── 01_data_exploration.ipynb    # Dataset analysis, visualizations, preprocessing validation
├── 02_training.ipynb            # Full fine-tuning pipeline (run on Kaggle T4)
└── README.md
```

---

## Quick Start — Inference

```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import torch

# Load model from HuggingFace
model_id = "YOUR_HF_USERNAME/qwen2vl-chartqa"

processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Load your chart image
image = Image.open("your_chart.png").convert("RGB")
question = "What is the highest value shown in this chart?"

# Format as chat message
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": question}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)

generated = out[0][inputs["input_ids"].shape[1]:]
print(processor.decode(generated, skip_special_tokens=True))
```

---

## How to Reproduce Training

1. Open [Kaggle](https://kaggle.com) → New Notebook → set accelerator to **GPU T4**
2. Get a HuggingFace **write token** at https://huggingface.co/settings/tokens
3. Upload and run `02_training.ipynb` cell by cell
4. Set your `HF_REPO_ID` in the config cell
5. The trained model uploads to HuggingFace automatically at the end (~45 min total)

---

## Training Configuration

| Parameter | Value | Reasoning |
|---|---|---|
| LoRA rank (r) | 16 | Balance between capacity and memory for 2B model on T4 |
| LoRA alpha | 32 | Standard: 2× rank for stable scaling |
| LoRA dropout | 0.05 | Light regularization to prevent overfitting |
| Target modules | q, k, v, o projections | Most impactful layers for visual reasoning |
| Quantization | 8-bit int8 | Keeps VRAM under T4's 16GB limit; more stable than 4-bit for Qwen2-VL |
| Learning rate | 2e-4 | Standard starting LR for LoRA fine-tuning |
| LR scheduler | CosineAnnealingLR | Smooth decay; better final convergence than linear |
| Effective batch size | 16 (1 × 16 accum) | Standard for VLM training; batch=1 required to avoid OOM on T4 |
| Epochs | 1 | Full pass over 2k samples; ~45 min on T4 |
| Optimizer | AdamW | Standard; weight_decay=0.01 for regularization |
| Max token length | 768 | Fits T4 with batch=1; 512 too short, 1024 OOM |
| Image pixels | 256–512 patches | 28×28 per patch; 512 max = T4 safe with enough chart detail |

---

## Results

| Metric | Value |
|---|---|
| Train Loss | 0.5040 |
| Val Loss | 0.6956 |

Sample inference outputs after 1 epoch of fine-tuning:
- Q: *"What's the color of graph with 56 as the highest value?"* → **Blue** ✅
- Q: *"In which year the difference between blue and green graph 1?"* → **2017** (GT: 2018, off by 1)
- Q: *"What does the blue line represent?"* → **Not too much/ not at all** ✅

---

## Requirements

```bash
pip install transformers peft bitsandbytes accelerate datasets huggingface_hub qwen-vl-utils matplotlib tqdm
```
