# 🍊 The Orange Problem — Multimodal SLM Fine-Tuning

Fine-tuning **SmolVLM-500M** on **ChartQA** for visual question answering on charts.

## Overview

| | |
|---|---|
| **Base Model** | [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) |
| **Dataset** | [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) |
| **Task** | Visual Question Answering on Charts |
| **Fine-tuning** | LoRA (r=16, alpha=32) with 4-bit NF4 quantization |
| **Hardware** | T4 GPU (Kaggle) |
| **HuggingFace Model** | 🔗 [YOUR_HF_USERNAME/smolvlm-500m-chartqa](https://huggingface.co/YOUR_HF_USERNAME/smolvlm-500m-chartqa) |

---

## Why These Choices?

### Model: SmolVLM-500M-Instruct
- Only **500M parameters** — fits comfortably on a T4 (16GB VRAM) with headroom for gradient storage
- Native multimodal support (image + text) without extra dependencies
- Maintained by HuggingFace — clean `transformers` integration, no version conflicts
- Instruction-tuned base — adapts quickly to chart QA with few epochs

### Dataset: ChartQA
- Rich multimodal dataset: chart images + natural language questions + answers
- Tests genuine **visual reasoning** on structured data (not just OCR)
- ~28,000 training examples; we use a 2,000-sample subset for T4 budget

### Fine-tuning method: LoRA
- Full fine-tuning of 500M parameters requires storing full gradients — too much VRAM on T4
- LoRA freezes the base model and adds small trainable rank-decomposition matrices (~1-2% of params)
- Result: 90%+ of full fine-tuning benefit at ~5% of the memory cost

---

## Repository Structure

```
├── 01_data_exploration.ipynb      # Dataset analysis, visualizations, preprocessing validation
├── 02_training.ipynb              # Full fine-tuning pipeline (run this on Kaggle T4)
└── README.md
```

---

## Quick Start — Inference

Pull the model from HuggingFace and run inference in a few lines:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# Load model directly from HuggingFace
model_id = "YOUR_HF_USERNAME/smolvlm-500m-chartqa"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Load your chart image
image = Image.open("your_chart.png").convert("RGB")
question = "What is the highest value shown in this chart?"

# Format as chat message
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"Look at this chart and answer the question.\n\nQuestion: {question}\nAnswer:"}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[[image]], return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)

# Decode only the generated answer (not the prompt)
generated = output[0][inputs["input_ids"].shape[1]:]
print(processor.decode(generated, skip_special_tokens=True))
```

---

## How to Reproduce Training

1. Open [Kaggle](https://kaggle.com) → New Notebook → set accelerator to **GPU T4**
2. Get a HuggingFace **write token** at https://huggingface.co/settings/tokens
3. Add it as a Kaggle Secret named `HF_TOKEN` (Add-ons → Secrets)
4. Upload and run `02_training.ipynb` cell by cell
5. Change `HF_USERNAME` in Step 10 to your HuggingFace username
6. The trained model uploads automatically at the end (~60-90 min total)

---

## Training Configuration

| Parameter | Value | Reasoning |
|---|---|---|
| LoRA rank (r) | 16 | Good capacity/memory balance for 500M model on T4 |
| LoRA alpha | 32 | Standard: 2x rank for stable scaling |
| LoRA dropout | 0.05 | Light regularization to prevent overfitting |
| Target modules | all-linear | Covers attention + FFN layers for maximum expressivity |
| Quantization | 4-bit NF4 | Keeps VRAM under T4's 16GB limit during training |
| Learning rate | 2e-4 | Standard starting LR for LoRA adapters |
| LR scheduler | cosine | Smoother decay than linear, better final convergence |
| Warmup ratio | 0.05 | 5% warmup prevents instability in early steps |
| Effective batch size | 16 (2 x 8 accum) | Standard for VLM training; fits T4 via gradient accumulation |
| Epochs | 3 | Enough signal on 2k samples without overfitting |
| Optimizer | paged_adamw_8bit | 4x less memory than standard AdamW |
| Max token length | 512 | Covers 99%+ of ChartQA samples |

---

## Requirements

```
transformers==4.49.0
peft
datasets
accelerate
bitsandbytes
huggingface_hub
pillow
trl
matplotlib
```

```bash
pip install transformers==4.49.0 peft datasets accelerate bitsandbytes huggingface_hub pillow trl matplotlib
```
