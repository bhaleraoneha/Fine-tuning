# Study Buddy — LoRA Fine-Tuning Project

Fine-tuned a small open LLM (Qwen2.5-0.5B-Instruct) with **LoRA** so it always answers in one fixed, beginner-friendly format:
`Short answer:` / `Example:` / `Remember:` using very simple words.

## Task
- **Narrow task:** a specific answer format and tone (simple explanations for students).
- **Base model:** Qwen/Qwen2.5-0.5B-Instruct
- **Tools:** Hugging Face PEFT + TRL, Google Colab free T4 GPU

## Dataset
- `data/train.jsonl`: 34 hand-written examples (chat-message format: system / user / assistant)
- `data/train_full.jsonl`: <FILL IN: seed + N examples generated with the free Gemini API, then manually reviewed>
- Total used for training: <FILL IN> examples

## Training config
| Setting | Value |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 2e-4 |
| Epochs | 4 |
| Batch size | 4 (grad. accumulation 2) |

## Training loss
![loss curve](outputs/loss_curve.png)

<FILL IN: loss went from X to Y over N steps. Describe the curve honestly (smooth? noisy? flattened?).>

## Before / After comparison
Full side-by-side outputs for 8 unseen prompts: [`outputs/comparison.md`](outputs/comparison.md)

<PASTE 2-3 examples here from your real comparison.md>

## Honest assessment
<FILL IN from your own results. For example: Did the fine-tuned model follow the 3-part format? Did the base model? Did any answers contain factual mistakes? Where did it still fail? Remember that a 0.5B model can make factual errors, so the fine-tune mainly changed the STYLE and FORMAT, not the knowledge.>

## How to reproduce
1. Open Google Colab, set Runtime > T4 GPU, upload this folder.
2. `pip install -q -r requirements.txt`
3. (optional) `GEMINI_API_KEY=... python make_dataset_gemini.py`
4. `python train.py`
5. `python compare.py`
