# Study Buddy: LoRA Fine-Tuning of a Small LLM

I fine-tuned a small open-source language model (**Qwen2.5-0.5B-Instruct**) with **LoRA** so that it always answers in one fixed, beginner-friendly format:

`Short answer:` / `Example:` / `Remember:`, written in very simple words.

## Task
- **Narrow task:** a specific answer format and tone (simple explanations for students).
- **Base model:** Qwen/Qwen2.5-0.5B-Instruct
- **Tools:** Hugging Face PEFT + TRL, Google Colab free T4 GPU

## Dataset
- **161 examples** in chat-message format (system / user / assistant).
- 34 hand-written examples (`data/train.jsonl`) + 127 examples generated with the free Google Gemini API (`data/generated.jsonl`).
- The merged file used for training is `data/train_full.jsonl`.
- Every example follows the same 3-part format.
- Limitation: I skimmed the generated examples but did not fact-check every one.

## Training config
| Setting | Value |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 2e-4 (cosine schedule) |
| Epochs | 4 |
| Batch size | 4 (gradient accumulation 2) |
| Max sequence length | 512 |
| Trainable parameters | 2,162,688 of 496,195,456 (0.44%) |
| Hardware | Google Colab free T4 GPU, about 1 minute of training (84 steps) |

## Training loss
![Training loss curve](outputs/loss_curve.png)

Loss fell from **3.366 to 0.963** over 84 steps. Most of the drop happened in the first epoch (3.37 to about 1.16). After that the curve flattened and fluctuated between about 0.85 and 1.1, so extra epochs added little. Token accuracy rose from about 0.44 to about 0.77. The raw numbers are in `outputs/loss_log.json`.

## Before / After comparison
I ran the **same 8 unseen prompts** through the base model and the fine-tuned model, using the same system prompt and greedy decoding. Full outputs: [`outputs/comparison.md`](outputs/comparison.md)

**Prompt: What is a firewall?**

*Base model:* A firewall is like a strong barrier that keeps bad things from entering your computer or network, just like a gatekeeper keeps out unwanted visitors. It's made to protect your computer and keep it safe from hackers or other malicious people trying to harm you.

*Fine-tuned model:*
```
Short answer: A firewall is a computer program that keeps your internet safe by checking incoming and outgoing traffic to make sure it's not bad.

Example: Think of it like a security guard at the door who checks every visitor before letting them inside, just like a real firewall does on your computer network.

Remember: Firewalls help protect you from hackers and other bad people trying to break into your online world.
```

**Prompt: Why is the sky blue?**

*Base model:* The sky appears blue because it reflects sunlight, which has different colors depending on its wavelength. The blue light from the sun gets absorbed by the Earth's atmosphere, making it appear as blue. This phenomenon is called Rayleigh scattering.

*Fine-tuned model:*
```
Short answer: The sky is blue because it reflects sunlight.

Example: When you look at a clear blue sky, all the colors of the rainbow come together to make the bright blue light that we see.

Remember: Sunlight makes the sky appear blue because it bounces off tiny water droplets in the air and mixes with the Earth's atmosphere.
```

## Honest assessment
**What worked (format):** The fine-tuned model followed the 3-part format in **8 of 8** unseen prompts, while the base model followed it in **0 of 8**, even though both received the same system prompt. This is a clear effect of the fine-tuning. The tuned answers are also shorter and use simpler words.

**What did not work (accuracy):** Factual quality did not improve and sometimes got worse. For example, the tuned model explained the blue sky wrongly (the base model at least named Rayleigh scattering). Several analogies were confusing or wrong (operating system, GDP, variable), and neither model explained airplane lift or noise-cancelling headphones correctly. The firewall answer was a good result.

**Conclusion:** With a 0.5B model and 161 examples, LoRA taught the model a **style and format**, but it did not add new knowledge. A small model can still make factual mistakes.

**Limitations:** Only 8 test prompts, no automatic scoring, generated training data not fully fact-checked, and a very small base model. Next steps I would try: a larger base model such as Qwen2.5-1.5B, fact-checked data, and more varied question types.

## Project files
| File | Purpose |
|---|---|
| `build_seed.py` | Builds the 34 hand-written examples |
| `make_dataset_gemini.py` | Generates more examples with the free Gemini API |
| `train.py` | LoRA training with PEFT + TRL |
| `compare.py` | Before/after comparison on 8 unseen prompts |
| `common.py` | Shared settings (model name, system prompt, test prompts) |
| `data/` | Training data |
| `outputs/` | Loss curve, loss log, comparison results, LoRA adapter |

## How to reproduce
1. Open Google Colab, choose a **T4 GPU** runtime, and upload this repo.
2. `pip install -r requirements.txt` (if you see a torchao error, run `pip uninstall -y torchao`)
3. *(Optional)* set `GEMINI_API_KEY` and run `python make_dataset_gemini.py`
4. `python train.py`
5. `python compare.py`
