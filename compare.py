"""Run the SAME prompts through the base model and the fine-tuned model. Saves outputs/comparison.md"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common import MODEL, SYSTEM, TEST_PROMPTS

tok = AutoTokenizer.from_pretrained(MODEL)
dev = "cuda" if torch.cuda.is_available() else "cpu"
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
model = PeftModel.from_pretrained(base, "outputs/adapter").eval()

def gen(q):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=150, do_sample=False, repetition_penalty=1.1)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

lines = ["# Before vs After (same prompts, greedy decoding)\n"]
for q in TEST_PROMPTS:
    with model.disable_adapter():
        before = gen(q)
    after = gen(q)
    lines += [f"## Prompt: {q}\n", "**Base model:**\n", before + "\n", "**Fine-tuned model:**\n", after + "\n", "---\n"]
    print("done:", q)
open("outputs/comparison.md", "w", encoding="utf-8").write("\n".join(lines))
print("Saved outputs/comparison.md")
