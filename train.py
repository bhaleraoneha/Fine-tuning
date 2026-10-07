"""LoRA fine-tuning with PEFT + TRL. Run on Google Colab (free T4 GPU)."""
import os, sys, json, inspect, torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import MODEL

DATA = "data/train_full.jsonl" if os.path.exists("data/train_full.jsonl") else "data/train.jsonl"
RANK, ALPHA, LR, EPOCHS = 16, 32, 2e-4, 4
print("Using data:", DATA, "| model:", MODEL, "| GPU:", torch.cuda.is_available())

tok = AutoTokenizer.from_pretrained(MODEL)
ds = load_dataset("json", data_files=DATA)["train"]
ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}, remove_columns=ds.column_names)
print("Examples:", len(ds)); print(ds[0]["text"][:400])

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)

lora = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

cfg_kwargs = dict(output_dir="outputs/checkpoints", num_train_epochs=EPOCHS, per_device_train_batch_size=4,
                  gradient_accumulation_steps=2, learning_rate=LR, logging_steps=2, save_strategy="no",
                  report_to="none", fp16=torch.cuda.is_available(), dataset_text_field="text", lr_scheduler_type="cosine")
params = inspect.signature(SFTConfig.__init__).parameters
cfg_kwargs["max_length" if "max_length" in params else "max_seq_length"] = 512
args = SFTConfig(**cfg_kwargs)

tk = "processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else "tokenizer"
trainer = SFTTrainer(model=model, args=args, train_dataset=ds, peft_config=lora, **{tk: tok})
trainer.model.print_trainable_parameters()
trainer.train()

os.makedirs("outputs", exist_ok=True)
trainer.model.save_pretrained("outputs/adapter"); tok.save_pretrained("outputs/adapter")
logs = [(l["step"], l["loss"]) for l in trainer.state.log_history if "loss" in l]
json.dump(logs, open("outputs/loss_log.json", "w"))
json.dump({"model": MODEL, "data": DATA, "examples": len(ds), "lora_rank": RANK, "lora_alpha": ALPHA,
           "learning_rate": LR, "epochs": EPOCHS}, open("outputs/config_used.json", "w"), indent=2)
plt.plot(*zip(*logs)); plt.xlabel("step"); plt.ylabel("training loss"); plt.title("Training loss"); plt.grid(True)
plt.savefig("outputs/loss_curve.png", dpi=150)
print("Saved adapter + outputs/loss_curve.png. First loss %.3f -> last loss %.3f" % (logs[0][1], logs[-1][1]))
