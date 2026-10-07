"""OPTIONAL: grow the dataset to 100-200 examples using the FREE Google Gemini API.
Get a free key at https://aistudio.google.com/apikey
Run:  GEMINI_API_KEY=your_key python make_dataset_gemini.py
Windows PowerShell:  $env:GEMINI_API_KEY="your_key"; python make_dataset_gemini.py
If the model name is rejected, check the current free model name in Google AI Studio and set GEMINI_MODEL.
"""
import os, json, time, requests
from common import SYSTEM

KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
assert KEY, "Set GEMINI_API_KEY first"

TOPICS = """rain; thunder; lightning; the moon phases; a solar eclipse; the water cycle; food chain; human heart; lungs; digestion; bones; muscles; sleep; immune system; bacteria; virus; fossils; dinosaurs; climate change; recycling; solar panels; wind energy; nuclear power; petrol engine; electric car; airplane wings; rocket; satellite; telescope; microscope; atom; molecule; acid and base; chemical reaction; rust; fire; ice melting; boiling water; sound waves; light reflection; shadows; friction; momentum; simple machines; levers; pulley; speed and velocity; prime numbers; fractions; percentages; probability; average; geometry; algebra; zero; binary numbers; pixels; bits and bytes; RAM; hard disk; SSD; operating system; Linux; compiler; variable; loop; function; array; stack; queue; sorting; search; bug; debugging; version control; open source; website; browser; cookie; domain name; DNS; server; router; firewall; VPN; phishing; malware; two-factor authentication; email; social media; search engine; chatbot; large language model; training data; overfitting; deep learning; computer vision; speech recognition; robot; drone; 3D printing; virtual reality; QR code; Bluetooth; USB; touchscreen; camera; stock market; bank; loan; tax; budget; savings; GDP; unemployment; trade; currency; startup; marketing; brand; constitution; election; law; court; United Nations; history of the printing press; the Industrial Revolution; democracy in ancient Greece""".split(";")
TOPICS = [t.strip() for t in TOPICS if t.strip()]

PROMPT = """Write one study-buddy answer for a student who asks: "{q}"
Rules: use very simple words, be factually correct, and output EXACTLY this format with no extra text:
Short answer: <one sentence>

Example: <one everyday example or comparison, 1-2 sentences>

Remember: <one key takeaway sentence>"""

def ask(q):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}"
    body = {"contents": [{"parts": [{"text": PROMPT.format(q=q)}]}]}
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

def valid(a):
    return a.startswith("Short answer:") and "\n\nExample:" in a and "\n\nRemember:" in a

out = []
for t in TOPICS:
    q = f"What is {t}?" if not t.startswith(("how","why")) else t
    try:
        a = ask(q)
    except Exception as e:
        print("skip", t, e); time.sleep(5); continue
    if valid(a):
        out.append({"messages": [{"role":"system","content":SYSTEM},{"role":"user","content":q},{"role":"assistant","content":a}]})
        print("ok", len(out), q)
    else:
        print("bad format, skipped:", q)
    time.sleep(4)   # stay under free-tier rate limit

with open("data/generated.jsonl", "w", encoding="utf-8") as f:
    for ex in out: f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# merge with hand-written seed -> data/train_full.jsonl
with open("data/train.jsonl", encoding="utf-8") as f: seed = [l for l in f if l.strip()]
with open("data/train_full.jsonl", "w", encoding="utf-8") as f:
    f.writelines(seed)
    for ex in out: f.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"Done. {len(seed)} seed + {len(out)} generated -> data/train_full.jsonl")
print("IMPORTANT: open data/generated.jsonl and skim it. Delete wrong/odd examples before training.")
