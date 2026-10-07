# HOW TO RUN (Google Colab, free)

1. Go to https://colab.research.google.com and sign in with Google. Click **New notebook**.
2. Menu: **Runtime > Change runtime type > T4 GPU > Save**.
3. Click the folder icon (left side), drag and drop **lora-study-buddy.zip** into it.
4. Create code cells and run them one by one (Shift+Enter):

Cell 1 — unzip and enter folder
    !unzip -q lora-study-buddy.zip
    %cd lora-study-buddy

Cell 2 — install libraries (takes 1-2 min)
    !pip install -q -r requirements.txt

Cell 3 — (OPTIONAL) get more data with the free Gemini API
    import os
    os.environ["GEMINI_API_KEY"] = "PASTE_YOUR_FREE_KEY"   # from https://aistudio.google.com/apikey
    !python make_dataset_gemini.py
Then open data/generated.jsonl, skim it, delete bad lines. Skip this cell to train on the 34 seed examples only
(but the task asks for roughly 100-500, so do try it).

Cell 4 — train (about 5-15 minutes)
    !python train.py

Cell 5 — before/after comparison
    !python compare.py

Cell 6 — look at results
    from IPython.display import Image, display
    display(Image("outputs/loss_curve.png"))
    print(open("outputs/comparison.md").read())

Cell 7 — download your results
    !zip -qr results.zip outputs data
Then download results.zip from the left file panel.

# Then publish to GitHub
1. Create a new PUBLIC repo on github.com (e.g. study-buddy-lora).
2. Upload ALL files from this folder + the outputs/ folder (loss_curve.png, comparison.md, adapter/).
3. Edit README.md: replace every <FILL IN> with your real numbers and your real examples.
4. Paste the repo URL into Submit Task on the Interns Hub.

# Troubleshooting
- "CUDA out of memory": in train.py change per_device_train_batch_size=4 to 2.
- Gemini says model not found: set GEMINI_MODEL to the current free model name shown in Google AI Studio.
- TRL error about an argument: run  !pip install -U trl peft transformers  and re-run.
