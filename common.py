SYSTEM = "You are Study Buddy. Always answer in exactly three parts: 'Short answer:', 'Example:', 'Remember:'. Use very simple words."
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # small, open, no login needed. Try "Qwen/Qwen2.5-1.5B-Instruct" for better quality.
# Questions that are NOT in the training data (used for before/after comparison)
TEST_PROMPTS = [
    "How do airplanes fly?",
    "What is a firewall?",
    "Why is the sky blue?",
    "What is an operating system?",
    "How do earthquakes happen?",
    "What is GDP?",
    "What is a variable in programming?",
    "How do noise-cancelling headphones work?",
]
