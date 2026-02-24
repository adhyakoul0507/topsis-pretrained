# generate.py — Load real HuggingFace models, generate text, measure metrics

import time
import math
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Models to compare (small enough to run on CPU)
MODELS_TO_TEST = [
    {
        "name": "DistilGPT-2",
        "hf_id": "distilgpt2",
        "org": "HuggingFace",
    },
    {
        "name": "GPT-2",
        "hf_id": "gpt2",
        "org": "OpenAI",
    },
    {
        "name": "GPT-Neo 125M",
        "hf_id": "EleutherAI/gpt-neo-125m",
        "org": "EleutherAI",
    },
]

# ── Prompts used for evaluation
PROMPTS = [
    "Artificial intelligence is transforming the world by",
    "The future of healthcare depends on",
    "Once upon a time in a land far away,",
]


def compute_perplexity(model, tokenizer, text, device):
    """Compute perplexity of generated text (lower = better)."""
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids

    if input_ids.shape[1] < 2:
        return float("inf")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


def generate_from_model(model_info, prompts, max_new_tokens=80):
    """
    Load a model, generate text for each prompt, measure:
      - perplexity (lower = better)
      - inference speed in tokens/sec (higher = better)
      - avg output length (higher = better, shows fluency)
      - vocabulary diversity: unique words / total words (higher = better)
    Returns dict with all metrics + generated texts.
    """
    from colorama import Fore, Style
    hf_id = model_info["hf_id"]
    name  = model_info["name"]

    print(f"\n  {Fore.CYAN}Loading {Style.BRIGHT}{name}{Style.RESET_ALL}{Fore.CYAN} ({hf_id})...{Style.RESET_ALL}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model     = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"  {Fore.RED}✗ Failed to load {name}: {e}{Style.RESET_ALL}")
        return None

    # Handle missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    perplexities = []
    speeds       = []
    lengths      = []
    diversities  = []
    generated_texts = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

        # ── Generate
        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.92,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        # Decode only the new tokens
        new_tokens   = output_ids[0][inputs["input_ids"].shape[1]:]
        generated    = tokenizer.decode(new_tokens, skip_special_tokens=True)
        full_text    = prompt + " " + generated
        n_new_tokens = len(new_tokens)

        # ── Metrics
        ppl   = compute_perplexity(model, tokenizer, full_text, device)
        speed = n_new_tokens / elapsed if elapsed > 0 else 0

        words     = generated.split()
        diversity = len(set(words)) / len(words) if words else 0

        perplexities.append(ppl)
        speeds.append(speed)
        lengths.append(n_new_tokens)
        diversities.append(diversity)
        generated_texts.append({"prompt": prompt, "generated": generated})

        print(f"    {Fore.GREEN}✓{Style.RESET_ALL} Prompt done — {n_new_tokens} tokens in {elapsed:.1f}s  |  PPL={ppl:.1f}  |  Speed={speed:.1f} tok/s")

    # Free memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        **model_info,
        "perplexity":  round(sum(perplexities) / len(perplexities), 2),
        "speed":       round(sum(speeds)       / len(speeds),       2),
        "avg_length":  round(sum(lengths)      / len(lengths),      2),
        "diversity":   round(sum(diversities)  / len(diversities),  4),
        "generated_texts": generated_texts,
    }


def run_all_models(prompts=None):
    """Run generation on all models and return list of result dicts."""
    if prompts is None:
        prompts = PROMPTS

    results = []
    for m in MODELS_TO_TEST:
        r = generate_from_model(m, prompts)
        if r:
            results.append(r)
    return results
