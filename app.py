import streamlit as st
import pandas as pd
import numpy as np
import time
import math
import torch
import warnings
warnings.filterwarnings("ignore")

from topsis import run_topsis

# ── Page config
st.set_page_config(
    page_title="TOPSIS — HuggingFace Model Selector",
    page_icon="",
    layout="wide",
)

# ── Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.block-container { padding-top: 2rem; }

.winner-box {
    background: linear-gradient(135deg, #7c6ef722, #00d4aa11);
    border: 2px solid #7c6ef7;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.metric-card {
    background: #18181f;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.generated-box {
    background: #111118;
    border-left: 3px solid #7c6ef7;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    line-height: 1.7;
}
.step-badge {
    display: inline-block;
    background: #7c6ef722;
    border: 1px solid #7c6ef755;
    color: #7c6ef7;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    text-align: center;
    line-height: 28px;
    font-weight: 800;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Models available
ALL_MODELS = [
    {"name": "DistilGPT-2",   "hf_id": "distilgpt2",               "org": "HuggingFace", "size": "82MB"},
    {"name": "GPT-2",          "hf_id": "gpt2",                      "org": "OpenAI",      "size": "117MB"},
    {"name": "GPT-Neo 125M",   "hf_id": "EleutherAI/gpt-neo-125m",  "org": "EleutherAI",  "size": "125MB"},
]

CRITERIA = [
    {"key": "perplexity",  "label": "Perplexity",         "type": "cost",    "unit": "PPL",   "weight": 30},
    {"key": "speed",       "label": "Speed (tok/s)",       "type": "benefit", "unit": "tok/s", "weight": 25},
    {"key": "diversity",   "label": "Vocab Diversity",     "type": "benefit", "unit": "%",     "weight": 25},
    {"key": "avg_length",  "label": "Avg Output Length",   "type": "benefit", "unit": "tok",   "weight": 20},
]

DEFAULT_PROMPTS = [
    "Artificial intelligence is transforming the world by",
    "The future of healthcare depends on",
    "Once upon a time in a land far away,",
]


# ── Generation function (cached per model+prompt combo)
@st.cache_resource(show_spinner=False)
def load_model(hf_id):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated, len(new_tokens), elapsed


def compute_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    if input_ids.shape[1] < 2:
        return 999.0
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())


def run_model(model_info, prompts, max_new_tokens):
    tokenizer, model = load_model(model_info["hf_id"])
    perplexities, speeds, lengths, diversities = [], [], [], []
    generated_texts = []

    for prompt in prompts:
        generated, n_tokens, elapsed = generate_text(tokenizer, model, prompt, max_new_tokens)
        full_text = prompt + " " + generated
        ppl = compute_perplexity(model, tokenizer, full_text)
        speed = n_tokens / elapsed if elapsed > 0 else 0
        words = generated.split()
        diversity = len(set(words)) / len(words) if words else 0

        perplexities.append(ppl)
        speeds.append(speed)
        lengths.append(n_tokens)
        diversities.append(diversity)
        generated_texts.append({"prompt": prompt, "generated": generated})

    return {
        **model_info,
        "perplexity":  round(sum(perplexities) / len(perplexities), 2),
        "speed":       round(sum(speeds)       / len(speeds),       2),
        "avg_length":  round(sum(lengths)      / len(lengths),      2),
        "diversity":   round(sum(diversities)  / len(diversities),  4),
        "generated_texts": generated_texts,
    }


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

st.title(" TOPSIS — HuggingFace Text Generation Model Selector")
st.caption("Loads real pre-trained models · Generates actual text · Uses TOPSIS to find the best one")

st.divider()

# ── Sidebar: settings
with st.sidebar:
    st.header(" Settings")

    st.subheader("Models")
    selected_models = []
    for m in ALL_MODELS:
        if st.checkbox(f"{m['name']} ({m['size']})", value=True, key=m["hf_id"]):
            selected_models.append(m)

    st.subheader("Generation")
    max_tokens = st.slider("Max new tokens", 30, 150, 80)

    st.subheader("Prompts")
    prompts = []
    for i, p in enumerate(DEFAULT_PROMPTS):
        val = st.text_input(f"Prompt {i+1}", value=p, key=f"prompt_{i}")
        if val.strip():
            prompts.append(val.strip())

    st.subheader("TOPSIS Weights")
    weights = []
    for c in CRITERIA:
        icon = "↓" if c["type"] == "cost" else "↑"
        w = st.slider(f"{icon} {c['label']}", 0, 50, c["weight"], key=f"w_{c['key']}")
        weights.append(w)

    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ── TOPSIS Steps tab info
tab1, tab2 = st.tabs(["Results", "TOPSIS Steps"])

with tab2:
    steps = [
        ("Decision Matrix",     "Rows = models, Columns = criteria. Raw measured values from actual generation."),
        ("Normalise",           "rᵢⱼ = xᵢⱼ / √(Σxᵢⱼ²) — Euclidean column norm makes all criteria comparable."),
        ("Apply Weights",       "vᵢⱼ = wⱼ × rᵢⱼ — multiply by your chosen weights (normalised to sum=1)."),
        ("Ideal Best A⁺",       "benefit → max(vᵢⱼ),  cost → min(vᵢⱼ). This is the 'perfect' model."),
        ("Ideal Worst A⁻",      "benefit → min(vᵢⱼ),  cost → max(vᵢⱼ). This is the 'worst possible' model."),
        ("Distances D⁺ & D⁻",  "D⁺ᵢ = √Σ(vᵢⱼ−A⁺ⱼ)²   D⁻ᵢ = √Σ(vᵢⱼ−A⁻ⱼ)²  — Euclidean distances."),
        ("Closeness Score Cᵢ",  "Cᵢ = D⁻ᵢ / (D⁺ᵢ + D⁻ᵢ)  ∈ [0,1]. Closer to 1 means closer to ideal best."),
        ("Rank",                "Sort all models by Cᵢ descending. Highest score = best model for your weights."),
    ]
    for i, (title, desc) in enumerate(steps):
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown(f'<div class="step-badge">{i+1}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{title}**  \n{desc}")
        st.write("")

with tab1:
    if not run_btn:
        st.info(" Configure settings in the sidebar, then click **Run Analysis**")

        st.subheader("Models that will be tested")
        cols = st.columns(len(ALL_MODELS))
        for i, m in enumerate(ALL_MODELS):
            with cols[i]:
                st.metric(m["name"], m["size"], m["org"])
    else:
        if len(selected_models) < 2:
            st.error("Please select at least 2 models.")
            st.stop()
        if not prompts:
            st.error("Please enter at least 1 prompt.")
            st.stop()

        # ── Run generation
        all_results = []
        progress = st.progress(0, text="Starting...")

        for idx, m in enumerate(selected_models):
            progress.progress(idx / len(selected_models), text=f"⏳ Loading & running {m['name']}...")
            with st.spinner(f"Generating text with {m['name']}..."):
                result = run_model(m, prompts, max_tokens)
                all_results.append(result)

        progress.progress(1.0, text=" Generation complete!")
        time.sleep(0.5)
        progress.empty()

        # ── TOPSIS
        ranked, A_best, A_worst, w_norm = run_topsis(all_results, CRITERIA, weights)
        winner = ranked.iloc[0]

        # ── Winner banner
        st.markdown(f"""
        <div class="winner-box">
            <h2> Best Model: {winner['name']}</h2>
            <p style="color:#aaa;font-family:monospace">
                by {winner['org']} &nbsp;|&nbsp; 
                TOPSIS Score = <b style="color:#7c6ef7">{winner['score']:.4f}</b> &nbsp;|&nbsp;
                D⁺ = {winner['d_best']:.5f} &nbsp;|&nbsp;
                D⁻ = {winner['d_worst']:.5f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Rankings table
        st.subheader(" TOPSIS Rankings")
        medals = ["1", "2", "3"] + [f"#{i}" for i in range(4, 20)]
        display_df = pd.DataFrame({
            "Rank":        [medals[i] for i in range(len(ranked))],
            "Model":       ranked["name"].values,
            "Org":         ranked["org"].values,
            "Score (Cᵢ)":  ranked["score"].values,
            "D⁺":          ranked["d_best"].values,
            "D⁻":          ranked["d_worst"].values,
            "Perplexity↓": ranked["perplexity"].values,
            "Speed↑":      ranked["speed"].values,
            "Diversity↑":  (ranked["diversity"] * 100).round(1).astype(str) + "%",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Score bar chart
        st.subheader(" Score Comparison")
        chart_df = pd.DataFrame({
            "Model": ranked["name"].values,
            "TOPSIS Score": ranked["score"].values,
        }).set_index("Model")
        st.bar_chart(chart_df)

        # ── Real metrics
        st.subheader(" Real Measured Metrics")
        cols = st.columns(4)
        metric_keys = ["perplexity", "speed", "diversity", "avg_length"]
        metric_labels = ["Perplexity ↓", "Speed tok/s ↑", "Diversity ↑", "Avg Length ↑"]
        for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            with cols[i]:
                for r_idx, row in ranked.iterrows():
                    val = row[key]
                    if key == "diversity":
                        val = f"{val*100:.1f}%"
                    elif key == "perplexity":
                        val = f"{val:.2f}"
                    else:
                        val = f"{val:.1f}"
                    st.metric(f"{row['name']}" if i == 0 else "", val, label if r_idx == 1 else "")

        # Cleaner metrics table
        metrics_display = pd.DataFrame([{
            "Model":       r["name"],
            "Perplexity↓": r["perplexity"],
            "Speed↑":      r["speed"],
            "Diversity↑":  f"{r['diversity']*100:.1f}%",
            "Avg Length↑": r["avg_length"],
        } for r in all_results])
        st.dataframe(metrics_display, use_container_width=True, hide_index=True)

        # ── Weights used
        st.subheader("⚖️ Weights Used")
        w_df = pd.DataFrame({
            "Criterion": [c["label"] for c in CRITERIA],
            "Weight %":  [f"{w*100:.1f}%" for w in w_norm],
            "Type":      [c["type"] for c in CRITERIA],
        })
        st.dataframe(w_df, use_container_width=True, hide_index=True)

        # ── Generated texts
        st.subheader("Generated Texts")
        for r in all_results:
            with st.expander(f" {r['name']} — click to expand"):
                for t in r["generated_texts"]:
                    st.markdown(f"**Prompt:** {t['prompt']}")
                    st.markdown(f'<div class="generated-box">{t["generated"]}</div>', unsafe_allow_html=True)
                    st.write("")

        # ── Download CSV
        st.subheader(" Export")
        csv_cols = ["name", "org", "score", "d_best", "d_worst", "perplexity", "speed", "diversity", "avg_length"]
        csv_data = ranked[[c for c in csv_cols if c in ranked.columns]].to_csv(index=True)
        st.download_button(" Download Results CSV", csv_data, "topsis_results.csv", "text/csv")
