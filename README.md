# 🤗 TOPSIS — HuggingFace Text Generation Model Selector

A web app that loads real HuggingFace models, generates text, and uses TOPSIS to find the best one.

## 🚀 Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/topsis-model-selector.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch: `main` → file: `app.py`
5. Click **Deploy** ✅

That's it! Free hosting, public URL, runs in the browser.

---

## 💻 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🗂️ Files
```
topsis-web/
├── app.py            ← Streamlit web app (run this)
├── topsis.py         ← TOPSIS algorithm
├── generate.py       ← HuggingFace model loading + generation
├── requirements.txt  ← Dependencies
└── README.md
```
