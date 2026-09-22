# 🐱 vs 🐶 Cats vs Dogs CNN — From Scratch (PyTorch)

A complete, beginner-friendly **Machine Learning project** that classifies images into **Cats (0)** or **Dogs (1)** using a **Convolutional Neural Network built entirely from scratch in PyTorch** — no transfer learning, no pretrained weights.

Real photo dataset: **Microsoft "Dogs vs Cats"** (~25,000 real photos, streamed from Hugging Face `microsoft/cats_vs_dogs`).

---

## ✨ Features

- 🏗️ **CNN from scratch**: 4 double-conv (Conv→BatchNorm→ReLU×2) blocks + Global Avg Pool + Dense head + Dropout
- 🛡️ **Leak-free split**: stratified 80/20 train/val split done *before* augmentation
- 🔄 **Real-time augmentation**: random flip, rotation, color jitter
- 📊 **Metrics & plots**: Accuracy / Precision / Recall / F1 + training curves + confusion matrix
- 📓 **One notebook, 3 platforms**: same training runs on **Google Colab**, **Kaggle**, and **local**
- 🌐 **Hostable**: FastAPI server with a drag-and-drop web UI + `Dockerfile`
- 📄 **Full write-up**: see [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)

---

## 📂 Project Structure

```text
cats-vs-dogs-classifier/
├── config.py                  # paths, hyperparameters, seeds
├── download_data.py           # stream real dataset → data/{cats,dogs}
├── notebooks/
│   └── cat_dog_cnn_pytorch.ipynb   # runs on Colab / Kaggle / local
├── docs/PROJECT_REPORT.md          # full documentation-style report
├── src/
│   ├── model.py               # from-scratch CNN (torch.nn)
│   ├── data.py                # loading + train/val split
│   ├── train.py               # training + evaluation + plots + save
│   ├── evaluate.py            # evaluate a checkpoint + export plots
│   └── predict.py             # CLI + library prediction
├── api/
│   ├── main.py                # FastAPI app
│   └── static/index.html      # upload UI
├── tests/test_predict.py      # pytest smoke tests
├── requirements.txt
├── Dockerfile                 # for hosting
├── data/{cats,dogs}/          # downloaded real images
├── models/cat_dog_cnn.pt      # trained checkpoint
└── outputs/                   # plots
```

---

## 🚀 Local Setup (Linux/macOS)

```bash
# 1. Python 3.10–3.12 recommended
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download real dataset (1200 images per class)
python download_data.py --num-images 1200

# 4. Train (30 epochs ~ 30 min on a normal CPU)
python src/train.py --epochs 30 --batch-size 32 --lr 0.001

# 5. Predict a single image
python src/predict.py path/to/cat_or_dog.jpg

# 6. Evaluate a checkpoint (uses the model's own mean/std)
python src/evaluate.py

# 7. Run the web app → http://localhost:8000
uvicorn api.main:app --reload --port 8000
```

> 💡 **No `python3.12`?** Your system Python may be too new for PyTorch (3.13+/3.14 often lack wheels). Grab a standalone one:
>
> ```bash
> # download python-build-standalone (astral-sh) 3.12 .tar.gz, then:
> tar -xzf cpython-3.12.14-install_only.tar.gz -C ~/python312 --strip-components=1
> ~/python312/bin/python3.12 -m venv .venv
> ```

---

## 📓 Train on Google Colab or Kaggle

Open `notebooks/cat_dog_cnn_pytorch.ipynb`:

| Platform | What happens |
|----------|--------------|
| **Colab** | Auto-detected. Streams N images/class from Hugging Face, trains on free GPU, auto-downloads `cat_dog_cnn.pt` |
| **Kaggle** | Auto-detected. Uses the built-in **Dogs vs Cats** dataset at `/kaggle/input/dogs-vs-cats`, saves model to `/kaggle/working` |
| **Local** | Auto-detected. Reuses `data/{cats,dogs}` (or downloads them) |

Steps: **Runtime → Run all**.

---

## 🌐 Hosting

### Option A — Docker (Render, Railway, any VPS)

```bash
docker build -t cats-dogs-cnn .
# mount trained weights so they aren't baked in:
docker run -v $(pwd)/models:/app/models -p 8000:8000 cats-dogs-cnn
```

### Option B — Render.com (free tier)

1. Push this repo to GitHub
2. New **Web Service** → runtime **Docker**
3. Deploy. The API serves the UI at `/` and predictions at `/predict`.

### Option C — Manual / VPS

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### API endpoints

| Method | Path       | Description                                  |
|--------|------------|----------------------------------------------|
| GET    | `/`        | HTML upload page                             |
| GET    | `/health`  | `{"status": "ok", "model_loaded": true}`     |
| POST   | `/predict` | multipart `file` → `{"label": "cat","confidence": 98.2}` |

---

## 🧠 Architecture

```text
Input (128×128×3)                             Output shape
  │
  ├─ [Conv2d(3→32)+BN+ReLU] → [Conv2d(32→32)+BN+ReLU] → MaxPool   (64, 64, 32)
  ├─ [Conv2d(32→64)+BN+ReLU] → [Conv2d(64→64)+BN+ReLU] → MaxPool  (32, 32, 64)
  ├─ [Conv2d(64→128)+BN+ReLU] → [Conv2d(128→128)+BN+ReLU] → MaxPool (16,16,128)
  ├─ [Conv2d(128→256)+BN+ReLU] → [Conv2d(256→256)+BN+ReLU] → MaxPool (8, 8, 256)
  ├─ Global Average Pool (1×1)              → 256
  ├─ Dense(128) + ReLU + Dropout(0.5)         128
  └─ Dense(1) ‑‑ sigmoid ‑‑> P(dog)            1
```

- Loss: **BCEWithLogitsLoss** (binary cross-entropy)
- Optimizer: **Adam** (lr 1e-3, weight decay 1e-4) + **cosine LR schedule**
- Regularization: **Dropout 0.5** + **BatchNorm** + **data augmentation**

---

## 📊 Evaluation

| Metric | Meaning |
|--------|---------|
| Accuracy | overall correct predictions |
| Precision | of all predicted *dogs*, how many are truly dogs |
| Recall | of all actual *dogs*, how many were found |
| F1-Score | harmonic mean of precision & recall |

Plots are written to `outputs/confusion_matrix.png` and `outputs/sample_predictions.png`.

> The bundled `models/cat_dog_cnn.pt` was trained on **Kaggle** via the notebook
> (accuracy ~78%, ROC-AUC ~0.84 on a 20% held-out split) and carries its own
> dataset-specific `mean`/`std`, which `predict.py`, `evaluate.py` and the API
> respect automatically.

---

## ✅ Run the tests

```bash
pytest -q
```

---

Made for B.Tech AI/ML · MIT License