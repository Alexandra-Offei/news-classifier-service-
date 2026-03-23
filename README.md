# AG News Text Classifier — Microservice

**EAI 6010 · Module 5 Assignment**
**Author:** Alexandra Offei

A REST API microservice that classifies news text into one of four categories using a **TF-IDF + Logistic Regression** model trained on the AG News dataset (Module 3).

---

## Categories

| ID | Label |
|----|-------|
| 0  | World |
| 1  | Sports |
| 2  | Business |
| 3  | Science/Technology |

---

## Live Service URL

```
https://news-classifier-offei.onrender.com
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/predict` | POST | Classify a news text snippet |
| `/docs` | GET | Interactive Swagger UI (test in browser) |

---

## Quick Test (no install needed)

Open this URL in any browser to test interactively:

```
https://news-classifier-offei.onrender.com/docs
```

Or use curl:

```bash
curl -X POST https://news-classifier-offei.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"NASA launches new telescope to study exoplanets\"}"
```

Expected response:

```json
{
  "predicted_category": "Science/Technology",
  "category_id": 3,
  "confidence": 0.9312,
  "all_probabilities": {
    "World": 0.0241,
    "Sports": 0.0089,
    "Business": 0.0358,
    "Science/Technology": 0.9312
  },
  "input_text": "NASA launches new telescope to study exoplanets"
}
```

---

## Input / Output

**Input** — POST `/predict` with a JSON body:

```json
{ "text": "your news headline or snippet here" }
```

**Output** — JSON with predicted category, confidence score, and probabilities for all four classes.

---

## Project Files

```
├── main.py              # FastAPI application (the microservice)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container build instructions
├── save_model.py        # Helper script to export your trained model
├── model_pipeline.pkl   # Your saved model (add this from your notebook)
└── README.md            # This file
```

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload

# Visit http://localhost:8000/docs to test
```

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set environment to **Docker**
5. Click **Deploy** — Render builds and hosts it automatically

> **Note:** Free-tier services sleep after 15 minutes of inactivity. The first request after idle may take up to 60 seconds. Subsequent requests respond in under 1 second.

---

## Model Details

- **Algorithm:** TF-IDF vectorization + Logistic Regression (scikit-learn)
- **Dataset:** AG News (10,000 training / 2,000 test samples)
- **Accuracy:** 90.55% on test set
- **Macro F1:** 0.905
- **Training time:** ~1.5 seconds on CPU
