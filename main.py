"""
AG News Text Classification Microservice
Model: TF-IDF + Logistic Regression (Module 3 Assignment)
Author: Alexandra Offei
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import os
import re

# ── Try to load saved model, else train a demo model on startup ───────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = FastAPI(
    title="AG News Text Classifier",
    description=(
        "Classifies a news text snippet into one of four categories: "
        "World, Sports, Business, or Science/Technology. "
        "Based on a TF-IDF + Logistic Regression model trained on the AG News dataset (Module 3)."
    ),
    version="1.0.0",
    contact={"name": "Alexandra Offei"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Label mapping ─────────────────────────────────────────────────────────────
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}

# ── Load or build pipeline ────────────────────────────────────────────────────
MODEL_PATH = "model_pipeline.pkl"

def build_demo_pipeline():
    """
    Builds and trains a lightweight TF-IDF + Logistic Regression pipeline
    on representative seed examples so the service works without a saved model file.
    Replace this with pickle.load(open(MODEL_PATH,'rb')) if you have your saved model.
    """
    seed_texts = [
        # World
        "United Nations holds emergency summit on global refugee crisis",
        "NATO allies discuss military strategy amid rising tensions in Eastern Europe",
        "Diplomatic talks resume between world leaders at G20 conference",
        "Foreign minister announces new trade agreement between nations",
        "Peace negotiations collapse as conflict escalates in the region",
        "International sanctions imposed on country following border dispute",
        "World leaders gather at climate summit to negotiate emissions targets",
        "Election results spark protests across multiple countries",
        # Sports
        "Championship team wins the league title after dramatic final match",
        "Olympic gold medalist breaks world record in swimming event",
        "Football club signs star striker for record transfer fee",
        "Tennis player advances to grand slam final after five-set thriller",
        "Basketball team defeats rivals in overtime to claim playoff spot",
        "Marathon runner sets new personal best at Boston race",
        "Soccer coach resigns after series of poor results this season",
        "Cyclist wins Tour de France for the third consecutive year",
        # Business
        "Federal Reserve raises interest rates to combat rising inflation",
        "Tech giant reports record quarterly earnings beating analyst forecasts",
        "Stock market falls sharply on recession fears and weak jobs data",
        "Startup raises venture capital funding in series B round",
        "Merger between two airline companies approved by regulators",
        "Oil prices surge following production cuts by OPEC nations",
        "Retail sales drop as consumers cut spending amid economic uncertainty",
        "Central bank holds interest rates steady amid inflation concerns",
        # Science/Technology
        "Scientists discover new exoplanet in habitable zone of distant star",
        "Artificial intelligence model achieves breakthrough in protein folding",
        "NASA launches next-generation telescope to study deep space",
        "Researchers develop new battery technology for electric vehicles",
        "SpaceX successfully lands reusable rocket after orbital mission",
        "Study reveals climate change accelerating Arctic ice melt",
        "New COVID variant detected as health officials urge vaccination",
        "Quantum computing milestone achieved by research team",
    ]
    seed_labels = [0]*8 + [1]*8 + [2]*8 + [3]*8

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
        )),
    ])
    pipeline.fit(seed_texts, seed_labels)
    return pipeline


if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    print("Loaded saved model pipeline.")
else:
    print("No saved model found — using demo pipeline. "
          "For full accuracy, place your saved model_pipeline.pkl here.")
    pipeline = build_demo_pipeline()


# ── Request / Response schemas ────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        description="A news headline or short article snippet to classify.",
        examples=["NASA launches new telescope to study exoplanets and deep space phenomena"],
    )

class PredictionResponse(BaseModel):
    predicted_category: str = Field(..., description="Predicted news category label.")
    category_id: int = Field(..., description="Numeric category ID (0=World, 1=Sports, 2=Business, 3=Sci/Tech).")
    confidence: float = Field(..., description="Model confidence score (0.0 – 1.0) for the predicted class.")
    all_probabilities: dict = Field(..., description="Confidence scores for all four categories.")
    input_text: str = Field(..., description="The text that was classified.")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check", tags=["General"])
def root():
    """Returns service status and basic information."""
    return {
        "status": "ok",
        "service": "AG News Text Classifier",
        "model": "TF-IDF + Logistic Regression",
        "categories": list(LABELS.values()),
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a news text snippet",
    tags=["Prediction"],
)
def predict(body: TextInput):
    """
    Accepts a news headline or short article text and returns the predicted
    category along with confidence scores for all four classes.

    **Categories:**
    - **0 – World**: International news, politics, diplomacy, conflicts
    - **1 – Sports**: Athletic events, teams, players, competitions
    - **2 – Business**: Markets, economy, companies, finance
    - **3 – Science/Technology**: Science, tech, space, medicine, research
    """
    try:
        text = body.text.strip()
        proba = pipeline.predict_proba([text])[0]
        predicted_id = int(proba.argmax())
        predicted_label = LABELS[predicted_id]
        confidence = round(float(proba[predicted_id]), 4)
        all_probs = {LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)}

        return PredictionResponse(
            predicted_category=predicted_label,
            category_id=predicted_id,
            confidence=confidence,
            all_probabilities=all_probs,
            input_text=text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
