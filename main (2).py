"""
AG News Text Classification Microservice
Model: TF-IDF + Logistic Regression (Module 3 Assignment)
Author: Alexandra Offei
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pickle
import os

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
    Builds and trains a TF-IDF + Logistic Regression pipeline on a broad,
    balanced set of seed examples covering geopolitics, war, religion, conflict,
    diplomacy, sports, business, and science/technology.
    Replace with your saved model_pipeline.pkl for full AG News accuracy.
    """
    seed_texts = [
        # ── WORLD (label 0) ──────────────────────────────────────────────────
        # Politics & diplomacy
        "United Nations holds emergency summit on global refugee crisis",
        "NATO allies discuss military strategy amid rising tensions in Eastern Europe",
        "Diplomatic talks resume between world leaders at G20 conference",
        "Foreign minister announces new trade agreement between nations",
        "International sanctions imposed on country following border dispute",
        "World leaders gather at climate summit to negotiate emissions targets",
        "Election results spark protests across multiple countries",
        "President signs executive order on immigration policy reform",
        "Prime minister resigns following vote of no confidence in parliament",
        "Ambassador expelled as diplomatic relations between two nations deteriorate",
        "United Nations security council votes on ceasefire resolution",
        "Government imposes martial law following widespread civil unrest",
        "Bilateral talks collapse as both sides reject proposed peace deal",
        "Foreign aid suspended amid concerns over human rights violations",
        "World Health Organization declares international public health emergency",
        # War, conflict & military
        "Iran war fears grow as military tensions escalate in the region",
        "US military conducts airstrikes targeting militant strongholds",
        "Troops advance on capital as civil war enters decisive phase",
        "Ceasefire agreement reached after weeks of intense fighting",
        "Ayatollah issues fatwa amid rising tensions with Western nations",
        "Iran nuclear deal negotiations stall as deadline approaches",
        "Missile strike kills dozens in densely populated urban area",
        "Rebel forces seize control of key border crossing",
        "Pentagon deploys additional troops to conflict zone",
        "War crimes tribunal indicts former general for atrocities",
        "Iran threatens retaliation following US sanctions announcement",
        "Military coup topples government overnight in surprise takeover",
        "Refugees flee conflict as fighting intensifies near civilian areas",
        "Drone strikes target terrorist training camps across border",
        "Nuclear standoff intensifies as Iran enriches uranium beyond limits",
        # Religion & world affairs
        "Ayatollah condemns western interference in Middle Eastern affairs",
        "Pope calls for peace as conflict spreads across the region",
        "Religious leaders meet to discuss interfaith dialogue and cooperation",
        "Islamic state claims responsibility for attack on government building",
        "Vatican issues statement condemning violence against civilians",
        "Muslim leaders denounce terrorist attacks in the name of religion",
        "Israel and Palestine clash over disputed holy sites in Jerusalem",
        "Iran supreme leader calls for resistance against foreign powers",
        "Archbishop urges world leaders to pursue peaceful resolution",
        "Mosque attack sparks international condemnation and protests",
        # ── SPORTS (label 1) ─────────────────────────────────────────────────
        "Championship team wins the league title after dramatic final match",
        "Olympic gold medalist breaks world record in swimming event",
        "Football club signs star striker for record transfer fee",
        "Tennis player advances to grand slam final after five-set thriller",
        "Basketball team defeats rivals in overtime to claim playoff spot",
        "Marathon runner sets new personal best at Boston race",
        "Soccer coach resigns after series of poor results this season",
        "Cyclist wins Tour de France for the third consecutive year",
        "World Cup qualifier ends in controversial draw amid referee dispute",
        "Boxer retains heavyweight title with stunning knockout in round seven",
        "Cricket team collapses as opposition bowlers dominate the innings",
        "Rugby union side wins Six Nations championship on points difference",
        "Formula One driver claims pole position at Monaco Grand Prix",
        "Swimmer shatters Olympic record in butterfly final at world championships",
        "Baseball pitcher throws perfect game for first time in a decade",
        "NBA star agrees record contract extension with franchise team",
        "Golf major winner dedicates trophy to late coach and mentor",
        "Athletes call for boycott of games over host country human rights record",
        "Coach suspended after players allege misconduct during training camp",
        "Stadium deal approved as city agrees to fund new sports arena",
        # ── BUSINESS (label 2) ───────────────────────────────────────────────
        "Federal Reserve raises interest rates to combat rising inflation",
        "Tech giant reports record quarterly earnings beating analyst forecasts",
        "Stock market falls sharply on recession fears and weak jobs data",
        "Startup raises venture capital funding in series B round",
        "Merger between two airline companies approved by regulators",
        "Oil prices surge following production cuts by OPEC nations",
        "Retail sales drop as consumers cut spending amid economic uncertainty",
        "Central bank holds interest rates steady amid inflation concerns",
        "Company files for bankruptcy protection after years of mounting losses",
        "Trade deficit widens as imports outpace exports for third quarter",
        "Hedge fund manager charged with insider trading by SEC investigators",
        "Corporate earnings miss expectations sending shares lower in after-hours",
        "Inflation rises to forty year high as energy costs continue to surge",
        "Unemployment rate falls to record low as job market remains tight",
        "Takeover bid rejected by board as shareholders demand higher premium",
        "Supply chain disruptions continue to affect global manufacturing output",
        "Housing market cools as mortgage rates climb to decade high levels",
        "Amazon announces thousands of layoffs amid declining advertising revenue",
        "European Central Bank signals further rate hikes to curb inflation",
        "Commodity prices soar as drought threatens agricultural supply chains",
        # ── SCIENCE / TECHNOLOGY (label 3) ───────────────────────────────────
        "Scientists discover new exoplanet in habitable zone of distant star",
        "Artificial intelligence model achieves breakthrough in protein folding",
        "NASA launches next-generation telescope to study deep space phenomena",
        "Researchers develop new battery technology for electric vehicles",
        "SpaceX successfully lands reusable rocket after orbital mission",
        "Study reveals climate change accelerating Arctic ice melt at record rate",
        "New COVID variant detected as health officials urge vaccination campaigns",
        "Quantum computing milestone achieved by research team at university",
        "Self-driving car completes first fully autonomous cross-country journey",
        "Gene editing technique shows promise in treating hereditary diseases",
        "Mars rover discovers evidence of ancient water in rock samples",
        "Tech company unveils new smartphone with revolutionary camera system",
        "Cybersecurity breach exposes millions of user passwords and personal data",
        "Scientists develop new vaccine showing high efficacy in clinical trials",
        "Machine learning algorithm outperforms doctors in cancer detection study",
        "5G network rollout accelerates as telecoms compete for market share",
        "Fusion energy experiment achieves net energy gain for first time ever",
        "Robotics firm debuts humanoid robot capable of complex household tasks",
        "Social media platform introduces AI-powered content moderation system",
        "Internet outage affects millions as major cloud provider suffers failure",
    ]

    seed_labels = [0]*40 + [1]*20 + [2]*20 + [3]*20

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
@app.get("/", response_class=HTMLResponse, summary="Landing page", tags=["General"])
@app.head("/", summary="Health check (HEAD)", tags=["General"])
def root():
    """HTML landing page — visible in a browser. HEAD is used by Render health checks."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AG News Text Classifier</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 760px; margin: 60px auto;
           padding: 0 24px; color: #1a1a1a; background: #f9f9f9; }
    h1   { color: #1a3a5c; border-bottom: 3px solid #0f6e56; padding-bottom: 10px; }
    h2   { color: #1a3a5c; margin-top: 32px; }
    .badge { display:inline-block; background:#e1f5ee; color:#0f6e56;
             padding:4px 12px; border-radius:6px; font-size:13px; font-weight:bold; }
    table { border-collapse: collapse; width: 100%; margin-top: 10px; }
    th    { background: #1a3a5c; color: white; padding: 8px 12px; text-align:left; }
    td    { padding: 8px 12px; border-bottom: 1px solid #ddd; font-size:14px; }
    tr:nth-child(even) td { background: #ebf3fb; }
    pre   { background:#eef3fa; border-left:4px solid #1a3a5c; padding:14px;
            font-size:13px; overflow-x:auto; border-radius:4px; }
    a     { color: #1a3a5c; font-weight: bold; }
    .note { background:#fff8e1; border-left:4px solid #f0a500; padding:10px 14px;
            font-size:13px; border-radius:4px; margin-top:16px; }
  </style>
</head>
<body>
  <h1>AG News Text Classifier</h1>
  <span class="badge">Live</span> &nbsp;
  <span style="font-size:14px;color:#555;">TF-IDF + Logistic Regression &nbsp;|&nbsp; EAI 6010 Module 5 &nbsp;|&nbsp; Alexandra Offei</span>

  <h2>What this service does</h2>
  <p>Classifies a news headline or short article snippet into one of four categories:
  <strong>World</strong>, <strong>Sports</strong>, <strong>Business</strong>, or
  <strong>Science/Technology</strong>.</p>

  <h2>Try it now</h2>
  <p>The easiest way to test is the interactive docs page — no tools needed:</p>
  <p><a href="/docs">&#x1F517; Open interactive API docs (/docs)</a></p>

  <h2>Quick API reference</h2>
  <table>
    <tr><th>Endpoint</th><th>Method</th><th>Purpose</th></tr>
    <tr><td>/</td><td>GET</td><td>This page</td></tr>
    <tr><td>/predict</td><td>POST</td><td>Classify a news text snippet</td></tr>
    <tr><td>/docs</td><td>GET</td><td>Interactive Swagger UI</td></tr>
  </table>

  <h2>Example request</h2>
  <pre>POST /predict
Content-Type: application/json

{ "text": "NASA launches new telescope to study exoplanets" }</pre>

  <h2>Example response</h2>
  <pre>{
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
}</pre>

  <div class="note">
    <strong>Note:</strong> This service runs on Render's free tier.
    If it has been idle, the first request may take up to 60 seconds to wake up.
    Subsequent requests respond in under 1 second.
  </div>
</body>
</html>
""", status_code=200)


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
