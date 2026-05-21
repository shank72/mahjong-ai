# Riichi Mahjong AI

A full-stack AI system that learns to play Riichi Mahjong by training on real game data. Given a hand and the game state, it recommends the optimal discard — and models what opponents might be building from their visible discards.

**Live demo:** [your-demo-link.vercel.app](https://your-demo-link.vercel.app)
**API:** [your-api.railway.app](https://your-api.railway.app/docs)

---

## What it does

- **Discard model** — a neural network trained on 26M+ Tenhou phoenix-level game records that predicts the best tile to discard given your hand, dora indicators, seat wind, and turn number
- **Opponent modeling** — encodes each opponent's visible discard history to estimate their riichi risk and likely waits, adjusting the discard recommendation accordingly
- **Baseline comparison** — benchmarked against a rule-based tenpai-chasing bot; the neural model achieves **X% higher win rate** (fill in after training)
- **Explainability** — SHAP values show *why* the model chose a tile, surfaced in the demo UI as per-tile confidence bars
- **Playable demo** — React frontend where you enter a hand and see real-time AI recommendations with confidence scores

---

## Architecture

```
riichi-mahjong-ai/
├── data/
│   ├── download_logs.py       # fetch Tenhou phoenix logs
│   ├── parse_tenhou.py        # parse .mjlog XML into game states
│   ├── feature_encoder.py     # encode state → feature vector (uses mahjong lib)
│   ├── build_dataset.py       # build train/val .npz dataset
│   └── explore.ipynb          # dataset exploration notebook
│
├── model/
│   ├── baseline_bot.py        # rule-based tenpai-chasing baseline
│   ├── discard_model.py       # PyTorch MLP discard classifier (34-class output)
│   ├── opponent_model.py      # opponent risk estimator from discard sequences
│   ├── train.py               # training loop with validation
│   ├── evaluate.py            # win-rate evaluation vs baseline
│   └── explain.py             # SHAP explainability
│
├── api/
│   ├── main.py                # FastAPI app
│   ├── schemas.py             # Pydantic request/response models
│   └── inference.py           # model loading + inference
│
└── frontend/
    └── src/
        ├── App.jsx
        ├── components/
        │   ├── HandInput.jsx              # tile selector UI
        │   ├── DiscardRecommendation.jsx  # confidence bar chart
        │   └── OpponentTracker.jsx        # opponent discard tracker
        └── api/mahjong.js                 # API client
```

---

## Feature vector

Each training example encodes the full game state as a fixed-length vector:

| Feature group | Size | Description |
|---|---|---|
| Hand tiles        | 34 | One-hot count per tile type (man 1–9, pin 1–9, sou 1–9, honors 1–7)
| Shanten number    | 1 |  Tiles away from tenpai (via `mahjong` lib)
| Dora indicators   | 34 | Which tiles are dora this round
| Seat wind         | 4 |  One-hot: East / South / West / North
| Round wind        | 4 |  One-hot
| Turn number       | 1 |  Normalized 0–1
| Is tenpai         | 1 |  Binary flag
| Opponent discards (×3)     | 3 × 34 | Each opponent's visible discard counts 
| Opponent riichi flags (×3) | 3 | Whether each opponent has declared riichi 
| **Total** | **~150** | |

---

## Quickstart

### 1. Clone and install

```bash
#git clone https://github.com/your-username/riichi-mahjong-ai.git
#cd riichi-mahjong-ai
py -3.11 -m venv myvenv
source myvenv/Scripts/activate
pip install -r requirements.txt
kaggle datasets download -d hphphp123321/tenhou-4-player-riichi-mahjong-dataset -p data/raw
```

### 2. Download and build the dataset

```bash
python data/download_logs.py --year 2023 --out data/raw/
python data/parse_tenhou.py  --in  data/raw/ --out data/parsed/
python data/build_dataset.py --in  data/parsed/ --out data/dataset.npz
```

Parsing ~100k logs takes around 10–15 minutes. The resulting dataset is ~500MB.

### 3. Train

```bash
python model/train.py \
  --dataset data/dataset.npz \
  --epochs 30 \
  --batch-size 512 \
  --out model/checkpoints/
```

Training on CPU takes several hours; on a GPU (~10 min on a T4). Use [Google Colab](https://colab.research.google.com) or [Kaggle Notebooks](https://www.kaggle.com/code) for free GPU access.

### 4. Evaluate

```bash
python model/evaluate.py \
  --checkpoint model/checkpoints/best.pt \
  --dataset data/dataset.npz
```

### 5. Run the API

```bash
uvicorn api.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

### 6. Run the frontend

```bash
cd frontend
npm install
npm run dev
# opens at http://localhost:5173
```

---

## API

```
POST /recommend
```

**Request:**
```json
{
  "hand": ["1m","2m","3m","4p","5p","6p","7s","8s","9s","1z","2z","3z","4z"],
  "dora_indicators": ["5m"],
  "seat_wind": "east",
  "round_wind": "east",
  "turn": 8,
  "opponents": [
    { "discards": ["9m","1z","8p"], "is_riichi": false },
    { "discards": ["7s","2z"], "is_riichi": true },
    { "discards": ["1m","9s"], "is_riichi": false }
  ]
}
```

**Response:**
```json
{
  "recommended_discard": "4z",
  "confidence_scores": {
    "4z": 0.61,
    "3z": 0.18,
    "2z": 0.09,
    "1z": 0.08,
    "...": "..."
  },
  "shanten": 1,
  "opponent_riichi_risk": [0.12, 0.87, 0.21]
}
```

---

## Results

| Model | Discard accuracy | Win rate (1000-game sim) |
|---|---|---|
| Random | 2.9% | baseline |
| Rule-based (tenpai chase) | — | +0% (baseline) |
| Discard model (no opp.) | X% | +X% |
| Discard model + opponent | X% | +X% |

*Fill in after training.*

---

## Tech stack

| Layer | Tech |
|---|---|
| Game logic | [`mahjong`](https://github.com/MahjongRepository/mahjong) (shanten, hand scoring) |
| Data | Python, lxml, pandas, NumPy |
| ML | PyTorch, scikit-learn, SHAP |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | React 18, Vite, Recharts, Tailwind CSS |

---

## Training data

Logs are downloaded from [Tenhou.net](https://tenhou.net) phoenix-level replays — the highest rank tier, meaning every game in the training set was played by expert-level players. This is the same data source used by most published Riichi Mahjong AI research.

---

## License

MIT