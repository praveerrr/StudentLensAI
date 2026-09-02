# StudentLens AI

AI-powered academic intelligence platform — predicts student final scores, flags
at-risk students, and visualizes performance analytics.

**Live demo (frontend, standalone):** https://studentlens-ai.vercel.app

## What's in this repo

```
studentlens-ai-project/
  index.html          <- standalone frontend (HTML+CSS+JS, one file, no backend needed)
  backend/             <- real REST API + SQLite database + scikit-learn models
    main.py            <- FastAPI app (all endpoints)
    database.py         <- SQLAlchemy engine/session
    models.py           <- SQLAlchemy ORM model (students table)
    schemas.py           <- Pydantic request/response schemas
    ml.py                 <- dataset generation + training pipeline
    requirements.txt
  README.md
  .gitignore
```

There are now **two independent ways to run this project**, described below.

---

## Option A — Standalone frontend (zero setup)

`index.html` is fully self-contained: it generates its own dataset and trains
its own Linear Regression + Random Forest models **in the browser**, in plain
JavaScript, with no server and no dependencies. This is what's deployed at
the live demo link above.

```bash
# just open it
open index.html          # macOS
# or
python3 -m http.server 8000   # then visit http://localhost:8000
```

Use this if you just want to demo the UI with zero setup.

---

## Option B — Real backend (FastAPI + SQLite + scikit-learn)

The `backend/` folder is a genuine client-server implementation: a persistent
SQLite database, a real REST API, and models trained with scikit-learn
instead of hand-rolled browser JS. Every endpoint below was tested end-to-end
(including validation errors and 404s) before being included here.

### Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

On first run it automatically generates 1,000 synthetic student records,
stores them in `studentlens.db` (created automatically), and trains both
models. Interactive API docs are at **http://localhost:8000/docs**.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Server + model status |
| GET | `/api/students` | Paginated, searchable, filterable, sortable student list |
| GET | `/api/students/{id}` | Single student + live AI-predicted score |
| GET | `/api/dashboard/summary` | KPIs: total students, average score, high-risk count, model accuracy |
| GET | `/api/dashboard/trends` | Average score by semester |
| GET | `/api/analytics/correlation` | Pearson correlation matrix (optionally filtered by department/semester) |
| GET | `/api/analytics/feature-importance` | Random Forest feature importances |
| POST | `/api/predict` | Predict a score from 7 input features (validated with Pydantic) |
| GET | `/api/model/performance` | MAE/RMSE/R² for both models + predicted-vs-actual/residual points |
| POST | `/api/model/retrain` | Retrain both models on the current database contents |
| POST | `/api/dataset/regenerate?n=1000` | Wipe the DB and generate a fresh synthetic dataset |
| POST | `/api/dataset/upload` | Upload a CSV (multipart form file) to replace the dataset and retrain |
| GET | `/api/predictions/export` | Download a CSV of actual vs. predicted scores for every student |

Risk levels (Low/Medium/High) are computed the same way as the frontend:
percentile-based bands (bottom 15% / next 30% / top 55%) recomputed every
time the dataset changes, so the risk views stay meaningfully populated
regardless of the score distribution.

### CSV upload format

```
Student_ID, Study_Hours, Attendance, Previous_Score, Assignment_Score,
Internal_Marks, Sleep_Hours, Participation, Final_Score
```

```bash
curl -X POST http://localhost:8000/api/dataset/upload \
  -F "file=@your_data.csv"
```

### Notes on this implementation

- CORS is wide open (`allow_origins=["*"]`) for easy local development —
  tighten this before exposing it publicly.
- Models are trained once at startup and again on-demand via `/retrain` or
  after a dataset change — never inside a hot request path.
- The database file (`backend/studentlens.db`) is created automatically and
  git-ignored; delete it any time to reset to a fresh seeded dataset on next
  startup.

---

## Deploying either piece

- **Frontend** (`index.html`): any static host — Vercel, Netlify, GitHub Pages.
- **Backend** (`backend/`): any Python host that runs ASGI apps — Render,
  Railway, Fly.io, a VPS with `uvicorn`/`gunicorn`, etc. The frontend and
  backend are independent today; wiring the frontend to call this API instead
  of computing client-side is a natural next step if you want one deployed
  full-stack app instead of two separate pieces.
