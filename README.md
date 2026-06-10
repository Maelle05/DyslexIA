# DyslexIA
AI-based Dyslexia Detection using Eye Tracking - Le Wagon final project

---

## Presentation

Dyslexia is a neurological disorder which effects about 5-10% of the total population which amounts to about 700 million worldwide. It is a language-based learning disability. It's symptoms are different for different people. It generally effects the way in which people read and write. Among the total population of people having difficulties with reading, writing, speaking and spellings, about 70-80% suffer from some level of dyslexia. Dyslexia is generally defined in a spectrum of difficulties. The current methods of detecting dyslexia is based on a series of reading, writing and speaking tests. The drawback of this system of testing is that they are quite expensive and not available everywhere.

DyslexIA is an app that can use eye-tracking data from someone reading a text with his webcam and predict if this person is dyslexic or not.

Eye-tracking data is collected (angles yaw/pitch of each iris) using **MediaPipe FaceLandmarker**, send to a **FastAPI** endpoint and classified using a **XGBoost** model trained on clinical research data (see credits).

Project is made using two independent components :

- `dyslexia_client` — web client using Vue 3 + TypeScript + Vite
- `dyslexia_api` — Prediction api using FastAPI + XGBoost model

---

## Architecture

```
┌─────────────────────────────────┐        ┌──────────────────────────────┐
│        dyslexia_client          │        │        dyslexia_api          │
│                                 │        │                              │
│  Vue 3 + Vite + Tailwind CSS    │◄──────►│  FastAPI + XGBoost           │
│  MediaPipe (eye-tracking)       │  HTTP  │  /passage  → texte           │
│  Pinia (state)                  │        │  /text-question → QCM        │
│                                 │        │  /predict  → diagnostic      │
└─────────────────────────────────┘        └──────────────────────────────┘
```

---

## Technical Stack

### Client (`dyslexia_client`)
| Tool | Usage |
|---|---|
| Vue 3 + TypeScript | Framework UI |
| Vite | Bundler |
| Tailwind CSS v4 | Styles |
| Pinia | State management |
| Vue Router | Navigation |
| MediaPipe Tasks Vision | Real-time eye-tracking |
| GSAP + ScrollTrigger | Homepage animations |

### API (`dyslexia_api`)
| Tool | Usage |
|---|---|
| FastAPI | HTTP server |
| XGBoost | Classification model |
| Pandas / NumPy | Data processing |
| Uvicorn | ASGI server |

---

## Installation

### Prerequisites

- Node.js ≥ 20
- Python ≥ 3.10
- npm ≥ 11

### Local client

```bash
make install-dev
```

### API

```bash
make install-dev
```

---

## Environment variables

Copy example file and set the values to your own :

```bash
cd dyslexia_client
cp .env.example .env.development
```

**`.env.example`**
```env
VITE_API_URL=http://localhost:8000
```

---

## How to use this project

### API

```bash
make run_api
```

Check that API server is on : [http://localhost:8000/](http://localhost:8000/) -> should return 200 : { "welcome": "Welcome to DyslexIA API Server" }

### Client

```bash
make run_client
```

Application is available at : [http://localhost:5173](http://localhost:5173)

---

## Structure du projet

```
dyslexIA/
├── dyslexia_client/                # Vue 3 web client
│   ├── src/
│   │   ├── views/
│   │   │   ├── HomeView.vue        # Presentation page (GSAP)
│   │   │   ├── TestView.vue        # Multi-steps test page
│   │   │   └── ResultsView.vue     # Results page
│   │   ├── components/
│   │   │   ├── NavBar.vue
│   │   │   ├── test/
│   │   │   │   ├── CameraDetect.vue     # Step 1 : webcam verification
│   │   │   │   ├── Calibration.vue      # Step 2 : eye-tracking calibration
│   │   │   │   ├── Countdown.vue        # Step 3 : countdown
│   │   │   │   ├── ReadingSession.vue   # Step 4 : reading session
│   │   │   │   ├── Question.vue         # Step 5 : Comprehension question
│   │   │   │   └── PredictionView.vue   # Step 6 : Send tracking file to model and show results
│   │   │   └── chart/
│   │   │       ├── GazeHeatmap.vue      # Reading heatmap
│   │   │       ├── GazeChart.vue        # Yaw/pitch graph over time
│   │   │       └── GazeXYChart.vue      # XY gaze trajectory
│   │   ├── lib/
│   │   │   ├── gazeUtils.ts        # Eye-tracking computations(angles, spheres)
│   │   │   └── aoi.ts              # Area of interest computations
│   │   ├── stores/
│   │   │   └── session.ts          # Pinia store (session data)
│   │   ├── router/
│   │   │   └── index.ts
│   │   └── types/
│   │       └── index.ts            # TypeScript interfaces
│   ├── .env.example
│   └── vite.config.ts
│
└── dyslexia_api/                   # API FastAPI
    ├── api/
    │   └── app.py                  # Endpoints REST
    ├── model/
    │   ├── xgboost.py              # Wrapper XGBoost model
    │   └── xgboost_dyslexia_model_v2.json # Model saved file
    └── processing/
        ├── process_v2.py           # Feature engineering
        ├── text_generation.py      # Text generation
        └── text_question.py        # Question generation
```

---

## User flow

```
1. Webcam        → Allow access + preview
2. Calibration   → Eye sphere detection + calibration over the screen center
3. Countdown      → 3… 2… 1…
4. Reading       → Gaze angles recording (yaw/pitch)
5. Question      → Comprehension question over the text
6. Analysis       → CSV send to → API → XGBoost result
7. Results    → Heatmap + graphs + prediction score and proba
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/healthcheck` | Liveness probe |
| `GET` | `/passage` | LLM Text generation |
| `GET` | `/text-question` | Text generation + question |
| `POST` | `/predict` | Model prediction |

### CSV format needed by `/predict`

```
time,x_left,y_left,x_right,y_right
25,0.85,1.03,-0.88,1.00
43,0.89,1.23,-0.56,1.34
...
```

### Result returned by `/predict`

```json
{
  "Prediction": "Dyslexique",
  "Probability": 0.82
}
```

## Research and Development of the model
R&D notebooks are availble in the main branch.

## Disclaimer
This app was made for french users as a first experiment so the prediction output, text and question generation are in french right now.

Eye-tracking will yield better results with high quality webcams and good lighting.

This application is an experiment and can't replace the diagnostic of a professional. If you have any doubt about you or someone being dyslexic we recommend for you to seek a professional opinion.

## Credits
### Data :
- Dostalova, N., Svaricek, R., Sedmidubsky, J., Culemann, W., Sasinka, C., Zezula, P., & Cenek, J. (2024). ETDD70: Eye-tracking Dyslexia Dataset [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13332134

- Nilsson Benfatto M, Öqvist Seimyr G, Ygge J, Pansell T, Rydberg A, Jacobson C. Screening for Dyslexia Using Eye Tracking during Reading. PLoS One. 2016 Dec 9;11(12):e0165508. doi: 10.1371/journal.pone.0165508. PMID: 27936148; PMCID: PMC5147795.

### Eye-tracking computations
- https://github.com/JEOresearch/EyeTracker/tree/main/Webcam3DTracker

### Made by:
- Maëlle, Justine, Manon et Yoann
- Le Wagon Nantes #Batch-2275
