# DyslexIA - client 👁️

Outil de dépistage de la dyslexie par eye-tracking, basé sur l'analyse des mouvements oculaires durant la lecture.

---

## Présentation

DyslexIA analyse en temps réel les mouvements oculaires d'un utilisateur durant une session de lecture. Les données de regard (angles yaw/pitch de chaque iris) sont collectées via **MediaPipe FaceLandmarker**, envoyées à une API **FastAPI**, et classifiées par un modèle **XGBoost** entraîné sur des patterns oculaires cliniques.

Le projet est composé de deux parties indépendantes :

- `dyslexia_client` — interface web Vue 3 + TypeScript + Vite
- `dyslexia_api` — API de prédiction FastAPI + modèle ML

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

## Stack technique

### Client (`dyslexia_client`)
| Outil | Rôle |
|---|---|
| Vue 3 + TypeScript | Framework UI |
| Vite | Bundler |
| Tailwind CSS v4 | Styles |
| Pinia | State management |
| Vue Router | Navigation |
| MediaPipe Tasks Vision | Eye-tracking temps réel |
| GSAP + ScrollTrigger | Animations page d'accueil |

### API (`dyslexia_api`)
| Outil | Rôle |
|---|---|
| FastAPI | Serveur HTTP |
| XGBoost | Modèle de classification |
| Pandas / NumPy | Traitement des données |
| Uvicorn | Serveur ASGI |

---

## Installation

### Prérequis

- Node.js ≥ 20
- Python ≥ 3.10
- npm ≥ 11

### Client local

```bash
make install-dev
```

### API

```bash
make install-dev
```

---

## Variables d'environnement

Copie le fichier d'exemple et renseigne les valeurs :

```bash
cd dyslexia_client
cp .env.example .env.development
```

**`.env.example`**
```env
VITE_API_URL=http://localhost:8000
```

---

## Lancer le projet

### API

```bash
make run_api
```

Vérifier que l'API tourne : [http://localhost:8000/](http://localhost:8000/)

### Client

```bash
make run_client
```

L'application est disponible sur : [http://localhost:5173](http://localhost:5173)

---

## Structure du projet

```
dyslexIA/
├── dyslexia_client/                # Interface web Vue 3
│   ├── src/
│   │   ├── views/
│   │   │   ├── HomeView.vue        # Page de présentation (GSAP)
│   │   │   ├── TestView.vue        # Page de test multi-steps
│   │   │   └── ResultsView.vue     # Page de résultats
│   │   ├── components/
│   │   │   ├── NavBar.vue
│   │   │   ├── test/
│   │   │   │   ├── CameraDetect.vue     # Step 1 : vérification caméra
│   │   │   │   ├── Calibration.vue      # Step 2 : calibration oculaire
│   │   │   │   ├── Countdown.vue        # Step 3 : décompte
│   │   │   │   ├── ReadingSession.vue   # Step 4 : session de lecture
│   │   │   │   ├── Question.vue         # Step 5 : QCM de compréhension
│   │   │   │   └── PredictionView.vue   # Step 6 : envoi + résultat
│   │   │   └── chart/
│   │   │       ├── GazeHeatmap.vue      # Heatmap de lecture par mot
│   │   │       ├── GazeChart.vue        # Graphe yaw/pitch dans le temps
│   │   │       └── GazeXYChart.vue      # Trajectoire XY du regard
│   │   ├── lib/
│   │   │   ├── gazeUtils.ts        # Calculs eye-tracking (angles, sphères)
│   │   │   └── aoi.ts              # Mesure des zones d'intérêt (mots)
│   │   ├── stores/
│   │   │   └── session.ts          # Pinia store (données de session)
│   │   ├── router/
│   │   │   └── index.ts
│   │   └── types/
│   │       └── index.ts            # Interfaces TypeScript
│   ├── .env.example
│   └── vite.config.ts
│
└── dyslexia_api/                   # API FastAPI
    ├── api/
    │   └── app.py                  # Endpoints REST
    ├── model/
    │   ├── xgboost.py              # Wrapper modèle XGBoost
    │   └── xgboost_dyslexia_model_v2.json
    └── processing/
        ├── process_v2.py           # Feature engineering
        ├── text_generation.py      # Génération de texte
        └── text_question.py        # Génération de QCM
```

---

## Flux utilisateur

```
1. Caméra        → Vérification accès + preview
2. Calibration   → Lock des sphères oculaires + calibration centre écran
3. Décompte      → 3… 2… 1…
4. Lecture       → Enregistrement des angles de regard (yaw/pitch)
5. Question      → QCM de compréhension du texte lu
6. Analyse       → Envoi CSV → API → prédiction XGBoost
7. Résultats     → Heatmap + graphes + score de prédiction
```

---

## API

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/healthcheck` | Liveness probe |
| `GET` | `/passage` | Génère un texte de lecture |
| `GET` | `/text-question` | Génère un texte + QCM associé |
| `POST` | `/predict` | Prédiction dyslexie depuis CSV de regard |

### Format CSV attendu par `/predict`

```
time,angle1_l,angle2_l,angle1_r,angle2_r
25,0.85,1.03,-0.88,1.00
43,0.89,1.23,-0.56,1.34
...
```

### Réponse `/predict`

```json
{
  "Prediction": "Dyslexique",
  "Probability": 0.82
}
```
