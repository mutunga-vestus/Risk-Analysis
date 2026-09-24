# Credit Risk Modelling

Predict whether a loan applicant is **good** (lower risk) or **bad** (higher risk) credit using the German Credit Data and an Extra Trees classifier. Includes a Streamlit web app, Postgres audit logging, and Docker support.

## Features

- **Model**: Extra Trees Classifier trained on the [German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Web app**: Streamlit UI for entering applicant details and getting instant risk predictions
- **Audit log**: Predictions are stored in Postgres (optional — app still works if the DB is down)
- **Docker**: One-command setup with `docker-compose`

## Project structure

```
RiskModelling/
├── analysis.ipynb          # EDA + model training notebook
├── app.py                  # Streamlit application
├── db.py                   # SQLAlchemy models & logging helpers
├── german_credit_data.csv  # Source dataset
├── extar_trees_credit_model.pkl
├── *_encoder.pkl           # Label encoders for categorical features
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env                    # Local DB credentials (do not commit)
└── README.md
```

## Features used by the model

| Feature            | Type        | Notes                                      |
|--------------------|-------------|--------------------------------------------|
| Age                | Numeric     | Years                                      |
| Sex                | Categorical | male / female                              |
| Job                | Numeric     | Skill level 0–3                            |
| Housing            | Categorical | own / rent / free                          |
| Saving accounts    | Categorical | little / moderate / rich / quite rich / NA |
| Checking account   | Categorical | little / moderate / rich / NA              |
| Credit amount      | Numeric     | Loan amount                                |
| Duration           | Numeric     | Loan duration in months                    |

Target: **Risk** → `good` (1) or `bad` (0)

> **Note:** The original dataset also contains a `Purpose` column. It is currently excluded from the model.

## Quick start (Docker — recommended)

1. Copy environment variables (or edit the existing `.env`):

```bash 
# .env
DB_NAME=credit_risk
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432
```

2. Start everything:

```bash
docker compose up --build
```

3. Open the app: [http://localhost:8501](http://localhost:8501)

- App runs on port **8501**
- Postgres is exposed on host port **5433** (container port 5432)

## Local development (without Docker)

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> For the notebook you also need: `matplotlib`, `seaborn`, `xgboost`

```bash
pip install matplotlib seaborn xgboost
```

### 2. Database (optional)

If you want prediction logging, run a local Postgres instance and set the variables in `.env`.  
If Postgres is unavailable the app still works — it just skips logging and shows a warning.

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

### 4. Explore / retrain (notebook)

```bash
jupyter notebook analysis.ipynb
```

The notebook performs EDA, trains Decision Tree / Random Forest / Extra Trees / XGBoost models, and saves the best Extra Trees model plus label encoders.

## Model artefacts

| File                              | Description                          |
|-----------------------------------|--------------------------------------|
| `extra_trees_credit_model.pkl`    | Trained Extra Trees classifier       |
| `Sex_encoder.pkl`                 | LabelEncoder for Sex                 |
| `Housing_encoder.pkl`             | LabelEncoder for Housing             |
| `Saving accounts_encoder.pkl`     | LabelEncoder for Saving accounts     |
| `Checking account_encoder.pkl`    | LabelEncoder for Checking account    |
| `target_encoder.pkl`              | LabelEncoder for Risk (good/bad)     |

To retrain, run the notebook end-to-end. It will overwrite the `.pkl` files.

## Environment variables

| Variable     | Default      | Description                |
|--------------|--------------|----------------------------|
| `DB_NAME`    | credit_risk  | Postgres database name     |
| `DB_USER`    | postgres     | Postgres user              |
| `DB_PASSWORD`| postgres     | Postgres password          |
| `DB_HOST`    | localhost    | Hostname (use `db` in Docker) |
| `DB_PORT`    | 5432         | Postgres port              |

## API / prediction flow

1. User fills applicant details in the Streamlit sidebar.
2. Categorical fields are transformed with the saved LabelEncoders.
3. The Extra Trees model returns class (GOOD/BAD) and probability.
4. Result is displayed and (if DB is available) written to `prediction_log`.

## Known limitations

- No automated tests yet.

## License

This project uses the publicly available German Credit Data. Use at your own risk for educational / research purposes.
