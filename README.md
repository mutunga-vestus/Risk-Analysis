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
## Model performance

Models were trained with 5-fold GridSearchCV (scoring = accuracy) on a train/test split of the German Credit Data. Results on the held-out test set:

| Model            | Test Accuracy | Best params (summary)                                      |
|------------------|---------------|------------------------------------------------------------|
| Decision Tree    | 58.1%         | max_depth=5                                                |
| Random Forest    | 61.9%         | n_estimators=100, min_samples_split=10                     |
| **Extra Trees**  | **64.8%**     | max_depth=10, n_estimators=100, min_samples_leaf=2         |
| XGBoost          | 65.7%         | learning_rate=0.2, max_depth=7, subsample=0.7              |

Extra Trees was selected for the app (saved as `extra_trees_credit_model.pkl`) as a strong ensemble baseline with good balance of performance and simplicity. XGBoost edged it slightly on accuracy but Extra Trees was preferred for the final artefact.

**Notes on metrics**
- Dataset is imbalanced (~70% good / 30% bad credit).
- Only accuracy was used for model selection. In a production credit system you would also track precision/recall for the “bad” class, ROC-AUC, and cost-sensitive metrics (false negatives are typically more expensive).
- No cross-validated AUC, F1, or confusion matrix is currently reported in the notebook.

## Dataset limitations (German Credit Data)

This project uses the well-known [Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (1,000 applicants, ~1973–1975). Important caveats:

| Limitation | Detail |
|------------|--------|
| **Age of data** | Collected ~50 years ago. Credit behaviour, products, and demographics have changed significantly. |
| **Small sample** | Only 1,000 rows → limited statistical power and higher risk of overfitting. |
| **Class imbalance** | ~700 good / 300 bad. Models can achieve decent accuracy by favouring the majority class. |
| **Selection bias** | Only applicants who were granted credit appear. Rejected applicants (and their true outcomes) are missing. |
| **Missing values** | `Saving accounts` and `Checking account` have many NaNs (treated as a category in this project). |
| **Feature subset** | Original data has more attributes (e.g. Purpose). This model uses only 8 features. |
| **Documentation / coding issues** | The widely circulated UCI version has known coding errors (see Grömping 2019 / South German Credit correction). Sex cannot always be reliably recovered from older “personal status & sex” encodings. |
| **Fairness concerns** | Historical credit data often encodes societal biases (gender, age, foreign-worker status, etc.). Models trained on it can amplify those biases. This project does **not** include fairness or bias audits. |
| **Cost asymmetry** | Misclassifying a bad applicant as good is typically more costly than the reverse. The original Statlog cost matrix reflects this (cost 5 vs 1); the current model optimises plain accuracy, not cost. |

**Bottom line:** This is a solid educational / demo project. It is **not** suitable for real credit decisions without modern data, proper validation, fairness analysis, and regulatory review.

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
