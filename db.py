"""
Database layer for the Credit Risk app.

Every prediction made through the Streamlit UI is logged to Postgres:
inputs, the model's output, and metadata (timestamp, model version).
This gives an audit trail, which matters for a credit-decisioning
tool -- you want to be able to answer "what did we tell this
applicant, and why" after the fact.

Connection settings are read from environment variables so the same
code works locally, in Docker Compose, and in a managed Postgres
instance in production:
    DB_HOST      (default: localhost)
    DB_PORT      (default: 5432)
    DB_NAME      (default: credit_risk)
    DB_USER      (default: postgres)
    DB_PASSWORD  (default: postgres)
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "credit_risk")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# pool_pre_ping avoids "server closed the connection" errors after idle periods
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class PredictionLog(Base):
    """One row per prediction made in the app."""

    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Inputs
    age = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)
    job = Column(Integer, nullable=False)
    housing = Column(String, nullable=False)
    saving_accounts = Column(String, nullable=False)
    checking_account = Column(String, nullable=False)
    credit_amount = Column(Float, nullable=False)
    duration = Column(Integer, nullable=False)

    # Output
    predicted_label = Column(String, nullable=False)   # "GOOD" or "BAD"
    probability_good = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)


def init_db():
    """Create tables if they don't already exist. Safe to call on every app start."""
    Base.metadata.create_all(bind=engine)


def log_prediction(*, age, sex, job, housing, saving_accounts, checking_account,
                    credit_amount, duration, predicted_label, probability_good,
                    model_version):
    """Persist one prediction. Returns True on success, False if the DB write failed.

    Failures are swallowed (after logging to stderr) rather than raised, so a
    database outage never breaks the applicant-facing prediction flow -- the
    user still gets their result even if the audit write fails.
    """
    session = SessionLocal()
    try:
        record = PredictionLog(
            age=age,
            sex=sex,
            job=job,
            housing=housing,
            saving_accounts=saving_accounts,
            checking_account=checking_account,
            credit_amount=credit_amount,
            duration=duration,
            predicted_label=predicted_label,
            probability_good=probability_good,
            model_version=model_version,
        )
        session.add(record)
        session.commit()
        return True
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see docstring
        session.rollback()
        print(f"[db] Failed to log prediction: {exc}")
        return False
    finally:
        session.close()