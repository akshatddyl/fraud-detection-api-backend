from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import List

# --- NEW IMPORTS for Database ---
from sqlalchemy import create_engine, Column, Float, String, Integer
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
# -------------------------------

# --- App Initialization ---
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An API to detect fraudulent credit card transactions.",
    version="1.0.0"
)

# --- Database Setup ---
# Render provides this env var automatically when you link the DB
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set. Please link a free Postgres database on Render.")

# Fix for Render's/Heroku's "postgres://" URL prefix which SQLAlchemy 1.4+ doesn't like
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DB Model (our new 'table') ---
# This class defines the columns in our "transactions" table
class TransactionHistory(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    Time = Column(Float)
    V1 = Column(Float)
    V2 = Column(Float)
    V3 = Column(Float)
    V4 = Column(Float)
    V5 = Column(Float)
    V6 = Column(Float)
    V7 = Column(Float)
    V8 = Column(Float)
    V9 = Column(Float)
    V10 = Column(Float)
    V11 = Column(Float)
    V12 = Column(Float)
    V13 = Column(Float)
    V14 = Column(Float)
    V15 = Column(Float)
    V16 = Column(Float)
    V17 = Column(Float)
    V18 = Column(Float)
    V19 = Column(Float)
    V20 = Column(Float)
    V21 = Column(Float)
    V22 = Column(Float)
    V23 = Column(Float)
    V24 = Column(Float)
    V25 = Column(Float)
    V26 = Column(Float)
    V27 = Column(Float)
    V28 = Column(Float)
    Amount = Column(Float)
    is_fraud = Column(Integer)
    probability_fraud = Column(Float)
    probability_genuine = Column(Float)

# Create the table on app startup
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

# --- Dependency to get DB session ---
# This helper function gives our endpoints a database connection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Load Model and Scalers ---
# (This is the same as before)
try:
    model = joblib.load('lightgbm_fraud_model.joblib')
    amount_scaler = joblib.load('amount_scaler.joblib')
    time_scaler = joblib.load('time_scaler.joblib')
except FileNotFoundError:
    raise RuntimeError("Model or scaler files not found. Make sure all .joblib files are in the same directory.")


# --- Pydantic Data Models ---
# (This is the same as before)
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResponse(BaseModel):
    is_fraud: int
    probability_fraud: float
    probability_genuine: float

# This is a Pydantic model for *reading* data from the DB
class TransactionHistoryResponse(Transaction):
    id: int
    is_fraud: int
    probability_fraud: float
    probability_genuine: float

    class Config:
        orm_mode = True # This tells Pydantic to read data from our SQL (ORM) object

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to welcome users to the API."""
    return {"message": "Welcome to the Fraud Detection API. Go to /docs for documentation."}

@app.post("/predict/", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud(transaction: Transaction, db: Session = Depends(get_db)): # <-- Inject DB Session
    """
    Predicts if a transaction is fraudulent.
    Receives transaction data, preprocesses it, and returns the fraud prediction.
    Saves the transaction and prediction to the database.
    """
    try:
        # --- Prediction Logic (same as before) ---
        df = pd.DataFrame([transaction.model_dump()])
        df['scaled_amount'] = amount_scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = time_scaler.transform(df['Time'].values.reshape(-1, 1))
        df_processed = df.drop(['Time', 'Amount'], axis=1)
        
        training_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                         'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                         'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                         'scaled_amount', 'scaled_time']
        
        df_processed = df_processed[training_cols]
        prediction = model.predict(df_processed)[0]
        probabilities = model.predict_proba(df_processed)[0]
        
        response = {
            "is_fraud": int(prediction),
            "probability_fraud": float(probabilities[1]),
            "probability_genuine": float(probabilities[0])
        }
        
        # --- NEW: Save to database ---
        # Create a new record using the Pydantic model and the response
        db_record = TransactionHistory(
            **transaction.model_dump(), # Unpacks all fields from the input (Time, V1...V28, Amount)
            is_fraud=response['is_fraud'],
            probability_fraud=response['probability_fraud'],
            probability_genuine=response['probability_genuine']
        )
        
        db.add(db_record)
        db.commit()
        # ---------------------------
        
        return response

    except Exception as e:
        db.rollback() # Rollback the database session on error
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/", response_model=List[TransactionHistoryResponse], tags=["History"])
def get_transaction_history(db: Session = Depends(get_db)):
    """
    Retrieves the list of all transactions and their predictions from the database.
    """
    try:
        # Query the database, order by most recent (descending id)
        history = db.query(TransactionHistory).order_by(TransactionHistory.id.desc()).all()
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
