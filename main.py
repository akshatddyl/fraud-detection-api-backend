from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import List

# --- App Initialization ---
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An API to detect fraudulent credit card transactions.",
    version="1.0.0"
)

# --- Load Model and Scalers ---
# These files must be in the same directory as main.py
try:
    model = joblib.load('lightgbm_fraud_model.joblib') # <-- FIXED FILENAME
    amount_scaler = joblib.load('amount_scaler.joblib')
    time_scaler = joblib.load('time_scaler.joblib')
except FileNotFoundError:
    # This error will now be more accurate
    raise RuntimeError("Model or scaler files not found. Make sure 'lightgbm_model.joblib', 'amount_scaler.joblib', and 'time_scaler.joblib' are in the same directory.")


# --- Data Models ---
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
    
# We remove TransactionRecord as we are disabling history

# --- Local Database (REMOVED) ---
# The local CSV file will not work on a free cloud host (ephemeral filesystem).
# We are disabling this feature for the demo.
# DB_FILE = "backend/transaction_history.csv"
# initialize_db()

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """Root endpoint to welcome users to the API."""
    return {"message": "Welcome to the Fraud Detection API. Go to /docs for documentation."}

@app.post("/predict/", response_model=PredictionResponse, tags=["Prediction"])
def predict_fraud(transaction: Transaction):
    """
    Predicts if a transaction is fraudulent.
    Receives transaction data, preprocesses it, and returns the fraud prediction.
    """
    try:
        # Create a DataFrame from the input transaction
        df = pd.DataFrame([transaction.model_dump()])

        # Preprocess: Scale 'Time' and 'Amount' using the *correct* loaded scalers
        # We need to reshape for a single sample
        df['scaled_amount'] = amount_scaler.transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = time_scaler.transform(df['Time'].values.reshape(-1, 1))
        
        # Drop original columns and select features for the model
        df_processed = df.drop(['Time', 'Amount'], axis=1)
        
        # Reorder columns to match the training order (important!)
        # This list must match the columns your LGBM model was trained on.
        # This order is from your original file.
        training_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                         'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                         'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                         'scaled_amount', 'scaled_time']
        
        # Ensure the columns are in the correct order
        df_processed = df_processed[training_cols]

        # Make prediction
        prediction = model.predict(df_processed)[0]
        probabilities = model.predict_proba(df_processed)[0]
        
        response = {
            "is_fraud": int(prediction),
            "probability_fraud": float(probabilities[1]),
            "probability_genuine": float(probabilities[0])
        }
        
        # --- History saving REMOVED ---
        # history_df.to_csv(DB_FILE, mode='a', header=False, index=False)
        
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/", response_model=List[dict], tags=["History"])
def get_transaction_history():
    """
    Retrieves the list of all transactions and their predictions.
    NOTE: This feature is disabled for the cloud demo as the free platform
    does not support persistent local file storage.
    """
    # Instead of reading a CSV, we just return an empty list.
    return []
    # --- Old code commented out ---
    # if not os.path.exists(DB_FILE):
    #     return []
    # try:
    #     df = pd.read_csv(DB_FILE)
    #     return df.to_dict('records')
    # except pd.errors.EmptyDataError:
    #     return [] # Return empty list if the csv is empty
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
