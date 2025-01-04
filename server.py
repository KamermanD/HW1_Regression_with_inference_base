from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import pickle
from typing import List
from train_core.pipline import pipeline, convert_to_numeric
from train_core.pipline import target_scaler, remove_anomalies
from typing import Final
from pathlib import Path
import uvicorn
import pandas as pd
import io

MODELS_PATH: Final[str] = Path(__file__).parent / "models_train.pickle"
PREDICT_CSV_PATH: Final[str] = Path(__file__).parent / "predict.csv"

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class FitResponse(BaseModel):
    message: str
    
class Items(BaseModel):
    objects: List[Item]

@app.post("/fit", response_model=FitResponse, tags=["trainer"])
async def fit(file: UploadFile = File(...)):
    
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    contents = await file.read()
    try:
        df_train = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при чтении CSV-файла: {str(e)}")
    
    X_train = df_train.drop('selling_price', axis=1)
    y_train = df_train['selling_price']
    
    X_train = convert_to_numeric(X_train)
    
    X_train_cleaned, y_train_cleaned = remove_anomalies(
    X_train, y_train, thresholds={'km_driven': 600000, 'mileage': 9, 'max_power': 30})
        
    y_train_scaled = target_scaler.fit_transform(y_train_cleaned.values.reshape(-1, 1))
    
    pipeline.fit(X_train_cleaned, y_train_scaled)
    
    model_and_scalers = {
        'pipeline': pipeline,
        'target_scaler': target_scaler
    }
    
    with open(MODELS_PATH, 'wb') as f:
        pickle.dump(model_and_scalers, f)
    
    return FitResponse(
        message=f"Модель обучена и сохранена в model_train.pickle файл")

@app.post("/predict_item")
def predict_item(item: Item) -> float:

    with open(MODELS_PATH, 'rb') as f:
        loaded_data = pickle.load(f)
        
    pipeline_loaded = loaded_data['pipeline']
    target_scaler_loaded = loaded_data['target_scaler']
    
    item_dict  = item.model_dump()
    df_item = pd.DataFrame([item_dict])
    
    X_train = df_item.drop('selling_price', axis=1)
    y_train = df_item['selling_price']
    
    X_train = convert_to_numeric(X_train)
    
    X_train_cleaned, y_train_cleaned = remove_anomalies(
    X_train, y_train, thresholds={'km_driven': 600000, 'mileage': 9, 'max_power': 30})
    
    predictions_scaled = pipeline_loaded.predict(X_train_cleaned)
    predictions = target_scaler_loaded.inverse_transform(predictions_scaled.reshape(-1, 1))
    
    return float(predictions[0][0])

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)) -> List[float]:
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при чтении CSV-файла: {str(e)}")
    
    with open(MODELS_PATH, 'rb') as f:
        loaded_data = pickle.load(f)
        
    pipeline_loaded = loaded_data['pipeline']
    target_scaler_loaded = loaded_data['target_scaler']
    
    X_test = df.drop('selling_price', axis=1)
    y_test = df['selling_price']
    
    X_test = convert_to_numeric(X_test)
    
    X_train_cleaned, y_train_cleaned = remove_anomalies(
    X_test, y_test, thresholds={'km_driven': 600000, 'mileage': 9, 'max_power': 30})
    
    predictions_scaled = pipeline_loaded.predict(X_train_cleaned)
    predictions = target_scaler_loaded.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    df_predictions = pd.DataFrame(predictions, columns=['predictions'])
    df_predictions.to_csv(PREDICT_CSV_PATH, index=True)

    return predictions

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)