# main.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from datetime import datetime
import requests
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, List
from sklearn.linear_model import LinearRegression
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------
# Configuration
# ---------------------------
BINANCE_API = os.getenv("BINANCE_API", "https://api.binance.com/api/v3")
CACHE_TTL = int(os.getenv("CACHE_TTL", 300))  # 5 minutes caching
SYMBOLS_CACHE = TTLCache(maxsize=1, ttl=CACHE_TTL)
PRICE_CACHE = TTLCache(maxsize=100, ttl=60)  # 1 minute cache for prices

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(
    title="CryptoPredict Pro",
    description="Real-time Crypto Prices & Predictions",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Prediction Model
# ---------------------------
class PredictionModel:
    def __init__(self):
        self.model = LinearRegression()
        
    def train_and_predict(self, prices: List[float], lookback_hours: int = 48) -> dict:
        """Simple linear regression prediction"""
        if len(prices) < lookback_hours:
            raise ValueError("Insufficient historical data")
            
        X = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices)
        self.model.fit(X, y)
        
        prediction = self.model.predict([[len(prices)]])[0]
        confidence = max(0.5, min(0.9, 1 - (np.std(y[-24:]) / y[-1])))  # Volatility-based confidence
        
        return {
            "prediction": round(float(prediction), 4),
            "confidence": round(float(confidence), 4)
        }

predictor = PredictionModel()

# ---------------------------
# Binance API Client
# ---------------------------
class BinanceClient:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_symbols(self) -> List[str]:
        """Get active trading pairs with retry logic"""
        response = requests.get(f"{BINANCE_API}/exchangeInfo")
        data = response.json()
        return [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_price_data(self, symbol: str, hours: int) -> List[float]:
        """Fetch historical prices with retry logic"""
        limit = min(max(hours * 2, 24), 168)  # 1 week max data
        response = requests.get(
            f"{BINANCE_API}/klines?symbol={symbol}&interval=1h&limit={limit}"
        )
        return [float(entry[4]) for entry in response.json()]

binance = BinanceClient()

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/")
async def root():
    return {
        "name": "CryptoPredict Pro",
        "status": "online",
        "timestamp": datetime.utcnow()
    }

@app.get("/symbols")
async def get_active_symbols():
    """Get all active trading pairs from Binance"""
    try:
        if not SYMBOLS_CACHE.get("symbols"):
            SYMBOLS_CACHE["symbols"] = binance.fetch_symbols()
        return {"symbols": SYMBOLS_CACHE["symbols"]}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Symbols service unavailable: {str(e)}"
        )

@app.get("/price/{symbol}")
async def get_current_price(symbol: str):
    """Get current price with caching"""
    try:
        if symbol not in PRICE_CACHE:
            response = requests.get(f"{BINANCE_API}/ticker/price?symbol={symbol}")
            data = response.json()
            PRICE_CACHE[symbol] = float(data["price"])
            
        return {
            "symbol": symbol,
            "price": PRICE_CACHE[symbol],
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Price check failed: {str(e)}"
        )

@app.get("/predict/{symbol}")
async def predict_price(symbol: str, hours: int = 24):
    """Price prediction endpoint"""
    try:
        # Validate input
        if hours < 1 or hours > 48:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hours must be between 1 and 48"
            )
            
        # Fetch data
        prices = binance.fetch_price_data(symbol, hours)
        
        # Generate prediction
        prediction = predictor.train_and_predict(prices)
        
        return {
            "symbol": symbol,
            "current_price": prices[-1],
            "prediction_window_hours": hours,
            **prediction,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction failed: {str(e)}"
        )

# ---------------------------
# Run Server
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        access_log=False
    )