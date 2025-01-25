# main.py
from fastapi import FastAPI, HTTPException, status
from datetime import datetime, timezone
import requests
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------
# Configuration
# ---------------------------
BINANCE_API = os.getenv("BINANCE_API", "https://api.binance.com/api/v3")
CACHE_TTL = int(os.getenv("CACHE_TTL", 120))  # 2 minutes caching
ALLOWED_INTERVALS = ["1h", "4h", "1d", "1w"]
INTERVAL_HOURS = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}

# Caching setup
SYMBOLS_CACHE = TTLCache(maxsize=1, ttl=CACHE_TTL)
OHLCV_CACHE = TTLCache(maxsize=500, ttl=300)
PREDICTION_CACHE = TTLCache(maxsize=100, ttl=600)

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(
    title="CryptoPredict Pro+",
    description="Advanced Crypto Analytics & Predictions",
    version="0.2.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Enhanced Prediction Model
# ---------------------------
class AdvancedPredictor:
    def __init__(self):
        self.models = {
            "linear": LinearRegression(),
            "moving_avg": self._create_moving_avg_model()
        }
        
    def _create_moving_avg_model(self):
        """Simple moving average model"""
        class MovingAverage:
            def predict(self, prices, window=3):
                return np.convolve(prices, np.ones(window)/window, mode='valid')[-1]
        return MovingAverage()
    
    def _calculate_confidence(self, prices):
        """Multi-factor confidence calculation"""
        volatility = np.std(prices[-24:]) / np.mean(prices[-24:])
        trend_strength = np.polyfit(range(len(prices)), prices, 1)[0]
        return max(0.4, min(0.95, 1 - volatility + abs(trend_strength*100)))

    def predict(self, prices: List[float], interval: str) -> dict:
        """Enhanced prediction with multiple models"""
        if len(prices) < 48:
            raise ValueError("Insufficient historical data")
            
        X = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices)
        
        self.models["linear"].fit(X, y)
        lr_pred = self.models["linear"].predict([[len(X)]])[0]
        ma_pred = self.models["moving_avg"].predict(y)
        
        return {
            "prediction": round(float((lr_pred * 0.6) + (ma_pred * 0.4)), 4),
            "confidence": round(float(self._calculate_confidence(prices)), 4),
            "interval": interval,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

predictor = AdvancedPredictor()

# ---------------------------
# Binance API Client (Enhanced)
# ---------------------------
class BinanceClient:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_symbols(self) -> List[str]:
        """Get active trading pairs with retry logic"""
        response = requests.get(f"{BINANCE_API}/exchangeInfo")
        data = response.json()
        return [s["symbol"] for s in data["symbols"] if s["status"] == "TRADING"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict]:
        """Fetch OHLCV data with structured response"""
        cache_key = f"{symbol}_{interval}"
        if cache_key in OHLCV_CACHE:
            return OHLCV_CACHE[cache_key]
            
        response = requests.get(
            f"{BINANCE_API}/klines?symbol={symbol}&interval={interval}&limit={limit}"
        )
        data = response.json()
        
        ohlcv = [{
            "timestamp": entry[0],
            "open": float(entry[1]),
            "high": float(entry[2]),
            "low": float(entry[3]),
            "close": float(entry[4]),
            "volume": float(entry[5]),
        } for entry in data]
        
        OHLCV_CACHE[cache_key] = ohlcv
        return ohlcv

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_current_price(self, symbol: str) -> float:
        """Get current price with caching"""
        if symbol in OHLCV_CACHE:
            return OHLCV_CACHE[symbol][-1]["close"]
            
        response = requests.get(f"{BINANCE_API}/ticker/price?symbol={symbol}")
        data = response.json()
        return float(data["price"])

binance = BinanceClient()

# ---------------------------
# Enhanced API Endpoints
# ---------------------------
@app.get("/")
async def root():
    return {
        "name": "CryptoPredict Pro+",
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat()
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

@app.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str, 
    interval: str = "1h",
    limit: int = 100
):
    """Get OHLCV data for charting"""
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid interval. Allowed: {ALLOWED_INTERVALS}"
        )
        
    try:
        ohlcv = binance.fetch_ohlcv(symbol, interval, limit)
        return {
            "symbol": symbol,
            "interval": interval,
            "data": ohlcv,
            "count": len(ohlcv),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Data fetch failed: {str(e)}"
        )

@app.get("/predict/{symbol}")
async def predict_price(
    symbol: str,
    interval: str = "1h",
    prediction_window: int = 24
):
    """Enhanced prediction endpoint"""
    try:
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid interval. Allowed: {ALLOWED_INTERVALS}"
            )
            
        ohlcv = binance.fetch_ohlcv(symbol, interval)
        closes = [entry["close"] for entry in ohlcv]
        prediction = predictor.predict(closes, interval)
        
        return {
            "symbol": symbol,
            "interval": interval,
            "current_price": closes[-1],
            "prediction_window": f"{prediction_window * INTERVAL_HOURS[interval]}h",
            **prediction
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