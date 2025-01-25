# CryptoPredict Pro+ API

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)

Advanced cryptocurrency price prediction API with real-time data from Binance. Built with FastAPI for high performance and scalability.

## Features

- 📈 **Real-time OHLCV Data** - Historical candlestick data for charting
- 🔮 **Price Predictions** - Combined linear regression + moving average model
- ⚡ **Caching System** - Reduced Binance API calls with TTLCache
- 🕒 **Multiple Timeframes** - 1h, 4h, 1d, and 1w intervals
- 🌐 **CORS Enabled** - Ready for frontend integration

## Installation

1. Clone repository:
```bash
git clone https://github.com/chumbacash/crypto-data-api.git
cd crypto-data-api
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Start server:
```bash
uvicorn main:app --reload
```

## Configuration

Environment variables (create ```.env``` file):

```bash
BINANCE_API=https://api.binance.com/api/v3
CACHE_TTL=300  # 5 minutes
PORT=8000
```

### API Endpoints

#### Get Active Symbols
```bash
GET /symbols
```

#### Response:
```bash
{
  "symbols": ["BTCUSDT", "ETHUSDT", ...]
}
```

### Get Historical Data
```bash
GET /historical/{symbol}?interval=1h&limit=100
```

#### Parameters:

```interval```: 1h, 4h, 1d, 1w
```limit```: Number of candles (max 500)

### Get Price Prediction
```bash
GET /predict/{symbol}?interval=1h&prediction_window=24
```

#### Response:

```bash
{
  "symbol": "BTCUSDT",
  "interval": "1h",
  "current_price": 50234.56,
  "prediction_window": "24h",
  "prediction": 49876.54,
  "confidence": 0.87,
  "last_updated": "2023-08-20T14:30:00+00:00"
}
```

### Example Usage
```bash
# Get BTC/USDT prediction for next 24 hours
curl http://localhost:8000/predict/BTCUSDT

# Get last 100 4-hour candles
curl http://localhost:8000/historical/BTCUSDT?interval=4h
```


### Future Enhancements:

 - WebSocket support for real-time updates

 - User authentication with API keys

 - Technical indicators (RSI, MACD) endpoint

 - Enhanced machine learning models

## Disclaimer:

Predictions are for educational purposes only. Cryptocurrency trading carries substantial risk, Chumba Labs / Chumbacash is not liable for your decisions.