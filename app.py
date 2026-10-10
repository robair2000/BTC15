from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title='Bitcoin 15-Minuten Quant Dashboard',
    page_icon='⚡',
    layout='wide',
)

st.title('⚡ Bitcoin (BTC) 15-Minuten Scalping Quant Model')
st.markdown(
    'Dit dashboard haalt live **15-minuten candles** op via Binance, berekent'
    ' technische indicatoren en voorspelt met XGBoost de exacte prijs over'
    ' **15 minuten**.'
)


# --- 15-MINUTEN DATA OPHAAL VIA BINANCE API ---
@st.cache_data(ttl=300)  # Cache vernieuwen elke 5 minuten
def load_binance_15m_data(symbol='BTCUSDT'):
  try:
    # Binance endpoint voor klines (candles), limit = 1000 candles (ongeveer 10 dagen aan 15m data)
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=1000'
    response = requests.get(url, timeout=10)
    data = response.json()

    # Structuur van Binance kline: [Open time, Open, High, Low, Close, Volume, Close time, ...]
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    df['Date'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
      df[col] = df[col].astype(float)

    df = df[['Date', 'open', 'high', 'low', 'close', 'volume']]
    return df
  except Exception as e:
    st.error(f'Fout bij ophalen van Binance data: {e}')
    return None


# 100% Realtime actuele prijs
def fetch_realtime_binance_price(symbol='BTCUSDT'):
  try:
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
    response = requests.get(url, timeout=3)
    data = response.json()
    if 'price' in data:
      return float(data['price'])
  except Exception:
    pass
  return None


def add_technical_indicators(df):
  # Snellere periodes voor 15m grafieken
  df['SMA_20'] = df['close'].rolling(window=20).mean()
  df['SMA_50'] = df['close'].rolling(window=50).mean()
  df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
  df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
  df['MACD'] = df['EMA_9'] - df['EMA_21']
  df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

  delta = df['close'].diff()
  gain = delta.where(delta > 0, 0.0)
  loss = -delta.where(delta < 0, 0.0)
  avg_gain = gain.rolling(window=14).mean()
  avg_loss = loss.rolling(window=14).mean()
  rs = avg_gain / avg_loss
  df['RSI'] = 100 - (100 / (1 + rs))
  return df


@st.cache_data(ttl=300)
def fetch_crypto_fear_greed():
  try:
    url = 'https://api.alternative.me/fng/?limit=1'
    response = requests.get(url, timeout=5)
    data = response.json()['data']
    return float(data[0]['value'])
  except Exception:
    return 50.0


# --- DATA LADEN EN SAMENVOEGEN ---
with st.spinner('Bezig met ophalen van 15-minuten Binance candles...'):
  df = load_binance_15m_data('BTCUSDT')

  if df is not None and not df.empty:
    df = add_technical_indicators(df)
    current_fng = fetch_crypto_fear_greed()
    df['fear_greed'] = current_fng

    # Target: Sluitingsprijs van de *volgende* 15-minuten candle
    df['Target'] = df['close'].shift(-1)
    df = df.dropna().reset_index(drop=True)
  else:
    st.stop()

# --- SIDEBAR CONFIGURATIE ---
st.sidebar.header('⚙️ Model Configuratie (15m)')
test_size = st.sidebar.slider(
    'Test / Train Splitsing (%)', 10, 40, 20, step=5
)
n_estimators = st.sidebar.slider(
    'XGBoost Bomen (n_estimators)', 50, 300, 100, step=50
)
max_depth = st.sidebar.slider('Maximale Boomdiepte (max_depth)', 3, 10, 4)

features = [
    'close',
    'open',
    'high',
    'low',
    'volume',
    'SMA_20',
    'SMA_50',
    'EMA_9',
    'EMA_21',
    'MACD',
    'MACD_Signal',
    'RSI',
    'fear_greed',
]

X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100.0, shuffle=False
)

# --- MODEL TRAINING ---
model = xgb.XGBRegressor(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42
)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# --- DASHBOARD WEERGAVE ---
realtime_price = fetch_realtime_binance_price('BTCUSDT')
current_price = realtime_price if realtime_price else float(df['close'].iloc[-1])

col1, col2, col3 = st.columns(3)
col1.metric('Huidige BTC Koers (Live)', f'${current_price:,.2f}')
col2.metric('Model Score ($R^2$ op 15m)', f'{score:.2f}')
col3.metric('Laatste 15m RSI', f"{df['RSI'].iloc[-1]:.2f}")

st.markdown('---')

st.subheader('📊 Laatste 15-Minuten Candles & EMA Koerslijn')
st.line_chart(df.tail(100).set_index('Date')[['close', 'EMA_9', 'EMA_21']])

st.subheader('🔍 Belangrijkste factoren voor de 15m voorspelling')
importances = pd.Series(model.feature_importances_, index=features).sort_values(
    ascending=True
)
st.bar_chart(importances)

# Voorspelling voor over 15 minuten
latest_data = X.iloc[-1].values.reshape(1, -1)
predicted_price = model.predict(latest_data)[0]
price_diff = predicted_price - current_price
percentage_diff = (price_diff / current_price) * 100

st.markdown('---')
st.subheader('🔮 Prijsvoorspelling voor de Volgende 15 Minuten')

col_a, col_b = st.columns(2)
col_a.metric(
    'Voorspelde Prijs over 15m',
    f'${predicted_price:,.2f}',
    delta=f'{percentage_diff:+.3f}%',
)

if predicted_price > current_price:
  col_b.success(
      f'**Verwachting: STIJGING (15m) 📈**\n\nHet model verwacht dat de koers'
      f' over 15 minuten hoger staat (${price_diff:+,.2f}).'
  )
else:
  col_b.warning(
      f'**Verwachting: DALING (15m) 📉**\n\nHet model verwacht dat de koers'
      f' over 15 minuten lager staat (${price_diff:+,.2f}).'
  )

st.info(
    '💡 *Dit scalping-model gebruikt rechtstreekse 15-minuten Binance candles'
    ' en snelle EMA/RSI indicatoren om kortetermijnbewegingen te voorspellen.*'
)
