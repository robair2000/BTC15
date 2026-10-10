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
    'Dit geavanceerde 15-minuten dashboard combineert **Binance orderflow**'
    ' (Taker Buy Ratio), **VWAP**, **Bollinger Bands %B**, **ATR** en'
    ' **technische indicatoren** om de prijs over 15 minuten te voorspellen.'
)


# --- 15-MINUTEN DATA OPHAAL VIA BINANCE API ---
@st.cache_data(ttl=300)
def load_binance_15m_data(symbol='BTCUSDT'):
  try:
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=1000'
    response = requests.get(url, timeout=10)
    data = response.json()

    df = pd.DataFrame(
        data,
        columns=[
            'open_time',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'close_time',
            'quote_asset_volume',
            'number_of_trades',
            'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume',
            'ignore',
        ],
    )

    df['Date'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in [
        'open',
        'high',
        'low',
        'close',
        'volume',
        'taker_buy_base_asset_volume',
    ]:
      df[col] = df[col].astype(float)

    df = df[[
        'Date',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'taker_buy_base_asset_volume',
    ]]
    return df
  except Exception as e:
    st.error(f'Fout bij ophalen van Binance data: {e}')
    return None


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


def add_advanced_indicators(df):
  # 1. Taker Buy Ratio (Koopdruk vs Verkoopdruk)
  df['taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / (
      df['volume'] + 1e-8
  )

  # 2. VWAP (Volume Weighted Average Price) - Rollend over 96 candles (= 24 uur op 15m)
  typical_price = (df['high'] + df['low'] + df['close']) / 3
  df['VWAP'] = (typical_price * df['volume']).rolling(window=96).sum() / (
      df['volume'].rolling(window=96).sum() + 1e-8
  )
  df['VWAP_distance'] = (df['close'] - df['VWAP']) / (df['VWAP'] + 1e-8)

  # 3. Bollinger Bands %B (Intraday positie in de banden)
  sma_20 = df['close'].rolling(window=20).mean()
  std_20 = df['close'].rolling(window=20).std()
  upper_band = sma_20 + (std_20 * 2)
  lower_band = sma_20 - (std_20 * 2)
  df['BB_pct_b'] = (df['close'] - lower_band) / (
      upper_band - lower_band + 1e-8
  )

  # 4. ATR (Average True Range) voor volatiliteit
  high_low = df['high'] - df['low']
  high_close = np.abs(df['high'] - df['close'].shift())
  low_close = np.abs(df['low'] - df['close'].shift())
  true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
  df['ATR'] = true_range.rolling(window=14).mean()

  # 5. Klassieke indicatoren
  df['SMA_20'] = sma_20
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
with st.spinner('Bezig met ophalen en berekenen van 15m orderflow data...'):
  df = load_binance_15m_data('BTCUSDT')

  if df is not None and not df.empty:
    df = add_advanced_indicators(df)
    current_fng = fetch_crypto_fear_greed()
    df['fear_greed'] = current_fng

    # Target: Sluitingsprijs over 15 minuten
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
    'taker_buy_ratio',
    'VWAP_distance',
    'BB_pct_b',
    'ATR',
    'SMA_20',
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
col3.metric(
    'Laatste Taker Buy Ratio', f"{df['taker_buy_ratio'].iloc[-1]:.3f}"
)

st.markdown('---')

st.subheader('📊 Laatste 15-Minuten Candles & VWAP Afstand')
st.line_chart(df.tail(100).set_index('Date')[['close', 'VWAP']])

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
    '💡 *Dit scalping-model gebruikt orderflow (Taker Buy Ratio), VWAP-afwijkingen'
    ' en Bollinger Bands om nauwkeurig de koersrichting over 15 minuten te'
    ' voorspellen.*'
)
