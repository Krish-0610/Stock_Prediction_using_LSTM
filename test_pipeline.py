import yfinance as yf
import pandas as pd
from tensorflow import keras
from training.utils import flatten_columns
from features.technical_indicators import add_technical_indicators
from features.macro_features import get_macro_features
from features.flags import build_flag_dataframe
from training.prepare_data import prepare_data
from training.model import build_model_v1
from datetime import datetime
from keras.models import load_model

# ? Should i organize all data into csv file ?

# config
TICKER = "^NSEBANK"
TIME_STEP = 100
TARGET_COLS = ["Close", "Open"]

# 1. Fetch data
now = datetime.now()
start = datetime(now.year - 10, now.month, now.day)
end = now
price_df = yf.download(TICKER, start, end)
assert not price_df.empty, "Price data empty"

# 2. Indicators
price_df = add_technical_indicators(price_df)
price_df = flatten_columns(price_df)
assert "RSI" in price_df.columns, "Indicators missing"

# 3. Macro
macro_df = get_macro_features(price_df.index.min(), price_df.index.max())
macro_df = macro_df.reindex(price_df.index, method="ffill")
macro_df = flatten_columns(macro_df)

# 4. Binary Flags
flag_df = build_flag_dataframe(price_df.index)
flag_df = flag_df.shift(1).fillna(0)
flag_df.columns = flag_df.columns.astype(str)


final_df = pd.concat([price_df, macro_df, flag_df], axis=1)
df = flatten_columns(final_df)
df.dropna(inplace=True)
df = df.rename(columns={f"Open_{TICKER}": "Open", f"Close_{TICKER}": "Close"})

# 6. Features
features = df.columns.tolist()
print(features)

print(df.shape)

# 7. Prepare data
X_train, y_train, X_test, y_test, target_scaler = prepare_data(
    df, features=features, target_cols=TARGET_COLS, time_step=TIME_STEP
)

print("Train shapes:", X_train.shape, y_train.shape)
print("Test shapes:", X_test.shape, y_test.shape)

# 8. Build model
model = build_model_v1(TIME_STEP, X_train.shape[2])


model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.1)
model.summary()
model.save("./models/test/model.keras")

m = load_model("./models/test/model.keras")

y_pred = m.predict(X_test[:5])
print(y_pred.shape)

print("PIPELINE TEST PASSED")

df.to_csv("data.csv", index=False)  # TODO: Change name to ticker index
