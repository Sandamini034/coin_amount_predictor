import pandas as pd
import sqlite3
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from collections import Counter

DB_PATH = os.path.join(os.path.dirname(__file__), './database/supermarket.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'coin_forecast_model.pkl')

COIN_DENOMINATIONS = [10, 5, 2, 1]
DENOM_COLUMNS = [f"count_{d}" for d in COIN_DENOMINATIONS]  

def round_up_to_nearest_10(amount):
    amount = round(amount)
    remainder = amount % 50

    if remainder == 0:
        return amount, 0
    
    rounded = amount + (50- remainder)
    diff = rounded - amount
    return rounded, diff

def greedy_coin_breakdown(remainder):
    breakdown = {}
    remaining = remainder

    for denom in COIN_DENOMINATIONS:
        if remaining <= 0:
            break
        count = remaining // denom
        if count > 0:
            breakdown[denom] = int(count)
            remaining -= count * denom
    return breakdown

def build_daily_coin_table(df):
    df = df.copy()
    #converts the Date column to datetime format to understand in pandas
    df["date"] = pd.to_datetime(df["date"])

    daily_records = []

    for date_val, group in df.groupby(df["date"].dt.date):
        totals = Counter()
        #group by date and create mini data frames 
        for amt in group["net_amount"]:
            #don't need the first value 
            _, remainder = round_up_to_nearest_10(amt)
            totals.update(greedy_coin_breakdown(remainder))

        record = {
            "date" : pd.to_datetime(date_val),
            "total_revenue": group["net_amount"].sum(),
            "transaction_count": group["invoice_no"].nunique(),
        }

        for d in COIN_DENOMINATIONS:
            record[f"count_{d}"] = totals.get(d, 0)
        daily_records.append(record)
    
    daily = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True) 
    return daily 

def add_lag_features(daily):
    daily = daily.copy()
    daily["DayOfWeek"] = daily["date"].dt.dayofweek
    daily["Month"] = daily["date"].dt.month
    daily["DayOfMonth"] = daily["date"].dt.day
    daily["IsWeekend"] = (daily["DayOfWeek"] >= 4).astype(int)

    daily["prev_day_revenue"] = daily["total_revenue"].shift(1)
    daily["prev_day_transactions"] = daily["transaction_count"].shift(1)

    #rolling is done to reduce sensitivity over just one day
    daily["rolling_7day_avg_revenue"] = daily["total_revenue"].shift(1).rolling(7).mean()
    daily["rolling_7day_avg_transactions"] = daily["transaction_count"].shift(1).rolling(7).mean()
    daily["same_dow_last_week_revenue"] = daily["total_revenue"].shift(7)

    for d in COIN_DENOMINATIONS:
        col = f"count_{d}"
        daily[f"prev_day_{col}"] = daily[col].shift(1)
        daily[f"rolling_7day_avg_{col}"] = daily[col].shift(1).rolling(7).mean()

    return daily

FEATURE_COLUMNS = ([
    "DayOfWeek", "Month", "DayOfMonth", "IsWeekend",
    "prev_day_revenue", "prev_day_transactions",
    "rolling_7day_avg_revenue", "rolling_7day_avg_transactions",
    "same_dow_last_week_revenue"
] + [f"prev_day_count_{d}" for d in COIN_DENOMINATIONS] 
  + [f"rolling_7day_avg_count_{d}" for d in COIN_DENOMINATIONS]
)

def train():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT date, net_amount, invoice_no FROM invoice_line_items", conn)
    print(df.columns.tolist())
    conn.close()

    daily = build_daily_coin_table(df)
    daily = add_lag_features(daily)
    #dropna is used to remove any rows with missing values, which can occur due to the lag features
    #then again that is reset to have a clean index
    daily = daily.dropna().reset_index(drop=True)

    if len(daily) < 20:
        print(f"Warning: only {len(daily)} usable rows after adding lag features. "
              f"Predictions will be unreliable until you have more historical days.")

    X = daily[FEATURE_COLUMNS]
    y = daily[DENOM_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    #enumerate is used to get both the index and the value of each element in the DENOM_COLUMNS list
    for i, denom_col in enumerate(DENOM_COLUMNS):
        #[:, i] select all rows and the i-th column of the preds array, which corresponds to the predictions for the current denomination
        mae = mean_absolute_error(y_test[denom_col], preds[:, i])
        print(f"MAE for {denom_col}: {mae:.2f}")
    
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model, daily

#daily parameter is optional if it doesn't get anything then it assigns none
def predict_next_day(daily=None):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    if daily is None:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT date, net_amount, invoice_no FROM invoice_line_items", conn)
        conn.close()
        daily = build_daily_coin_table(df)

    last_date = daily["date"].max()
    next_date = last_date + pd.Timedelta(days=1)
    recent = daily.sort_values("date").tail(7)

    features = {
        "DayOfWeek": next_date.dayofweek,
        "Month": next_date.month,
        "DayOfMonth": next_date.day,
        "IsWeekend": int(next_date.dayofweek >= 4),
        "prev_day_revenue": daily.iloc[-1]["total_revenue"],
        "prev_day_transactions": daily.iloc[-1]["transaction_count"],
        "rolling_7day_avg_revenue": recent["total_revenue"].mean(),
        "rolling_7day_avg_transactions": recent["transaction_count"].mean(),
    }

    # same weekday last week
    same_dow_row = daily[daily["date"] == (next_date - pd.Timedelta(days=7))]
    features["same_dow_last_week_revenue"] = (
        same_dow_row["total_revenue"].values[0] if len(same_dow_row) else recent["total_revenue"].mean()
    )

    for d in COIN_DENOMINATIONS:
        col = f"count_{d}"
        features[f"prev_day_{col}"] = daily.iloc[-1][col]
        features[f"rolling_7day_avg_{col}"] = recent[col].mean()

    X_next = pd.DataFrame([features])[feature_columns]
    predicted = model.predict(X_next)[0]

    result = {denom: max(0, round(count)) for denom, count in zip(COIN_DENOMINATIONS, predicted)}

    print(f"\nPredicted coin needs for {next_date.date()}:")
    for denom in COIN_DENOMINATIONS:
        print(f"Rs.{denom} x {result[denom]}")

    return next_date, result


if __name__ == "__main__":
    model, daily = train()
    predict_next_day(daily)