import pandas as pd
import sqlite3
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DB_PATH = os.path.join(os.path.dirname(__file__), './database/supermarket.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cash_float_model.pkl')

FEATURE_COLUMNS = [
    "DayOfWeek", "DayOfWeek_sin", "DayOfWeek_cos",
    "Month", "DayOfMonth", "IsWeekend",
    "prev_day_revenue", "prev_day_transactions",
    "rolling_7day_avg_revenue", "rolling_7day_avg_transactions"
]

def build_daily_table(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    daily = df.groupby("Date").agg(
        total_revenue=("Net_Amount", "sum"),
        transaction_count=("Invoice_No", "nunique")
    ).reset_index()

    daily["DayOfWeek"] = daily["Date"].dt.dayofweek

    #sin/cos allows the model to undersatnd that sunday and monday are close to each other
    daily["DayOfWeek_sin"] = np.sin(2 * np.pi * daily["DayOfWeek"] / 7)
    daily["DayOfWeek_cos"] = np.cos(2 * np.pi * daily["DayOfWeek"] / 7)

    daily["Month"] = daily["Date"].dt.month
    daily["DayOfMonth"] = daily["Date"].dt.day
    daily["IsWeekend"] = (daily["DayOfWeek"] >= 5).astype(int)

    # lagged features so tomorrow's prediction only uses info we already have today
    daily["prev_day_revenue"] = daily["total_revenue"].shift(1)
    daily["prev_day_transactions"] = daily["transaction_count"].shift(1)
    daily["rolling_7day_avg_revenue"] = daily["total_revenue"].shift(1).rolling(7).mean()
    daily["rolling_7day_avg_transactions"] = daily["transaction_count"].shift(1).rolling(7).mean()

    # target: today's cash float need, based on today's actual revenue (fine for training)
    daily["cash_float_needed"] = daily["total_revenue"] * 0.10

    """
    print(
    daily[
        [
            "Date",
           "total_revenue",
           "rolling_7day_avg_revenue"
       ]
    ].head(20).to_string(index=False)
       )
    """

    return daily

def train():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT Date, Net_Amount, Invoice_No FROM transactions", conn)
    conn.close()

    daily = build_daily_table(df)
    daily = daily.dropna().reset_index(drop=True)

    print(daily.head(10))

    X = daily[FEATURE_COLUMNS]
    y = daily["cash_float_needed"]

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #chronological split
    split_index = int(len(daily)*0.8)

    train_data = daily.iloc[:split_index]
    test_data = daily.iloc[split_index:]

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["cash_float_needed"]

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["cash_float_needed"]

    #500 decision trees, n_jobs=-1 to use all CPU cores, random_state for reproducibility
    #min_samples_leaf=2 to prevent overfitting, max_features="sqrt" to get the best features for each tree
    model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Model R^2 Score: {model.score(X_test, y_test):.3f}")

    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model, daily

def predict_next_day(daily=None):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    if daily is None:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT Date, Net_Amount, Invoice_No FROM transactions", conn)
        conn.close()
        daily = build_daily_table(df)

    last_date = daily["Date"].max()
    next_date = last_date + pd.Timedelta(days=1)
    recent = daily.sort_values("Date").tail(7)

    features = {
        "DayOfWeek": next_date.dayofweek,
        "DayOfWeek_sin": np.sin(2 * np.pi * next_date.dayofweek / 7),
        "DayOfWeek_cos": np.cos(2 * np.pi * next_date.dayofweek / 7),
        "Month": next_date.month,
        "DayOfMonth": next_date.day,
        "IsWeekend": int(next_date.dayofweek >= 5),
        #iloc is used to select data by index number
        "prev_day_revenue": daily.iloc[-1]["total_revenue"],
        "prev_day_transactions": daily.iloc[-1]["transaction_count"],
        "rolling_7day_avg_revenue": recent["total_revenue"].mean(),
        "rolling_7day_avg_transactions": recent["transaction_count"].mean(),
    }

    X_next = pd.DataFrame([features])[feature_columns]
    predicted_cash = model.predict(X_next)[0]

    print(f"\nPredicted cash float needed for {next_date.date()}: Rs.{predicted_cash:.2f}")

    return next_date, round(float(predicted_cash), 2)

if __name__ == "__main__":
    model, daily = train()
    predict_next_day(daily)