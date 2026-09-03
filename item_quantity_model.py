import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import math

DB_PATH = os.path.join(os.path.dirname(__file__), "../database/supermarket.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "item_quantity_model.pkl")

FEATURE_COLUMNS = [
    "item_code_encoded",
    "DayOfWeek", "DayOfWeek_sin", "DayOfWeek_cos",
    "Month", "DayOfMonth", "IsWeekend",
    "prev_day_qty", "rolling_7day_avg_qty", "same_dow_last_week_qty"
]

def build_daily_table(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby(["item_name", "date"]).agg(
        total_quantity = ("qty", "sum"),
        transaction_count = ("invoice_no", "nunique")
    ).reset_index()

    return daily

def fill_missing_days(daily):
    #pd.date_range(start, end, freq="D") creates a range of dates from start to end with daily frequency
    #even though the daily table has missing days for some items, want to fill those missing days with 0 quantity and 0 transaction count
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    items = daily["item_name"].unique()

    #MulitiIndex is used to include both item_name and date in the index of the DataFrame
    #from_product creates a cartesian product of the two arrays, so that we have all combinations of item_name and date
    #and store them in a MultiIndex object with the names "item_name" and "date"
    full_index = pd.MultiIndex.from_product([items, all_dates], names=["item_name", "date"])

    #set_index sets the index of the daily DataFrame to be a MultiIndex of item_name and date
    #reindex reindexes the daily DataFrame to have the full_index, filling in missing combinations with NaN
    #reset_index resets the index back to a regular index, and the MultiIndex columns become regular columns again
    daily_full = daily.set_index(["item_name", "date"]).reindex(full_index).reset_index()

    #fillna fills in the NaN values in the total_quantity column with 0
    daily_full["total_quantity"] = daily_full["total_quantity"].fillna(0)

    daily_full["transaction_count"] = daily_full["transaction_count"].fillna(0)

    return daily_full

def add_calendar_features(daily):
    daily = daily.copy()
    daily["DayOfWeek"] = daily["date"].dt.dayofweek
    daily["DayOfWeek_sin"] = np.sin(2 * np.pi * daily["DayOfWeek"] / 7)
    daily["DayOfWeek_cos"] = np.cos(2 * np.pi * daily["DayOfWeek"] / 7)
    daily["Month"] = daily["date"].dt.month
    daily["DayOfMonth"] = daily["date"].dt.day
    daily["IsWeekend"] = (daily["DayOfWeek"] >=5 ).astype(int)
    return daily

def add_lag_features(daily):
    daily = daily.copy()
    daily =  daily.sort_values(["item_name", "date"])

    grp = daily.groupby("item_name")["total_quantity"]

    daily["prev_day_qty"] = grp.shift(1)
    #.reset_index(level=0, drop=True) is used to reset the index of the rolling mean result to match the original daily DataFrame, dropping the item_name level from the index
    daily["rolling_7day_avg_qty"] = (grp.shift(1).rolling(7).mean().reset_index(level=0, drop=True))
    daily["same_dow_last_week_qty"] = grp.shift(7)

    return daily

def encode_items(daily, item_categories=None):
    daily = daily.copy()
    if item_categories is None:
        #converts item_name to a categorical type
        daily["item_name"] = daily["item_name"].astype("category")
        #cat.categories returns the categories of the categorical type, which are the unique item names
        item_categories = daily["item_name"].cat.categories
    else:
        daily["item_name"] = pd.Categorical(daily["item_name"], categories=item_categories)
    
    daily["item_code_encoded"] = daily["item_name"].cat.codes
    return daily, item_categories
    
def prepare_daily (df, item_categories=None):
    daily = build_daily_table(df)
    daily = fill_missing_days(daily)
    daily = add_calendar_features(daily)
    daily = add_lag_features(daily)
    daily, item_categories = encode_items(daily, item_categories)
    return daily, item_categories

def train():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT item_name, date, qty, invoice_no FROM invoice_line_items", conn)
    conn.close()

    daily, item_categories = prepare_daily(df)
    daily = daily.dropna().reset_index(drop=True)   

    print(f"Total item-day rows after cleaning: {len(daily)}")

    X = daily[FEATURE_COLUMNS]
    y = daily["total_quantity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators = 300, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R^2 Score: {model.score(X_test, y_test):.2f}")
    
    joblib.dump({
        "model":model,
        "feature_columns":FEATURE_COLUMNS,
        "item_categories":item_categories
    }, MODEL_PATH)

    return model, daily, item_categories

def predict_next_day_all_items(daily=None, item_categories=None):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    item_categories = saved["item_categories"]

    if daily is None:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT item_name, date, qty, invoice_no FROM invoice_line_items", conn)
        conn.close()
        daily, _ = prepare_daily(df, item_categories=item_categories)

    last_date = daily["date"].max()
    next_date = last_date + pd.Timedelta(days=1)

    results = []
    for item in item_categories:
        item_hist = daily[daily["item_name"] == item].sort_values("date")
        if item_hist.empty:
            continue
        recent = item_hist.tail(7)
        #using iloc grab the last row of the item_hist DataFrame, which contains the most recent data for that item
        last_row = item_hist.iloc[-1]

        same_dow_row = item_hist[item_hist["date"] == (next_date - pd.Timedelta(days=7))]
        same_dow_qty = same_dow_row["total_quantity"].values[0] if len(same_dow_row) else recent ["total_quantity"].mean()

        features = {
            "item_code_encoded": last_row["item_code_encoded"],
            "DayOfWeek": next_date.dayofweek,
            "DayOfWeek_sin": np.sin(2 * np.pi * next_date.dayofweek / 7),
            "DayOfWeek_cos": np.cos(2 * np.pi * next_date.dayofweek / 7),
            "Month": next_date.month,
            "DayOfMonth": next_date.day,
            "IsWeekend": int(next_date.dayofweek >= 5),
            "prev_day_qty": last_row["total_quantity"],
            "rolling_7day_avg_qty": recent["total_quantity"].mean(),
            "same_dow_last_week_qty": same_dow_qty,
        }

        X_next = pd.DataFrame([features])[feature_columns]
        predicted_qty = model.predict(X_next)[0]

        results.append({
            "item_name": item,
            "predicted_date": next_date.date(),
            "predicted_qty": max(0, round(predicted_qty, 2))
        })

    return pd.DataFrame(results).sort_values("predicted_qty", ascending=False)

def predict_next_n_days_for_item(item_name, n_days=30):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    item_categories = saved["item_categories"]

    if item_name not in item_categories:
        return pd.DataFrame([])

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT item_name, date, qty, invoice_no FROM invoice_line_items WHERE item_name = ?",
        conn,
        params=(item_name,)
    )
    conn.close()

    if df.empty:
        return pd.DataFrame([])

    daily, _ = prepare_daily(df, item_categories=item_categories)
    history = daily[daily["item_name"] == item_name].sort_values("date").reset_index(drop=True)
    if history.empty:
        return pd.DataFrame([])

    results = []
    last_date = history["date"].max()

    for _ in range(n_days):
        next_date = last_date + pd.Timedelta(days=1)
        recent = history.tail(7)
        last_row = history.iloc[-1]

        same_dow_row = history[history["date"] == (next_date - pd.Timedelta(days=7))]
        same_dow_qty = (same_dow_row["total_quantity"].values[0]
                        if len(same_dow_row) else recent["total_quantity"].mean())

        features = {
            "item_code_encoded": last_row["item_code_encoded"],
            "DayOfWeek": next_date.dayofweek,
            "DayOfWeek_sin": np.sin(2 * np.pi * next_date.dayofweek / 7),
            "DayOfWeek_cos": np.cos(2 * np.pi * next_date.dayofweek / 7),
            "Month": next_date.month,
            "DayOfMonth": next_date.day,
            "IsWeekend": int(next_date.dayofweek >= 5),
            "prev_day_qty": last_row["total_quantity"],
            "rolling_7day_avg_qty": recent["total_quantity"].mean(),
            "same_dow_last_week_qty": same_dow_qty,
        }

        X_next = pd.DataFrame([features])[feature_columns]
        predicted_qty = max(0, math.ceil(model.predict(X_next)[0]))

        results.append({
            "item_name": item_name,
            "predicted_date": next_date.date(),
            "predicted_qty": predicted_qty
        })

        # append the prediction as a new "actual" row so the next
        # iteration's lag features (prev_day_qty, rolling avg, etc.) update
        new_row = last_row.copy()
        new_row[["date", "total_quantity"]] = [next_date, predicted_qty]
        new_row[["DayOfWeek", "DayOfWeek_sin", "DayOfWeek_cos",
                  "Month", "DayOfMonth", "IsWeekend"]] = [
            next_date.dayofweek, features["DayOfWeek_sin"], features["DayOfWeek_cos"],
            next_date.month, next_date.day, features["IsWeekend"]
        ]
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        last_date = next_date

    return pd.DataFrame(results)

def predict_next_day_for_item(item_name):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    item_categories = saved["item_categories"]

    if item_name not in item_categories:
        return pd.DataFrame([])
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT item_name, date, qty, invoice_no FROM invoice_line_items WHERE item_name = ?",
        conn,
        params=(item_name,)
    )
    conn.close()

    if df.empty:
        return pd.DataFrame([])
    
    daily, _ = prepare_daily(df, item_categories=item_categories)

    last_date = daily["date"].max()
    next_date = last_date + pd.Timedelta(days=1)

    item_hist = daily[daily["item_name"] == item_name].sort_values("date")
    if item_hist.empty:
        return pd.DataFrame([])

    recent = item_hist.tail(7)
    last_row = item_hist.iloc[-1]

    same_dow_row = item_hist[item_hist["date"] == (next_date - pd.Timedelta(days=7))]
    same_dow_qty = same_dow_row["total_quantity"].values[0] if len(same_dow_row) else recent["total_quantity"].mean()

    features = {
        "item_code_encoded": last_row["item_code_encoded"],
        "DayOfWeek": next_date.dayofweek,
        "DayOfWeek_sin": np.sin(2 * np.pi * next_date.dayofweek / 7),
        "DayOfWeek_cos": np.cos(2 * np.pi * next_date.dayofweek / 7),
        "Month": next_date.month,
        "DayOfMonth": next_date.day,
        "IsWeekend": int(next_date.dayofweek >= 5),
        "prev_day_qty": last_row["total_quantity"],
        "rolling_7day_avg_qty": recent["total_quantity"].mean(),
        "same_dow_last_week_qty": same_dow_qty,
    }

    X_next = pd.DataFrame([features])[feature_columns]
    predicted_qty = model.predict(X_next)[0]

    result = pd.DataFrame([{
        "item_name": item_name,
        "predicted_date": next_date.date(),
        "predicted_qty": max(0, math.ceil(predicted_qty))
    }])

    return result

if __name__ == "__main__":
    model, daily, item_categories = train()
    forecast = predict_next_day_all_items(daily, item_categories)
    print("\nPredicted quantity needs for next day, by item:")
    print(forecast.head(20).to_string(index=False))