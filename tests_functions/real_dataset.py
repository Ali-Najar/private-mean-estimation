import kagglehub

# Download latest version
path = kagglehub.dataset_download("priyamchoksi/credit-card-transactions-dataset")

print("Path to dataset files:", path)

import pandas as pd
import glob
import os

# Find all csv files inside the downloaded dataset
files = glob.glob(os.path.join(path, "*.csv"))

# If there’s more than one file, concat them
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


df_clean = df[["cc_num", "trans_date_trans_time", "amt"]].rename(
    columns={
        "cc_num": "user_id",
        "trans_date_trans_time": "event_time",
        "amt": "amount"
    }
)

# 2) Convert timestamps to datetime
df_clean["event_time"] = pd.to_datetime(df_clean["event_time"])

# 3) Sort chronologically (all users mixed together)
df_sorted = df_clean.sort_values("event_time").reset_index(drop=True)

print(df_sorted.head())
print(df_sorted.tail())

events_sorted = df_clean.sort_values("event_time").reset_index(drop=True)

# Get unique user IDs
unique_users = df_clean["user_id"].unique()

# Map to integer IDs (0..N-1)
user_id_map = {old: new for new, old in enumerate(unique_users)}

# Apply mapping
df_clean["user_id_int"] = df_clean["user_id"].map(user_id_map)


df_small = df_clean[["user_id_int", "amount"]].rename(
    columns={"user_id_int": "user_id"}
)

import numpy as np

n = len(df_small)

running_mean = np.cumsum(df_small['amount'].to_numpy()) / np.arange(1, n + 1)


import matplotlib.pyplot as plt

plt.plot(running_mean[:20000])
plt.ylim(64, 72)   # set y-axis range
plt.show()