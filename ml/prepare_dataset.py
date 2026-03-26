import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# path adjust if needed
DATA_PATH = "../particle-collision-simulator/data/raw/collision_events.csv.gz"

df = pd.read_csv(DATA_PATH)

# features (initial state)
X = df[
    [
        "particle_A_energy",
        "particle_B_energy",
        "collision_angle",
        "momentum_x",
        "momentum_y",
        "momentum_z",
    ]
].values

# targets (final state)
y = df[
    [
        "final_energy_1",
        "final_energy_2",
        "sqrt_s",
    ]
].values

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.preprocessing import StandardScaler

y_scaler = StandardScaler()

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

joblib.dump(y_scaler, "data/y_scaler.pkl")

# save processed data
os.makedirs("data", exist_ok=True)

np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)

np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

joblib.dump(scaler, "data/scaler.pkl")

print("dataset ready")
print("train shape:", X_train.shape)
print("test shape:", X_test.shape)