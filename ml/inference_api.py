import os
import torch
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import SurrogateModel


app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

x_scaler = joblib.load(os.path.join(BASE_DIR, "data", "scaler.pkl"))
y_scaler = joblib.load(os.path.join(BASE_DIR, "data", "y_scaler.pkl"))


model = SurrogateModel()

model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "models", "surrogate.pt"),
        map_location="cpu"
    )
)

model.eval()


class CollisionInput(BaseModel):

    particle_A_energy: float
    particle_B_energy: float
    collision_angle: float

    momentum_x: float
    momentum_y: float
    momentum_z: float


@app.post("/predict")

def predict(data: CollisionInput):

    x = np.array([[
        data.particle_A_energy,
        data.particle_B_energy,
        data.collision_angle,
        data.momentum_x,
        data.momentum_y,
        data.momentum_z
    ]])

    x_scaled = x_scaler.transform(x)

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(x_tensor).numpy()

    pred = y_scaler.inverse_transform(pred_scaled)

    return {

        "final_energy_1": float(pred[0][0]),
        "final_energy_2": float(pred[0][1]),
        "sqrt_s": float(pred[0][2])
    }

@app.post("/predict_batch")

def predict_batch(data: list[CollisionInput]):

    X = np.array([[
        d.particle_A_energy,
        d.particle_B_energy,
        d.collision_angle,
        d.momentum_x,
        d.momentum_y,
        d.momentum_z
    ] for d in data])

    X_scaled = x_scaler.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():

        pred_scaled = model(X_tensor).numpy()

    preds = y_scaler.inverse_transform(pred_scaled)

    results = []

    for p in preds:

        results.append({

            "final_energy_1": float(p[0]),
            "final_energy_2": float(p[1]),
            "sqrt_s": float(p[2])

        })

    return results