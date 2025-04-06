from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
with open("flood_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(
    MonsoonIntensity: float = Form(...),
    TopographyDrainage: float = Form(...),
    RiverManagement: float = Form(...),
    Deforestation: float = Form(...),
    Urbanization: float = Form(...),
    ClimateChange: float = Form(...),
    DamsQuality: float = Form(...),
    Siltation: float = Form(...),
    AgriculturalPractices: float = Form(...),
    Encroachments: float = Form(...),
    IneffectiveDisasterPreparedness: float = Form(...),
    DrainageSystems: float = Form(...),
    CoastalVulnerability: float = Form(...),
    Landslides: float = Form(...),
    Watersheds: float = Form(...),
    DeterioratingInfrastructure: float = Form(...),
    PopulationScore: float = Form(...),
    WetlandLoss: float = Form(...),
    InadequatePlanning: float = Form(...),
    PoliticalFactors: float = Form(...)
):
    features = np.array([[MonsoonIntensity, TopographyDrainage, RiverManagement,
                          Deforestation, Urbanization, ClimateChange, DamsQuality,
                          Siltation, AgriculturalPractices, Encroachments,
                          IneffectiveDisasterPreparedness, DrainageSystems,
                          CoastalVulnerability, Landslides, Watersheds,
                          DeterioratingInfrastructure, PopulationScore, WetlandLoss,
                          InadequatePlanning, PoliticalFactors]])
    prediction = model.predict(features)
    return {"prediction": float(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
