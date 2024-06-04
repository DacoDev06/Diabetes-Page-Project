from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from schemas.predictData import PredictData
from static.prediction import prediction


app  = FastAPI(title="Diabetes model",version="0.0.1")
import pickle

clf=pickle.load(open('./static/model.pkl','rb'))

app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las orígenes. Cambia esto según tus necesidades.
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc).
    allow_headers=["*"],  # Permitir todos los encabezados.
)


@app.get('/',tags=["page"],response_class=FileResponse)
async def message():
    file_path = os.path.join("templates", "index.html")
    return FileResponse(file_path)


@app.post("/predict", response_class=JSONResponse, response_model=PredictData)
async def predict(Feartures : PredictData):
    data = Feartures.features
    result = prediction(data)
    return JSONResponse(content={"result":result})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=2022)


    