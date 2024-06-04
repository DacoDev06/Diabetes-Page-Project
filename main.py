# from flask import Flask, request, jsonify, send_file



# app = Flask(__name__)

# # Cargar el modelo desde el archivo .pkl


# @app.route('/')
# def index():
#     return send_file('./templates/index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
    

# if __name__ == '__main__':
#     app.run(debug=True)



from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from schemas.predictData import PredictData
from Predict import predict


app  = FastAPI(title="Diabetes model",version="0.0.1")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get('/',tags=["page"],response_class=FileResponse)
async def message():
    file_path = os.path.join("templates", "index.html")
    return FileResponse(file_path)


@app.post("/predict", response_class=JSONResponse, response_model=PredictData)
def predict(Feartures : PredictData):
    data = Feartures.features
    return JSONResponse(content={"result":data})
    
    data = Features.features
    prediction = predict([data])


    