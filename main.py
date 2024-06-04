
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

from .predict import u


app  = FastAPI(title="Diabetes model",version="0.0.1")

app.mount("/static", StaticFiles(directory="static"), name="static")




@app.get('/',tags=["page"],response_class=FileResponse)
async def message():
    file_path = os.path.join("templates", "index.html")
    return FileResponse(file_path)
    