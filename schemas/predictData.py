from pydantic import BaseModel, Field
from typing import Optional


class PredictData(BaseModel):
    features: list 

    