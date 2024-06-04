from pydantic import BaseModel, Field
from typing import Optional


class PredictData(BaseModel):
    features: list 

# class PredictData(BaseModel):
#     age:int
#     genre:int
#     Polyuria :int
#     Polydipsia :int
#     Weight :int
#     Weakness :int
#     Polyphagia :int
#     Thrush :int
#     Blurring :int
#     Itching :int
#     Irritability :int
#     Healing :int
#     Paresis :int
#     Stiffness :int
#     Alopecia :int
#     Obesity :int
    