from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

db = []

class City(BaseModel):
    name: str
    timezone: str


@app.get('/')
def root():
    return {"hello world"}


def get_cities():
    pass

def get_city(city_id: int):
    pass