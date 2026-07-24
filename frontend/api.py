import requests

BASE_URL = "http://127.0.0.1:8000"


def forecast(data):
    response = requests.post(
        f"{BASE_URL}/forecast/",
        json=data,
    )
    response.raise_for_status()
    return response.json()


def recommend(data):
    response = requests.post(
        f"{BASE_URL}/recommendation/",
        json=data,
    )
    response.raise_for_status()
    return response.json()