import os
from pathlib import Path
from dotenv import load_dotenv, set_key
import requests

env_path = Path(".env")
load_dotenv(env_path)

username = os.getenv("EARTHDATA_USERNAME")
password = os.getenv("EARTHDATA_PASSWORD")

if not username or not password:
    raise ValueError("EARTHDATA_USERNAME или EARTHDATA_PASSWORD не найдены в .env файле")

API_URL = "https://appeears.earthdatacloud.nasa.gov/api/"

response = requests.post(f"{API_URL}login", auth=(username, password))
response.raise_for_status()

token = response.json()["token"]

set_key(env_path, "NASA_APP_EARS_TOKEN", token)

print("Ready")