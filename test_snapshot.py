import os
import requests
from dotenv import load_dotenv

load_dotenv()

ticker = "AAPL"
api_key = os.environ.get("FINANCIAL_DATASETS_API_KEY")

headers = {}
if api_key:
    headers["X-API-KEY"] = api_key

url = f"https://api.financialdatasets.ai/prices/snapshot/?ticker={ticker}"
print(f"URL: {url}")
response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Body: {response.text}")
