
import requests

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


def get_rxcui(drug_name):
    try:
        url = f"{BASE_URL}/rxcui.json?name={drug_name}"
        res = requests.get(url).json()

        ids = res.get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None

    except Exception as e:
        print(f"Error fetching RXCUI for {drug_name}: {e}")
        return None


def get_drug_info(rxcui):
    try:
        url = f"{BASE_URL}/rxcui/{rxcui}/properties.json"
        return requests.get(url).json()

    except Exception as e:
        print(f"Error fetching info: {e}")
        return None


def get_related(rxcui):
    try:
        url = f"{BASE_URL}/rxcui/{rxcui}/allrelated.json"
        return requests.get(url).json()

    except Exception as e:
        print(f"Error fetching related data: {e}")
        return None

