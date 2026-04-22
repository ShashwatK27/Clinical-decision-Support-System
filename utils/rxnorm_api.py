"""
rxnorm_api.py — NLM RxNav API client for drug validation and enrichment.

Provides optional online validation of drug names against the NIH NLM
RxNav REST API (https://rxnav.nlm.nih.gov/). This is non-blocking —
if the API is unavailable or times out, the system continues with local
fuzzy matching only.

Key functions:
    validate_drug_online(name) → ValidationResult
    get_rxcui(name)           → str | None
    get_drug_info(rxcui)      → dict | None
    get_related(rxcui)        → dict | None

Note: All HTTP calls use a 5-second timeout to prevent process hangs.
"""

from __future__ import annotations
from dataclasses import dataclass

import requests
from utils.logger_config import get_logger

logger = get_logger("rxnorm_api")

BASE_URL = "https://rxnav.nlm.nih.gov/REST"
_TIMEOUT = 5  # seconds


@dataclass
class ValidationResult:
    """Result of validating a single drug name against RxNorm."""
    input_name: str           # name as given by the fuzzy matcher
    is_valid: bool            # True if RxNorm recognised it
    rxcui: str | None         # RxNorm Concept Unique Identifier (e.g. "5640")
    canonical_name: str | None  # preferred RxNorm display name
    tty: str | None           # term type: "IN" (ingredient), "BN" (brand), etc.
    error: str | None         # set if the API call itself failed


def validate_drug_online(drug_name: str) -> ValidationResult:
    """
    Check whether drug_name is a valid drug in the RxNorm database.

    Returns a ValidationResult with is_valid=False and error set if the
    API call fails or times out.
    """
    base = ValidationResult(
        input_name=drug_name,
        is_valid=False,
        rxcui=None,
        canonical_name=None,
        tty=None,
        error=None,
    )

    rxcui = get_rxcui(drug_name)
    if rxcui is None:
        base.error = "Not found in RxNorm"
        return base

    base.rxcui = rxcui
    base.is_valid = True

    # Try to get the canonical name and term type
    try:
        url = f"{BASE_URL}/rxcui/{rxcui}/properties.json"
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        base.canonical_name = props.get("name")
        base.tty = props.get("tty")
    except requests.Timeout:
        logger.warning(f"Timeout fetching properties for RXCUI '{rxcui}'")
    except Exception as e:
        logger.warning(f"Could not fetch properties for RXCUI '{rxcui}': {e}")

    return base


def validate_drug_list(drugs: list[str]) -> list[ValidationResult]:
    """
    Validate a list of drug names against RxNorm.

    Returns one ValidationResult per drug, in the same order.
    Never raises — individual failures are captured in result.error.
    """
    results = []
    for drug in drugs:
        try:
            results.append(validate_drug_online(drug))
        except Exception as e:
            results.append(ValidationResult(
                input_name=drug,
                is_valid=False,
                rxcui=None,
                canonical_name=None,
                tty=None,
                error=str(e),
            ))
    return results


# ── Lower-level helpers ────────────────────────────────────────────────────────

def get_rxcui(drug_name: str) -> str | None:
    """Return the RxCUI identifier for a drug name, or None if not found."""
    try:
        url = f"{BASE_URL}/rxcui.json?name={requests.utils.quote(drug_name)}"
        res = requests.get(url, timeout=_TIMEOUT)
        res.raise_for_status()
        ids = res.json().get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None
    except requests.Timeout:
        logger.warning(f"Timeout fetching RXCUI for '{drug_name}'")
        return None
    except Exception as e:
        logger.error(f"Error fetching RXCUI for '{drug_name}': {e}")
        return None


def get_drug_info(rxcui: str) -> dict | None:
    """Return properties dict for a given RxCUI, or None on error."""
    try:
        url = f"{BASE_URL}/rxcui/{rxcui}/properties.json"
        return requests.get(url, timeout=_TIMEOUT).json()
    except requests.Timeout:
        logger.warning(f"Timeout fetching drug info for RXCUI '{rxcui}'")
        return None
    except Exception as e:
        logger.error(f"Error fetching info for RXCUI '{rxcui}': {e}")
        return None


def get_related(rxcui: str) -> dict | None:
    """Return all related concepts for a given RxCUI, or None on error."""
    try:
        url = f"{BASE_URL}/rxcui/{rxcui}/allrelated.json"
        return requests.get(url, timeout=_TIMEOUT).json()
    except requests.Timeout:
        logger.warning(f"Timeout fetching related data for RXCUI '{rxcui}'")
        return None
    except Exception as e:
        logger.error(f"Error fetching related data for RXCUI '{rxcui}': {e}")
        return None
