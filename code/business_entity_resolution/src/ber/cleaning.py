"""Conservative, local text transformations; no geocoding or identity lookups."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

LEGAL = {"pvt": "private", "ltd": "limited", "corp": "corporation",
         "inc": "incorporated", "co": "company"}
LEGAL_WORDS = set(LEGAL.values()) | {"llc", "llp", "plc", "sa", "sarl", "sas"}
STREETS = {"rd": "road", "ave": "avenue", "blvd": "boulevard", "hwy": "highway"}


def normalise(text, fold_accents=False):
    text = unicodedata.normalize("NFKC", str(text)).casefold().replace("&", " and ")
    if fold_accents:
        folded, latin_base = [], False
        for c in unicodedata.normalize("NFKD", text):
            if unicodedata.category(c).startswith("M"):
                if latin_base:
                    continue
            else:
                latin_base = "LATIN" in unicodedata.name(c, "")
            folded.append(c)
        text = unicodedata.normalize("NFC", "".join(folded))
    # Python's \w omits combining marks, including meaningful Indic vowel signs.
    # Preserve letters, numbers and marks; replace punctuation with separators.
    return " ".join("".join(c if unicodedata.category(c)[0] in {"L", "N", "M"} or c.isspace()
                            else " " for c in text).split())


def extract_postal(address, country):
    # Parsing hints, not a closed country list. Unknown labels have a numeric fallback.
    if country in {"india", "in", "ind"}:
        pattern = r"\b[1-9]\d{5}\b"
    elif country in {"us", "usa", "united states", "united states of america"}:
        pattern = r"\b\d{5}(?:-\d{4})?\b"
    elif country in {"france", "fr", "fra"}:
        pattern = r"\b\d{5}\b"
    else:
        pattern = r"\b\d{4,6}\b"
    hits = re.findall(pattern, address)
    return hits[-1].split("-")[0] if hits else ""


def clean_frame(frame):
    rows = []
    for row in frame.to_dict("records"):
        country = normalise(row["country"])
        name = " ".join(LEGAL.get(t, t) for t in normalise(row["business_name"]).split())
        core_words = name.split()
        while core_words and core_words[-1] in LEGAL_WORDS:
            core_words.pop()
        core = " ".join(core_words)
        address = " ".join(STREETS.get(t, t) for t in normalise(row["business_address"]).split())
        postal = extract_postal(row["business_address"], country)
        # Only infer the leading number. Landmarks and unit-first formats stay uncertain.
        house = re.match(r"^\s*(\d+[a-zA-Z]?(?:[-/]\d+[a-zA-Z]?)?)\b", row["business_address"])
        house = house.group(1).casefold() if house else ""
        if house == postal:
            house = ""
        unit = re.search(r"\b(?:unit|suite|flat|shop|apt|apartment)\s+([\w/-]+)",
                         row["business_address"], flags=re.I)
        rows.append({**row, "country_norm": country, "name_norm": name,
                     "name_core": core or name, "name_folded": normalise(core or name, True),
                     "address_norm": address, "address_folded": normalise(address, True),
                     "postal": postal, "house": house, "unit": unit.group(1).casefold() if unit else "",
                     "record_text": f"Name: {row['business_name']} | Address: {row['business_address']} | Country: {row['country']}"})
    columns = list(frame.columns) + ["country_norm", "name_norm", "name_core", "name_folded",
                                    "address_norm", "address_folded", "postal", "house", "unit", "record_text"]
    return pd.DataFrame(rows, columns=columns)


def exact_keys(row):
    country, postal, house = row["country_norm"], row["postal"], row["house"]
    keys = []
    if postal and house:
        keys.append(("postal_house", country, postal, house))
    if row["name_folded"]:
        keys.append(("name", country, row["name_folded"]))
    if house and row["address_folded"]:
        keys.append(("address", country, row["address_folded"]))
    return keys
