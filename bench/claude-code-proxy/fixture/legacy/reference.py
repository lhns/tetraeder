"""Reference data: country names, postal code shapes and address formatting."""

import re

COUNTRY_NAMES = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CY": "Cyprus",
    "CZ": "Czechia",
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "GR": "Greece",
    "HU": "Hungary",
    "IE": "Ireland",
    "IT": "Italy",
    "LV": "Latvia",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MT": "Malta",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
    "GB": "United Kingdom",
    "CH": "Switzerland",
    "NO": "Norway",
    "IS": "Iceland",
    "US": "United States",
    "CA": "Canada",
    "MX": "Mexico",
    "BR": "Brazil",
    "AR": "Argentina",
    "CL": "Chile",
    "CO": "Colombia",
    "PE": "Peru",
    "AU": "Australia",
    "NZ": "New Zealand",
    "JP": "Japan",
    "KR": "South Korea",
    "CN": "China",
    "IN": "India",
    "SG": "Singapore",
    "HK": "Hong Kong",
    "TW": "Taiwan",
    "TH": "Thailand",
    "VN": "Vietnam",
    "ID": "Indonesia",
    "MY": "Malaysia",
    "PH": "Philippines",
    "AE": "United Arab Emirates",
    "SA": "Saudi Arabia",
    "IL": "Israel",
    "TR": "Turkey",
    "ZA": "South Africa",
    "EG": "Egypt",
    "NG": "Nigeria",
    "KE": "Kenya",
    "MA": "Morocco",
}

# regex for the postal code of countries that have a fixed shape
POSTAL_PATTERNS = {
    "US": r"\d{5}(-\d{4})?",
    "CA": r"[A-Z]\d[A-Z] ?\d[A-Z]\d",
    "GB": r"[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}",
    "DE": r"\d{5}",
    "FR": r"\d{5}",
    "NL": r"\d{4} ?[A-Z]{2}",
    "ES": r"\d{5}",
    "IT": r"\d{5}",
    "CH": r"\d{4}",
    "AT": r"\d{4}",
    "BE": r"\d{4}",
    "SE": r"\d{3} ?\d{2}",
    "DK": r"\d{4}",
    "JP": r"\d{3}-\d{4}",
    "AU": r"\d{4}",
}


def country_name(code: str) -> str:
    return COUNTRY_NAMES.get(code.upper(), code.upper())


def is_known_country(code: str) -> bool:
    return code.upper() in COUNTRY_NAMES


def valid_postal_code(country: str, postal_code: str) -> bool:
    """True when the code fits the country's pattern; countries without a pattern accept anything non-empty."""
    pattern = POSTAL_PATTERNS.get(country.upper())
    value = postal_code.strip().upper()
    if pattern is None:
        return bool(value)
    return re.fullmatch(pattern, value) is not None


def format_postal_line(country: str, city: str, region: str, postal_code: str) -> str:
    """The line with city and postal code in the order the country uses."""
    country = country.upper()
    if country in ("US", "CA", "AU"):
        return f"{city}, {region} {postal_code}".strip(", ")
    if country == "GB":
        return f"{city}
{postal_code}"
    if country == "JP":
        return f"{postal_code} {region} {city}".strip()
    return f"{postal_code} {city}".strip()


def eu_members() -> list[str]:
    """Codes of the EU member states in the table (alphabetical)."""
    non_eu = {"GB", "CH", "NO", "IS"}
    european = [c for c in COUNTRY_NAMES if c in _EUROPE]
    return sorted(c for c in european if c not in non_eu)


_EUROPE = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT",
    "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "GB", "CH", "NO", "IS",
}
