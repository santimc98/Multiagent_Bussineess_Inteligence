import pandas as pd

from src.utils.cleaning_guards import detect_identifier_scientific_notation


def test_detect_identifier_scientific_notation():
    series = pd.Series(["1.23E+05", "4.56e+07", "7.89E-03"])
    detection = detect_identifier_scientific_notation(series)
    assert detection["flag"] is True
