from rxreadable.parser import parse_prescription, parse_line


def test_abbreviated_line():
    m = parse_line("1) Tab Pantoprazole 40mg 1 OD AC x 14d")
    assert (m.name, m.form, m.strength, m.frequency, m.timing, m.duration) == \
           ("Pantoprazole", "Tab", "40mg", "OD", "before food", "14 days")


def test_ocr_typo_and_lookalike():
    m = parse_line("2. Tab Losarten 50mg 1-0-1 x 5/7")
    assert m.name == "Losartan" and m.frequency == "1-0-1" and m.duration == "5 days"
    assert any("Look-alike" in w for w in m.warnings)


def test_correction_and_wrap():
    p = parse_prescription("Rx\n1. Tab ~~Valsartan~~ Losartan 50mg BD\nx 7d\nAdv: rest")
    assert p["medicines"][0]["name"] == "Losartan" and p["medicines"][0]["duration"] == "7 days"
    assert p["advice"] == ["Adv: rest"]


def test_easy_full_words():
    m = parse_line("1. Paracetamol 500 mg - 1 tablet three times a day for 5 days")
    assert (m.strength, m.dose, m.frequency, m.duration) == ("500mg", "1 tablet", "three times a day", "5 days")
