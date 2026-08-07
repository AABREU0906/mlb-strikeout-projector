from app.services.pitcher_prop_matcher import match_pitcher_name, normalize_pitcher_name, pair_over_under


def test_accents_stripped():
    assert normalize_pitcher_name("Jose Ramirez") == normalize_pitcher_name("José Ramírez")


def test_jr_suffix_stripped():
    assert normalize_pitcher_name("Ronald Acuna Jr.") == normalize_pitcher_name("Ronald Acuna")


def test_sr_suffix_stripped():
    assert normalize_pitcher_name("Fernando Tatis Sr") == normalize_pitcher_name("Fernando Tatis")


def test_ii_suffix_stripped():
    assert normalize_pitcher_name("Michael Harris II") == normalize_pitcher_name("Michael Harris")


def test_iii_suffix_stripped():
    assert normalize_pitcher_name("Some Player III") == normalize_pitcher_name("Some Player")


def test_hyphen_normalized():
    assert normalize_pitcher_name("Jean Segura-Smith") == normalize_pitcher_name("Jean Segura Smith")


def test_punctuation_normalized():
    assert normalize_pitcher_name("A.J. Puk") == normalize_pitcher_name("AJ Puk")


def test_empty_and_none_safe():
    assert normalize_pitcher_name(None) == ""
    assert normalize_pitcher_name("") == ""


def test_pitcher_matching_handles_accented_target():
    candidates = ["Cristopher Sanchez", "Zack Wheeler", "Aaron Nola"]
    matched = match_pitcher_name("Cristopher Sánchez", candidates)
    assert matched == "Cristopher Sanchez"


def test_pitcher_matching_no_match_returns_none():
    candidates = ["Cristopher Sanchez", "Zack Wheeler"]
    matched = match_pitcher_name("Nobody Here", candidates)
    assert matched is None


def test_ambiguous_pitcher_match_safely_skipped():
    candidates = ["Mike Smith", "Mike Smith"]
    matched = match_pitcher_name("Mike Smith", candidates)
    assert matched is None


def test_valid_same_line_pairing_succeeds():
    outcomes = [
        {"name": "Over", "point": 5.5, "price": -110, "last_update": "2026-07-15T18:00:00Z"},
        {"name": "Under", "point": 5.5, "price": -120, "last_update": "2026-07-15T18:00:00Z"},
    ]
    line = pair_over_under(outcomes)
    assert line is not None
    assert line.line == 5.5
    assert line.over_odds == -110
    assert line.under_odds == -120


def test_mismatched_lines_rejected():
    outcomes = [
        {"name": "Over", "point": 5.5, "price": -110},
        {"name": "Under", "point": 6.5, "price": -120},
    ]
    assert pair_over_under(outcomes) is None


def test_missing_under_side_rejected():
    outcomes = [{"name": "Over", "point": 5.5, "price": -110}]
    assert pair_over_under(outcomes) is None


def test_missing_over_side_rejected():
    outcomes = [{"name": "Under", "point": 5.5, "price": -120}]
    assert pair_over_under(outcomes) is None


def test_alternate_lines_mixed_in_causes_safe_rejection():
    outcomes = [
        {"name": "Over", "point": 5.5, "price": -110},
        {"name": "Under", "point": 5.5, "price": -120},
        {"name": "Over", "point": 6.5, "price": 130},
        {"name": "Under", "point": 4.5, "price": -150},
    ]
    assert pair_over_under(outcomes) is None


def test_empty_outcomes_returns_none():
    assert pair_over_under([]) is None
