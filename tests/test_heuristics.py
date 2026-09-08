from classification.heuristics.clause_rules import (
    rule_auto_renewal,
    rule_governing_law,
    rule_limitation_of_liability,
    rule_termination_for_convenience,
)


def test_governing_law_trigger():
    text = "This Agreement shall be governed by the laws of Delaware."
    assert rule_governing_law(text, 0.45) == 0.85


def test_governing_law_non_trigger():
    text = "The parties agree to cooperate in good faith."
    assert rule_governing_law(text, 0.45) == 0.45


def test_governing_law_low_confidence_not_boosted():
    # Below the 0.40 gate — regex match alone shouldn't force a boost.
    text = "This Agreement shall be governed by the laws of Delaware."
    assert rule_governing_law(text, 0.10) == 0.10


def test_auto_renewal_trigger():
    text = "This Agreement shall automatically renew for successive one-year terms."
    assert rule_auto_renewal(text, 0.40) == 0.80


def test_auto_renewal_non_trigger():
    text = "This Agreement ends on December 31."
    assert rule_auto_renewal(text, 0.40) == 0.40


def test_termination_for_convenience_trigger():
    text = "Either party may terminate this Agreement without cause upon 30 days notice."
    assert rule_termination_for_convenience(text, 0.20) == 0.90


def test_termination_for_convenience_non_trigger():
    text = "This Agreement may be terminated for material breach."
    assert rule_termination_for_convenience(text, 0.20) == 0.20


def test_limitation_of_liability_trigger():
    text = "In no event shall liability exceed... total liability shall not exceed $1 million."
    assert rule_limitation_of_liability(text, 0.30) == 0.88


def test_limitation_of_liability_non_trigger():
    text = "Each party shall maintain insurance coverage."
    assert rule_limitation_of_liability(text, 0.30) == 0.30
