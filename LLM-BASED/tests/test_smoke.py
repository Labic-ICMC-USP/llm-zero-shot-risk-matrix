from risk_scorer.risk_scorer import RiskAssessment

def test_schema_bounds():
    ra = RiskAssessment(severity_level=3, occurrence_likelihood=2, justification="Looks moderate and unlikely.")
    assert 1 <= ra.severity_level <= 5
    assert 1 <= ra.occurrence_likelihood <= 5
    assert isinstance(ra.justification, str) and len(ra.justification) > 8
