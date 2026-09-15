"""Name-matched co-mod test for src/api/kis/indicators.py (lean_check gate).

Mirrors the contract scenario test_indicators_kis_clients_use_data_key_kwargs
verbatim; the canonical scenario lives in tests/unit/api/kis/test_kis_client_helpers.py.
"""

from __future__ import annotations


def test_indicators_kis_clients_use_data_key_kwargs() -> None:
    import ast
    import inspect

    from src.api.kis import indicators

    tree = ast.parse(inspect.getsource(indicators))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "KisApiClient"
    ]

    assert len(calls) == 5
    for call in calls:
        assert call.args == []
        assert len(call.keywords) == 1
        assert call.keywords[0].arg is None
        inner = call.keywords[0].value
        assert isinstance(inner, ast.Call)
        assert isinstance(inner.func, ast.Name)
        assert inner.func.id == "kis_data_client_kwargs"
        assert inner.args == [] and inner.keywords == []
