def test_kis_sync_client_resolves_renamed_lazy_import(monkeypatch) -> None:
    import src.api.kis.client as kis_client_module
    from src.backfill.price import sources

    calls = {"ensure_token": 0, "session_closed": 0}

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            calls["session_closed"] += 1
            return False

    class _FakeKisApiClient:
        def __init__(self) -> None:
            self.token = None

        def create_session(self):
            return _FakeSession()

        async def ensure_token(self, session) -> None:
            calls["ensure_token"] += 1
            self.token = "fake-token"

    # Given: sources.py's global client cache is reset, and the monkeypatch
    # targets the DIRECT module -- matching what _kis_sync_client's lazy
    # import now resolves to after the rename.
    monkeypatch.setattr(sources, "_KIS_CLIENT", None)
    monkeypatch.setattr(kis_client_module, "KisApiClient", _FakeKisApiClient)

    # When
    client = sources._kis_sync_client()

    # Then: the fake, not the real client, was resolved, constructed, had its
    # token ensured, and its session was properly closed -- proving the renamed
    # lazy import works end-to-end.
    assert isinstance(client, _FakeKisApiClient)
    assert client.token == "fake-token"
    assert calls["ensure_token"] == 1
    assert calls["session_closed"] == 1
