from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ml.model_registry import download_models, upload_models


def test_upload_models_missing_token() -> None:
    """토큰이 없는 경우 upload_models가 False를 반증하는지 테스트."""
    result = upload_models(token="", repo_id="user/repo")
    assert result is False


def test_upload_models_missing_repo_id() -> None:
    """repo_id가 없는 경우 upload_models가 False를 반환하는지 테스트."""
    result = upload_models(token="hf_dummy_token", repo_id="")
    assert result is False


def test_upload_models_empty_directory(tmp_path: Path) -> None:
    """비어 있는 디렉터리 업로드 시도 시 False를 반환하는지 테스트."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()

    result = upload_models(
        model_dir=empty_dir,
        repo_id="user/repo",
        token="hf_dummy_token",
    )
    assert result is False


@patch("src.ml.model_registry.HfApi")
def test_upload_models_success(mock_hf_api_cls: MagicMock, tmp_path: Path) -> None:
    """정상적인 업로드 흐름 테스트 (Hugging Face API Mocking)."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dummy_file = model_dir / "model.bin"
    dummy_file.write_bytes(b"dummy model data")

    mock_api = MagicMock()
    mock_hf_api_cls.return_value = mock_api

    result = upload_models(
        model_dir=model_dir,
        repo_id="user/test-repo",
        token="hf_valid_token",
    )

    assert result is True
    mock_api.create_repo.assert_called_once_with(
        repo_id="user/test-repo",
        repo_type="model",
        private=True,
        token="hf_valid_token",
        exist_ok=True,
    )
    mock_api.upload_folder.assert_called_once_with(
        folder_path=str(model_dir),
        repo_id="user/test-repo",
        repo_type="model",
        token="hf_valid_token",
    )


def test_download_models_missing_token() -> None:
    """토큰이 없는 경우 download_models가 False를 반환하는지 테스트."""
    result = download_models(token="", repo_id="user/repo")
    assert result is False


@patch("src.ml.model_registry.snapshot_download")
def test_download_models_success(mock_snapshot: MagicMock, tmp_path: Path) -> None:
    """정상적인 다운로드 흐름 테스트 (Mocking)."""
    target_dir = tmp_path / "downloaded_models"

    result = download_models(
        model_dir=target_dir,
        repo_id="user/test-repo",
        token="hf_valid_token",
    )

    assert result is True
    mock_snapshot.assert_called_once_with(
        repo_id="user/test-repo",
        repo_type="model",
        local_dir=str(target_dir),
        token="hf_valid_token",
    )
