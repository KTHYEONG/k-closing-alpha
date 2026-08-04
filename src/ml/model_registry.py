import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL_DIR = Path("artifacts/models")


def upload_models(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    repo_id: str | None = None,
    token: str | None = None,
) -> bool:
    """artifacts/models 디렉터리의 모델 파일들을 Hugging Face Private Repository로 업로드합니다.

    Args:
        model_dir: 업로드할 모델들이 위치한 디렉터리 경로.
        repo_id: Hugging Face 레포지토리 ID (미지정 시 환경변수 HF_REPO_ID 사용).
        token: Hugging Face 액세스 토큰 (미지정 시 환경변수 HF_TOKEN 사용).

    Returns:
        bool: 업로드 성공 여부.
    """
    target_dir = Path(model_dir)
    resolved_token = token if token is not None else os.getenv("HF_TOKEN")
    resolved_repo_id = repo_id if repo_id is not None else os.getenv("HF_REPO_ID")

    if not resolved_token or resolved_token == "hf_your_access_token_here":  # noqa: S105
        logger.error("HF_TOKEN이 설정되지 않았거나 기본값입니다.")
        return False

    if not resolved_repo_id or resolved_repo_id == "your_username/your_model_repo_name":
        logger.error("HF_REPO_ID가 설정되지 않았거나 기본값입니다.")
        return False

    if not target_dir.exists() or not any(target_dir.iterdir()):
        logger.warning(f"업로드할 모델 파일이 {target_dir} 에 존재하지 않습니다.")
        return False

    api = HfApi()
    logger.info(f"Hugging Face 저장소({resolved_repo_id})로 모델 업로드를 시작합니다...")
    try:
        api.create_repo(
            repo_id=resolved_repo_id,
            repo_type="model",
            private=True,
            token=resolved_token,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(target_dir),
            repo_id=resolved_repo_id,
            repo_type="model",
            token=resolved_token,
        )
        logger.info("모델 업로드가 성공적으로 완료되었습니다!")
        return True
    except Exception as e:
        logger.error(f"모델 업로드 중 오류 발생: {e}")
        return False


def download_models(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    repo_id: str | None = None,
    token: str | None = None,
) -> bool:
    """Hugging Face Private Repository에서 모델 파일들을 로컬 디렉터리로 다운로드합니다.

    Args:
        model_dir: 다운로드받을 로컬 디렉터리 경로.
        repo_id: Hugging Face 레포지토리 ID (미지정 시 환경변수 HF_REPO_ID 사용).
        token: Hugging Face 액세스 토큰 (미지정 시 환경변수 HF_TOKEN 사용).

    Returns:
        bool: 다운로드 성공 여부.
    """
    target_dir = Path(model_dir)
    resolved_token = token if token is not None else os.getenv("HF_TOKEN")
    resolved_repo_id = repo_id if repo_id is not None else os.getenv("HF_REPO_ID")

    if not resolved_token or resolved_token == "hf_your_access_token_here":  # noqa: S105
        logger.error("HF_TOKEN이 설정되지 않았거나 기본값입니다.")
        return False

    if not resolved_repo_id or resolved_repo_id == "your_username/your_model_repo_name":
        logger.error("HF_REPO_ID가 설정되지 않았거나 기본값입니다.")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Hugging Face 저장소({resolved_repo_id})에서 모델 다운로드를 시작합니다...")
    try:
        snapshot_download(
            repo_id=resolved_repo_id,
            repo_type="model",
            local_dir=str(target_dir),
            token=resolved_token,
        )
        logger.info(f"모델이 성공적으로 {target_dir} 에 다운로드되었습니다!")
        return True
    except Exception as e:
        logger.error(f"모델 다운로드 중 오류 발생: {e}")
        return False
