"""프로젝트 전역 설정 (레거시 브릿지).

설정은 `src/config/` 도메인 모듈(base/kis/gsheet/trading)로 이관되었습니다.
이 모듈은 기존 `from src import settings` 후 `settings.XXX` 참조와
`from src.settings import Settings` 구문이 변경 없이 동작하도록
`src.config` 의 Singleton 및 모듈 레벨 상수를 재수출합니다.
"""

from __future__ import annotations

from src.config import *  # noqa: F401, F403
from src.config import __all__ as _config_exports

__all__ = [*_config_exports]
