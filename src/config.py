"""unified_settings.yaml 로더.

모든 세팅은 이 모듈을 통해서만 읽는다 (하드코딩 금지, CLAUDE.md 제약 3).

UNKNOWN_pending_server 규약:
  yaml에서 서버 확인 전 미확정 값은 다음 형태를 가진다.
      key:
        status: UNKNOWN_pending_server
        temp_default: <임시 기본값>
  resolve()는 temp_default를 반환하되 pending 목록에 기록한다.
  실행 시 pending 항목은 경고 로그로 출력된다 (조용히 임시값이 쓰이는 사고 방지).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

UNKNOWN = "UNKNOWN_pending_server"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "unified_settings.yaml"


class Config:
    """dict 래퍼. cfg.get('a.b.c')로 접근하고 UNKNOWN 노드는 temp_default로 해석한다."""

    def __init__(self, raw: dict, source_path: Path):
        self._raw = raw
        self.source_path = source_path
        self.pending: list[str] = []  # temp_default로 대체된 키 경로들
        self._resolved = self._resolve_node(copy.deepcopy(raw), path="")

    def _resolve_node(self, node: Any, path: str) -> Any:
        if isinstance(node, dict):
            # {status: UNKNOWN_pending_server, temp_default: X, ...} 패턴
            if node.get("status") == UNKNOWN:
                if "temp_default" not in node:
                    raise ValueError(
                        f"config '{path}': status={UNKNOWN} 인데 temp_default가 없습니다. "
                        f"unified_settings.yaml을 수정하세요."
                    )
                self.pending.append(path)
                resolved = copy.deepcopy(node)
                resolved["_resolved_value"] = self._resolve_node(node["temp_default"], path)
                return resolved
            return {k: self._resolve_node(v, f"{path}.{k}" if path else k) for k, v in node.items()}
        if node == UNKNOWN:
            # 스칼라형 UNKNOWN (temp_default 없음) — 값이 필요해지는 시점에 에러를 내도록 마커 유지
            self.pending.append(path)
            return node
        return node

    def get(self, dotted_key: str, default: Any = ...) -> Any:
        """'methods.avcd.ead_tau' 형태로 조회. UNKNOWN 노드는 temp_default 값을 반환."""
        node = self._resolved
        walked = []
        for part in dotted_key.split("."):
            walked.append(part)
            if not isinstance(node, dict):
                node = None
            elif isinstance(node, dict) and node.get("status") == UNKNOWN:
                # UNKNOWN 노드 내부로 더 들어가려는 경우 temp_default 쪽으로
                node = node["_resolved_value"]
                if isinstance(node, dict):
                    node = node.get(part)
                else:
                    node = None
            else:
                node = node.get(part)
            if node is None:
                if default is ...:
                    raise KeyError(f"config key not found: {dotted_key} (stopped at {'.'.join(walked)})")
                return default
        if isinstance(node, dict) and node.get("status") == UNKNOWN:
            return node["_resolved_value"]
        if node == UNKNOWN:
            # temp_default 없는 스칼라 UNKNOWN: caller가 default로 대체 동작을 선언했으면 그것을 사용
            if default is not ...:
                return default
            raise ValueError(
                f"config '{dotted_key}' = {UNKNOWN} 이며 temp_default가 없습니다. "
                f"서버에서 값을 확정하거나 임시 기본값을 yaml에 추가하세요."
            )
        return node

    def is_pending(self, dotted_key: str) -> bool:
        return dotted_key in self.pending

    def config_hash(self) -> str:
        """D2 스키마의 config_hash — 해석된(resolved) 설정 전체의 sha256 앞 12자리."""

        def strip(node: Any) -> Any:
            if isinstance(node, dict):
                if node.get("status") == UNKNOWN:
                    return {"status": UNKNOWN, "value": strip(node["_resolved_value"])}
                return {k: strip(v) for k, v in node.items()}
            return node

        blob = json.dumps(strip(self._resolved), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def warn_pending(self) -> None:
        if self.pending:
            logger.warning(
                "서버 확정 전 임시 기본값(temp_default)으로 동작 중인 항목 %d개: %s",
                len(self.pending), ", ".join(self.pending),
            )


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw, path)
    cfg.warn_pending()
    return cfg
