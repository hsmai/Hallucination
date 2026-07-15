"""러너 파이프라인 CPU 검증 — 중단-재개, D2 스키마 완전성, 채점/집계 연동 (MockModel)."""

import json
from pathlib import Path

import pytest
import yaml

from src.runner import build_parser, run
from src.score import mad_is_correct, score_jsonl

REPO = Path(__file__).resolve().parent.parent

D2_FIELDS = {"sample_id", "video_id", "benchmark", "category", "question", "ground_truth",
             "method", "model", "prediction", "correct", "internals",
             "inference_time_s", "seed", "config_hash"}


@pytest.fixture
def tmp_cfg(tmp_path):
    """results_dir만 tmp로 돌린 설정 사본."""
    raw = yaml.safe_load((REPO / "configs" / "unified_settings.yaml").read_text())
    raw["paths"]["results_dir"] = str(tmp_path / "results")
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True))
    return p


def run_cli(argv):
    return run(build_parser().parse_args(argv))


class TestRunnerPipeline:
    def test_resume_and_schema(self, tmp_cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO)  # data/qa 상대경로
        base_args = ["--model", "videollama2_av", "--method", "mad", "--benchmark", "cmm",
                     "--dry-run", "--config", str(tmp_cfg)]
        assert run_cli(base_args + ["--limit", "7"]) == 0
        out = tmp_path / "results" / "dryrun" / "cmm" / "videollama2_av__mad.jsonl"
        lines1 = out.read_text().strip().splitlines()
        assert len(lines1) == 7

        # 재개: limit 12 → 기존 7 skip, 5개만 추가. 기존 라인은 바이트 그대로 유지
        assert run_cli(base_args + ["--limit", "12"]) == 0
        lines2 = out.read_text().strip().splitlines()
        assert len(lines2) == 12
        assert lines2[:7] == lines1

        # D2 스키마 필드 완전성 + 내용
        for line in lines2:
            r = json.loads(line)
            assert set(r.keys()) == D2_FIELDS
            assert r["benchmark"] == "CMM" and r["method"] == "mad"
            assert r["seed"] == 42 and r["correct"] is None
            assert "::" in r["sample_id"]
            assert set(r["internals"]["mad"].keys()) == {"w_v", "w_a", "w_av"}
            assert abs(sum(r["internals"]["mad"].values()) - 1.0) < 1e-5  # softmax 확률

        # 결정성: MockAdapter는 재실행에도 동일 출력이어야 한다 (해시 RNG)
        # (inference_time_s는 실측 시간이라 비교에서 제외)
        def strip_time(lines):
            return [{k: v for k, v in json.loads(l).items() if k != "inference_time_s"}
                    for l in lines]
        out.unlink()
        assert run_cli(base_args + ["--limit", "7"]) == 0
        assert strip_time(out.read_text().strip().splitlines()) == strip_time(lines1)

    def test_all_methods_write_expected_internals(self, tmp_cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO)
        for method, key in [("base", None), ("vcd_ext", "vcd_ext"), ("avcd", "avcd")]:
            assert run_cli(["--model", "qwen2_5_omni_7b", "--method", method,
                            "--benchmark", "avhbench", "--dry-run", "--limit", "3",
                            "--config", str(tmp_cfg)]) == 0
            out = tmp_path / "results" / "dryrun" / "avhbench" / f"qwen2_5_omni_7b__{method}.jsonl"
            r = json.loads(out.read_text().strip().splitlines()[0])
            if key is None:
                assert r["internals"] == {}
            else:
                assert key in r["internals"]
        # AVCD internals 필수 필드 (blueprint D2)
        assert {"dominant", "ead_skipped_ratio"} <= set(r["internals"]["avcd"].keys())

    def test_score_fills_correct_and_is_idempotent(self, tmp_cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO)
        run_cli(["--model", "videollama2_av", "--method", "base", "--benchmark", "avhbench",
                 "--dry-run", "--limit", "10", "--config", str(tmp_cfg)])
        out = tmp_path / "results" / "dryrun" / "avhbench" / "videollama2_av__base.jsonl"
        s1 = score_jsonl(out, mad_is_correct)
        assert s1["overall"]["total"] == 10
        recs = [json.loads(l) for l in out.read_text().strip().splitlines()]
        assert all(isinstance(r["correct"], bool) for r in recs)
        s2 = score_jsonl(out, mad_is_correct)  # 재채점 안전
        assert s1["overall"] == s2["overall"]

    def test_meta_manifest(self, tmp_cfg, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO)
        args = ["--model", "videollama2_av", "--method", "base", "--benchmark", "cmm",
                "--dry-run", "--limit", "2", "--config", str(tmp_cfg)]
        run_cli(args)
        run_cli(args)  # 두 번째 실행도 매니페스트 append
        meta = json.loads((tmp_path / "results" / "dryrun" / "cmm"
                           / "videollama2_av__base.meta.json").read_text())
        assert len(meta) == 2
        assert meta[0]["config_hash"] == meta[1]["config_hash"]
        assert meta[1]["n_already_done"] == 2
        assert "config_pending" in meta[0]


class TestOverrides:
    """--set 오버라이드 (게이트 α그리드/β판정의 기반)."""

    def test_set_unknown_node_and_plain_node(self):
        from src.config import load_config
        from src.runner import apply_overrides
        cfg = load_config()
        apply_overrides(cfg, ["methods.avcd.alpha.cmm=1.5",        # UNKNOWN 노드
                              "methods.avcd.faithful_mode=false",  # 일반 노드 (json bool)
                              "benchmarks.avhbench.split=avcd_val"])
        assert cfg.get("methods.avcd.alpha.cmm") == 1.5
        assert cfg.get("methods.avcd.faithful_mode") is False
        assert cfg.get("benchmarks.avhbench.split") == "avcd_val"

    def test_set_bad_key_raises(self):
        from src.config import load_config
        from src.runner import apply_overrides
        cfg = load_config()
        with pytest.raises(KeyError):
            apply_overrides(cfg, ["methods.avcd.no_such_key=1"])
        with pytest.raises(ValueError):
            apply_overrides(cfg, ["missing_equals_sign"])

    def test_config_hash_changes_with_override(self):
        from src.config import load_config
        from src.runner import apply_overrides
        c1, c2 = load_config(), load_config()
        apply_overrides(c2, ["methods.avcd.alpha.cmm=3.0"])
        assert c1.config_hash() != c2.config_hash()  # 오버라이드가 manifest에 반영됨


class TestScorerRules:
    """MAD score.py 이식 정확성 — 원본 규칙의 대표 케이스."""

    @pytest.mark.parametrize("pred,gt,expect", [
        ("Yes", "Yes", True), ("no", "No", True), ("Yes.", "yes", True),
        ("No, there is no sound.", "no", True),
        ("Yes, the man is visible.", "no", False),
        ("yes or no", "yes", True),      # yes 패턴 우선 규칙 (원본 그대로)
        ("True", "yes", True), ("Incorrect", "no", True),
        ("maybe", "yes", False),
    ])
    def test_cases(self, pred, gt, expect):
        assert mad_is_correct(pred, gt) is expect
