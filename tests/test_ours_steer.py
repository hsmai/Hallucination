"""ours_steer (feat/ours-steer) — 분기 규칙·부호·배관 검증 (Mock, CPU)."""

import json
from pathlib import Path

import pytest
import yaml

from src.methods.ours_steer import detect_queried
from src.runner import build_parser, run

REPO = Path(__file__).resolve().parent.parent


class TestDetectQueried:
    """선배 keyword 판정 이식 정확성."""

    @pytest.mark.parametrize("q,bench,cat,expect,fail", [
        ("Is the dog making sound in the audio?", "avhbench", "Video-driven Audio Hallucination", "audio", False),
        ("Is the man visible in the video?", "avhbench", "Audio-driven Video Hallucination", "video", False),
        # 키워드 양쪽 다 없음 → AVH는 task 방향 fallback (kw_fail=True)
        ("Is it raining?", "avhbench", "Video-driven Audio Hallucination", "audio", True),
        ("Is it raining?", "avhbench", "Audio-driven Video Hallucination", "video", True),
        # CMM fallback = video
        ("Is it raining?", "cmm", "overrely_language_ignore_visual", "video", True),
        # 양쪽 키워드 공존 → fallback
        ("Did you see what makes the sound?", "cmm", "overrely_visual_ignore_audio", "video", True),
    ])
    def test_cases(self, q, bench, cat, expect, fail):
        got, kw_fail = detect_queried(q, bench, cat)
        assert got == expect and kw_fail is fail


class TestBranchAndSign:
    """분기 선택과 스티어링 부호 — mock adapter로 method.generate 경로 검증."""

    def _mk(self, lang_mode="amplify"):
        from src.config import load_config
        from src.methods import get_method
        from src.models import MockAdapter
        cfg = load_config()
        cfg._resolved["methods"]["ours_steer"]["lang_mode"] = lang_mode
        m = get_method("ours_steer")
        m.setup(MockAdapter("qwen2_5_omni_7b"), cfg, "cmm")
        return m

    def _sample(self, audio=True):
        from src.data import Sample
        return Sample(
            sample_id="x::q", video_id="x", benchmark="CMM",
            category="overrely_language_ignore_visual" if not audio else "overrely_visual_ignore_audio",
            question="Did you hear bird chirping in the audio?" if audio
            else "Did you see the zebra in the video?",
            ground_truth="yes", video_path="v.mp4",
            audio_path="a.wav" if audio else None, extra={})

    def test_bimodal_uses_contam_subtract(self):
        m = self._mk()
        out = m.generate(self._sample(audio=True))
        i = out["internals"]["ours_steer"]
        assert i["branch"] == "contam_subtract"
        assert i["queried"] == "audio"
        assert i["alpha"] == 1.0 and i["beta"] is None

    def test_unimodal_amplify(self):
        m = self._mk("amplify")
        out = m.generate(self._sample(audio=False))
        i = out["internals"]["ours_steer"]
        assert i["branch"] == "evidence_amplify"
        assert i["beta"] == 1.0 and i["alpha"] is None

    def test_unimodal_text_subtract_mode(self):
        m = self._mk("text_subtract")
        out = m.generate(self._sample(audio=False))
        assert out["internals"]["ours_steer"]["branch"] == "text_subtract"

    def test_avhbench_muxed_is_bimodal(self):
        """AVH: audio_path는 None이지만 먹싱 오디오 존재 → contam_subtract 분기여야 함."""
        from src.data import Sample
        m = self._mk()
        m.benchmark = "avhbench"
        s = Sample(sample_id="y::q", video_id="y", benchmark="AVHBench",
                   category="Video-driven Audio Hallucination",
                   question="Is the car making sound in the audio?",
                   ground_truth="Yes", video_path="v.mp4", audio_path=None,
                   extra={"audio_in_video": True})
        out = m.generate(s)
        assert out["internals"]["ours_steer"]["branch"] == "contam_subtract"


class TestSteerSign:
    """부호의 수학적 검증: vec = −α·d(차감) vs +β·d(증폭)."""

    def test_vector_directions(self):
        import torch
        h0 = torch.tensor([1.0, 2.0, 3.0])
        hm = torch.tensor([0.5, 1.0, 1.5])
        d = h0 - hm
        assert torch.allclose(-1.0 * d, torch.tensor([-0.5, -1.0, -1.5]))   # 차감
        assert torch.allclose(+1.0 * d, d)                                   # 증폭
        # 증폭 주입 후 상태 = h0 + β·d = (1+β)·d + h_m → 증거 성분 (1+β) 배 (VCD 원리)
        assert torch.allclose(hm + (1 + 1.0) * d, h0 + 1.0 * d)


class TestDryRunPipeline:
    def test_ours_steer_dry_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO)
        raw = yaml.safe_load((REPO / "configs" / "unified_settings.yaml").read_text())
        raw["paths"]["results_dir"] = str(tmp_path / "results")
        cfgp = tmp_path / "cfg.yaml"
        cfgp.write_text(yaml.safe_dump(raw, allow_unicode=True))
        for bench in ("cmm", "avhbench"):
            assert run(build_parser().parse_args(
                ["--model", "qwen2_5_omni_7b", "--method", "ours_steer", "--benchmark", bench,
                 "--dry-run", "--limit", "6", "--config", str(cfgp)])) == 0
            out = tmp_path / "results" / "dryrun" / bench / "qwen2_5_omni_7b__ours_steer.jsonl"
            recs = [json.loads(l) for l in out.read_text().strip().splitlines()]
            assert len(recs) == 6
            branches = {r["internals"]["ours_steer"]["branch"] for r in recs}
            if bench == "avhbench":
                assert branches == {"contam_subtract"}          # 먹싱 오디오 → 전부 2모달
            else:
                assert "contam_subtract" in branches            # Visual/Audio Dom
        # CMM 언어돔 분기 확인 (언어돔 샘플 포함해 넉넉히)
        assert run(build_parser().parse_args(
            ["--model", "videollama2_av", "--method", "ours_steer", "--benchmark", "cmm",
             "--dry-run", "--ids-file", "data/qa/gate_cmm_ids.txt", "--config", str(cfgp)])) == 0
        out = tmp_path / "results" / "dryrun" / "cmm" / "videollama2_av__ours_steer.jsonl"
        recs = [json.loads(l) for l in out.read_text().strip().splitlines()]
        by = {}
        for r in recs:
            by.setdefault(r["category"], set()).add(r["internals"]["ours_steer"]["branch"])
        assert by["overrely_visual_ignore_audio"] == {"contam_subtract"}
        assert by["overrely_audio_ignore_visual"] == {"contam_subtract"}
        assert by["overrely_language_ignore_visual"] == {"evidence_amplify"}   # ★ 제안 분기
