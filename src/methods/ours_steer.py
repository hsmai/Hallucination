"""Ours(Omni-Steering) + Language Dom 개선 분기 — feat/ours-steer.

현행 Ours(frozen, 선배 avh_contam/cmm_contam) 재현 + 제안 분기:

  분기 규칙 (lang_mode로 제어):
  - 지각 모달 2개 존재 (AVH 전부, CMM Visual/Audio Dom):
      C = 비질의 지각 모달(keyword 판정) → h ← h − α·(h_full − h_C-masked)   [현행 그대로]
  - 지각 모달 1개뿐 (CMM Language Dom 등 — cross-modal 오염원이 존재하지 않음):
      * lang_mode="amplify"(제안): h ← h + β·(h_full − h_V-masked)
        — 증거 성분에 (1+β) 가중 = VCD 대비 원리를 L22 hidden에서 (docs 참조: 근거는
          선배 표에서 Language 1등이 VCD라는 것 + 현행 텍스트 차감이 질문까지 깎는 진단)
      * lang_mode="text_subtract"(현행 재현·대조용): C = 텍스트 토큰 차감
  2-pass(+주입 1-pass) 구조·L22·keyword 판정은 선배 방법 그대로 유지 (원본 무수정, 이식).
"""

from __future__ import annotations

import time

import torch

from . import DecodingMethod

# 선배 코드의 keyword 목록 그대로 (cmm_contam.py / avh_contam.py)
KW_AUDIO = ("sound", "hear", "audio", "noise", "sounding", "heard", "hearing")
KW_VIDEO = ("see", "look", "visible", "shown", "appear", "seen", "visual", "visibly",
            "video", "watch")

AVH_TASK_VDAH = "Video-driven Audio Hallucination"


def detect_queried(question: str, benchmark: str, category: str):
    """(queried_modality, kw_fail) — 선배 _detect_queried 이식.
    AVHBench는 task 방향으로 fallback, CMM은 video로 fallback."""
    q = (question or "").lower()
    a = any(k in q for k in KW_AUDIO)
    v = any(k in q for k in KW_VIDEO)
    if a and not v:
        return "audio", False
    if v and not a:
        return "video", False
    if benchmark == "avhbench":
        return ("audio" if category == AVH_TASK_VDAH else "video"), True
    return "video", True


class OursSteer(DecodingMethod):
    name = "ours_steer"

    def setup(self, adapter, cfg, benchmark):
        super().setup(adapter, cfg, benchmark)
        self.layer = int(cfg.get("methods.ours_steer.layer"))
        self.alpha = float(cfg.get("methods.ours_steer.alpha"))
        self.beta = float(cfg.get("methods.ours_steer.beta"))
        self.lang_mode = cfg.get("methods.ours_steer.lang_mode")
        if self.lang_mode not in ("amplify", "text_subtract"):
            raise ValueError(f"lang_mode 미지원: {self.lang_mode!r}")

    def generate(self, sample) -> dict:
        t0 = time.time()
        ctx = self.adapter.prepare(sample, self.question_with_suffix(sample))
        sp = self.adapter.steer_prepare(ctx, self.layer)

        # 지각 모달 존재 판정은 "실제 콘텐츠" 기준 (시퀀스 기준 아님 — VL2 Language Dom은
        # 정렬 사양상 무음 오디오 토큰이 시퀀스에 존재하므로 인덱스로 판정하면 오판)
        has_video = sample.video_path is not None and len(sp["video_idx"]) > 0
        has_audio = bool(sample.audio_path or sample.extra.get("audio_in_video"))
        queried, kw_fail = detect_queried(sample.question, self.benchmark, sample.category)

        # pass 1: 원본 forward (h 캡처 + baseline logits)
        logits0, h0 = self.adapter.steer_forward(ctx, sp, mask_keys=None)

        if has_video and has_audio:                        # 지각 2모달 — 현행 규칙
            branch = "contam_subtract"
            c_idx = sp["video_idx"] if queried == "audio" else sp["audio_idx"]
            _, h_m = self.adapter.steer_forward(ctx, sp, mask_keys=c_idx)
            vec = -self.alpha * (h0 - h_m)
        elif self.lang_mode == "amplify":                  # 지각 1모달 — 제안: 증거 증폭
            branch = "evidence_amplify"
            e_idx = sp["video_idx"] if has_video else sp["audio_idx"]
            _, h_m = self.adapter.steer_forward(ctx, sp, mask_keys=e_idx)
            vec = +self.beta * (h0 - h_m)
        else:                                              # 지각 1모달 — 현행 재현: 텍스트 차감
            branch = "text_subtract"
            _, h_m = self.adapter.steer_forward(ctx, sp, mask_keys=sp["text_idx"])
            vec = -self.alpha * (h0 - h_m)

        # pass 2: 주입 forward
        logits = self.adapter.steer_predict(ctx, sp, vec)
        nxt = int(torch.argmax(logits))
        prediction = self.adapter.decode_tokens([nxt])

        return {
            "prediction": prediction,
            "internals": {"ours_steer": {
                "branch": branch, "queried": queried, "kw_fail": kw_fail,
                "layer": self.layer,
                "alpha": self.alpha if branch != "evidence_amplify" else None,
                "beta": self.beta if branch == "evidence_amplify" else None,
                "vec_norm": float(vec.norm()),
                "base_pred": self.adapter.decode_tokens([int(torch.argmax(logits0))]),
            }},
            "inference_time_s": time.time() - t0,
        }
