"""Qwen2.5-Omni-7B 실모델 어댑터 (서버 전용 — 로컬 import 금지: 제약 1).

- base / vcd_ext / mad : MAD repo qwen-omni/utils.py 구조 이식 (thinker forward + branch KV캐시)
- avcd                 : **신규 포팅** (공식 코드 없음 — blueprint 최대 리스크 항목)
  * span: input_ids의 <|AUDIO|>/<|VIDEO|> pad 토큰 id로 동적 boolean mask (interleave 대응)
  * attention: attn_implementation="eager" + attn_patch.patch_qwen25_omni()
  * dominance/threshold/마스킹 수식: src/adapters/common.py (공식 코드 대조 검증본)
  * "마지막 layer 제외"는 thinker text layer 수 기준 일반화

⚠ SERVER-UNVERIFIED: S1 스모크 검증 필수. 실패 시 fallback = Qwen×AVCD 칸
  "공식 코드 미지원" 표기 (blueprint §9-2).
"""

from __future__ import annotations

import logging

import torch

from ..models import BRANCHES, ModelAdapter
from .common import dominance_and_threshold, qwen_spans, spec_to_mask

logger = logging.getLogger(__name__)


class QwenOmniAdapter(ModelAdapter):
    def __init__(self, cfg, method: str):
        from transformers import (Qwen2_5OmniConfig,
                                  Qwen2_5OmniForConditionalGeneration,
                                  Qwen2_5OmniProcessor)

        self.cfg = cfg
        self.method = method
        self.name = "qwen2_5_omni_7b"
        self.model_key = "qwen2_5_omni_7b"
        self.is_avcd = method == "avcd"

        model_path = cfg.get("models.qwen2_5_omni_7b.local_path")
        if not model_path or "UNKNOWN" in str(model_path):
            model_path = cfg.get("models.qwen2_5_omni_7b.hf_id")

        # ---- 로드 방식: OURS probe/model_loader.py 이식 (2026-07-16) ----
        # 1) transformers>=4.52 + torch<2.6 조합에서 from_pretrained가 talker speaker
        #    dict(torch.load)를 막음(CVE 체크) → 신뢰된 로컬 체크포인트이므로 우회 (OURS 동일)
        try:
            import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _omni_mod
            _omni_mod.check_torch_load_is_safe = lambda *a, **k: None
        except Exception as e:
            logger.warning("check_torch_load_is_safe 우회 실패: %s", e)
        # 2) talker를 아예 생성하지 않음 (~5GB 절약 — 3090 24GB 대응)
        model_cfg = Qwen2_5OmniConfig.from_pretrained(model_path)
        if hasattr(model_cfg, "enable_audio_output"):
            model_cfg.enable_audio_output = False
        # 3) "mixed": 인코더(vision/audio)는 flash로 로드(OOM 방지, OURS 분석 모드와 동일),
        #    thinker text attention만 eager로 스왑 → 전 방법 동일 numerics + AVCD 패치 가능
        dtype = torch.bfloat16 if cfg.get("experiment.dtype") == "bfloat16" else torch.float16
        encoder_impl = cfg.get("models.qwen2_5_omni_7b.encoder_attn_implementation",
                               "flash_attention_2")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, config=model_cfg, torch_dtype=dtype, device_map="cuda",
            attn_implementation=encoder_impl)
        self._make_thinker_eager()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        if cfg.get("models.qwen2_5_omni_7b.disable_talker", True) and hasattr(self.model, "disable_talker"):
            self.model.disable_talker()   # 미생성 시 no-op
        self.model.eval()

        self.tokenizer = self.processor.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.system_prompt = cfg.get("prompts.qwen_system_prompt")
        self.use_audio_in_video = cfg.get("models.qwen2_5_omni_7b.use_audio_in_video", True)

        if self.is_avcd:
            from .attn_patch import patch_qwen25_omni
            self.patch_ctx = patch_qwen25_omni(self.model)
            tc = self.model.config.thinker_config
            self.audio_token_id = getattr(tc, "audio_token_index", None)
            self.video_token_id = getattr(tc, "video_token_index", None)
            assert self.audio_token_id is not None and self.video_token_id is not None, (
                "thinker_config에서 audio/video token index를 찾지 못함 — "
                "S1에서 config 구조 확인 필요")
        self._warned_no_audio = False

    def _make_thinker_eager(self):
        """thinker text decoder attention만 eager로 스왑 (OURS _make_thinker_eager 이식).
        flash/sdpa attention 클래스는 forward만 override한 순수 서브클래스라 retag가 안전."""
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAttention
        tm = self.model.thinker.model
        for layer in tm.layers:
            layer.self_attn.__class__ = Qwen2_5OmniAttention
        tm._attn_implementation = "eager"
        tm.config._attn_implementation = "eager"
        logger.info("thinker text attention -> eager (%d layers); encoders는 %s 유지",
                    len(tm.layers),
                    self.cfg.get("models.qwen2_5_omni_7b.encoder_attn_implementation",
                                 "flash_attention_2"))

    # ------------------------------------------------------------ 입력 준비

    def _conversation(self, ctx: dict, kind: str, text_override: str | None = None):
        """kind: va|v|a|t|head — MAD qwen-omni/utils.py 구성 + OURS data_loader.to_messages 규칙.

        AV 입력 (2026-07-16 확정):
        - AVHBench(먹싱): [video, text] + use_audio_in_video=True
        - CMM AV(별도 wav): [video, audio(wav), text] + use_audio_in_video=False
        - audio-only branch: audio 소스 = 별도 wav 있으면 wav, 없으면 mp4 트랙 (OURS와 동일)
        """
        content = []
        if kind in ("va", "v", "head") and ctx["video_path"]:
            content.append({"type": "video", "video": ctx["video_path"]})
        if kind in ("va", "head") and ctx["audio_path"]:      # CMM AV: 별도 wav 블록
            content.append({"type": "audio", "audio": ctx["audio_path"]})
        if kind == "a" and ctx["audio_src"]:
            content.append({"type": "audio", "audio": ctx["audio_src"]})
        content.append({"type": "text", "text": text_override or ctx["question"]})
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]

    def _inputs(self, conversation, use_audio_in_video: bool):
        from qwen_omni_utils import process_mm_info
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True,
                                                  tokenize=False)
        audios, images, videos = process_mm_info(conversation,
                                                 use_audio_in_video=use_audio_in_video)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True,
                                use_audio_in_video=use_audio_in_video)
        return {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()}

    def prepare(self, sample, question_with_suffix: str) -> dict:
        audio_in_video = bool(sample.extra.get("audio_in_video", False))
        # uaiv: 오디오가 mp4에 먹싱된 경우에만 True (별도 wav면 separate 블록, OURS resolve와 동일)
        uaiv = bool(self.use_audio_in_video and sample.video_path
                    and audio_in_video and not sample.audio_path)
        audio_src = sample.audio_path or (sample.video_path if audio_in_video else None)
        if audio_src is None and not self._warned_no_audio:
            logger.warning("audio 없음(%s 등) → 'a' branch는 텍스트만으로 대체", sample.sample_id)
            self._warned_no_audio = True
        return {"sample_id": sample.sample_id, "question": question_with_suffix,
                "video_path": sample.video_path, "audio_path": sample.audio_path,
                "audio_src": audio_src, "uaiv": uaiv}

    def decode_tokens(self, token_ids):
        return self.tokenizer.decode(
            [t for t in token_ids if t != self.eos_token_id], skip_special_tokens=True).strip()

    # ------------------------------------------------------------ base

    def greedy_generate(self, ctx: dict, max_new_tokens: int) -> str:
        uaiv = ctx["uaiv"]
        inputs = self._inputs(self._conversation(ctx, "va" if ctx["video_path"] else "t"), uaiv)
        with torch.no_grad():
            out = self.model.generate(**inputs, use_audio_in_video=uaiv,
                                      max_new_tokens=max_new_tokens,
                                      do_sample=False, temperature=0.0, return_audio=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------ 4-branch CD

    def _branch_kinds(self, ctx):
        # audio/video 부재 시 텍스트만으로 대체 (t와 동일 입력)
        kinds = []
        for b in BRANCHES:
            if b == "va":
                kinds.append(("va", ctx["uaiv"]) if ctx["video_path"] else ("t", False))
            elif b == "v":
                kinds.append(("v", False) if ctx["video_path"] else ("t", False))
            elif b == "a":
                kinds.append(("a", False) if ctx["audio_src"] else ("t", False))
            else:
                kinds.append(("t", False))
        return kinds

    def branch_prefill(self, ctx: dict):
        logits, branches = [], []
        with torch.inference_mode():
            for kind, uaiv in self._branch_kinds(ctx):
                inputs = self._inputs(self._conversation(ctx, kind), uaiv)
                out = self.model.thinker(**inputs, use_audio_in_video=uaiv, use_cache=True)
                logits.append(out.logits[0, -1, :].float().cpu())
                branches.append({"past": out.past_key_values, "uaiv": uaiv})
        return logits, {"branches": branches}

    def branch_step(self, state, token_id: int):
        tok = torch.tensor([[token_id]], device=self.model.device, dtype=torch.long)
        logits = []
        with torch.inference_mode():
            for b in state["branches"]:
                out = self.model.thinker(input_ids=tok, use_audio_in_video=b["uaiv"],
                                         use_cache=True, past_key_values=b["past"])
                b["past"] = out.past_key_values
                logits.append(out.logits[0, -1, :].float().cpu())
        return logits

    # ------------------------------------------------------------ MAD

    def modality_query_probs(self, ctx: dict, query_prompt: str):
        uaiv = ctx["uaiv"]
        conv = self._conversation(ctx, "head", text_override=query_prompt)
        inputs = self._inputs(conv, uaiv)
        with torch.inference_mode():
            out = self.model.thinker(**inputs, use_audio_in_video=uaiv)
        z = []
        for word in ("audio", "video", "both"):
            tok = self.tokenizer.encode(word)[0]
            z.append(out.logits[0, -1, tok].float().cpu())
        p = torch.softmax(torch.stack(z), dim=0).tolist()
        return p[0], p[1], p[2]

    # ------------------------------------------------------------ AVCD (신규 포팅)

    def _avcd_inputs(self, ctx: dict, generated_ids):
        uaiv = ctx["uaiv"]
        inputs = self._inputs(self._conversation(ctx, "va" if ctx["video_path"] else "t"), uaiv)
        if generated_ids:
            gen = torch.tensor([list(generated_ids)], device=self.model.device, dtype=torch.long)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], gen], dim=1)
            if "attention_mask" in inputs:
                pad = torch.ones_like(gen)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], pad], dim=1)
        return inputs, uaiv

    def _spans_for(self, input_ids_row: torch.Tensor):
        return qwen_spans(input_ids_row.tolist(), self.audio_token_id, self.video_token_id)

    def avcd_orig_forward(self, ctx: dict, generated_ids):
        inputs, uaiv = self._avcd_inputs(ctx, generated_ids)
        self.patch_ctx.enabled = True
        self.patch_ctx.recording = True
        self.patch_ctx.span_mask = None
        self.patch_ctx.reset_records()
        try:
            with torch.inference_mode():
                out = self.model.thinker(**inputs, use_audio_in_video=uaiv, use_cache=False)
        finally:
            self.patch_ctx.recording = False
            self.patch_ctx.enabled = False

        layer_rows = self.patch_ctx.records_in_order()   # layer별 (H, S)
        if len(layer_rows) != self.patch_ctx.num_layers:
            raise RuntimeError(
                f"attention 기록 layer 수({len(layer_rows)}) != thinker layer 수"
                f"({self.patch_ctx.num_layers}) — 패치 적용 범위 점검 필요")

        # ⚠ 주의: thinker 입력 시퀀스는 미디어 placeholder가 임베딩으로 치환·확장된 뒤라
        #   input_ids 길이와 attention S가 다를 수 있음 → S1에서 검증하고, 다르면
        #   spans를 임베딩 좌표로 재매핑해야 한다 (runbook 체크 항목).
        spans = self._spans_for(inputs["input_ids"][0])
        S = layer_rows[0].shape[-1]
        if spans["video"].shape[0] != S:
            raise RuntimeError(
                f"span 길이({spans['video'].shape[0]}) != attention S({S}) — "
                f"Qwen 시퀀스 확장 재매핑 필요 (docs/server_runbook.md 참조)")
        ctx["_spans"] = spans

        dominance, threshold = dominance_and_threshold(layer_rows, spans, exclude_last_layer=True)
        return out.logits[0, -1, :].float().cpu(), dominance, threshold

    def avcd_masked_forward(self, ctx: dict, generated_ids, mask_spec: str, threshold: float):
        inputs, uaiv = self._avcd_inputs(ctx, generated_ids)
        spans = ctx.get("_spans") or self._spans_for(inputs["input_ids"][0])
        self.patch_ctx.enabled = True
        self.patch_ctx.recording = False
        self.patch_ctx.span_mask = spec_to_mask(mask_spec, spans)
        self.patch_ctx.threshold = threshold
        try:
            with torch.inference_mode():
                out = self.model.thinker(**inputs, use_audio_in_video=uaiv, use_cache=False)
        finally:
            self.patch_ctx.span_mask = None
            self.patch_ctx.enabled = False
        return out.logits[0, -1, :].float().cpu()
