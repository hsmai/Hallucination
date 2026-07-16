"""VideoLLaMA2-AV 실모델 어댑터 (서버 전용 — 로컬 import 금지: 제약 1).

백엔드 선택 (method별):
- base / vcd_ext / mad : third_party/VideoLLaMA2 (vanilla, audio_visual 브랜치)
  + attn_implementation="eager", bf16. branch/query 로직은 MAD repo에서 이식(우리 코드).
- avcd                 : third_party/AVCD (공식 fork, 무수정 사용)
  fork의 model.forward가 (outputs, avg_dominance, threshold)를 반환하고 modality/threshold
  kwarg로 마스킹을 수행 — 공식 구현 그대로가 가장 충실한 재현.
  ⚠ fork는 내부에서 fp16 캐스팅을 하드코딩 → AVCD×VideoLLaMA2 run만 fp16
    (yaml methods.avcd.videollama2_dtype 참조, 게이트에서 영향 확인)

⚠ SERVER-UNVERIFIED: 이 파일 전체는 GPU 서버에서 S1 스모크로 검증한다.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

from ..models import BRANCHES, ModelAdapter

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VANILLA_DIR = REPO_ROOT / "third_party" / "VideoLLaMA2"
AVCD_DIR = REPO_ROOT / "third_party" / "AVCD"


def _import_videollama2(backend_dir: Path):
    """fork들이 전부 top-level 'videollama2' 패키지명을 쓰므로 경로 우선순위로 선택.
    (프로세스당 한 백엔드만 — 러너가 run 단위 프로세스라서 안전)"""
    if "videollama2" in sys.modules:
        loaded_from = Path(sys.modules["videollama2"].__file__).resolve()
        if backend_dir not in loaded_from.parents:
            raise RuntimeError(
                f"videollama2 패키지가 이미 다른 백엔드에서 로드됨: {loaded_from}\n"
                f"러너를 새 프로세스로 실행하세요 (run당 1프로세스).")
    sys.path.insert(0, str(backend_dir))
    import videollama2  # noqa
    return videollama2


class VideoLLaMA2Adapter(ModelAdapter):
    def __init__(self, cfg, method: str):
        self.cfg = cfg
        self.method = method
        self.name = "videollama2_av"
        self.model_key = "videollama2_av"
        self.is_avcd = method == "avcd"

        backend = AVCD_DIR if self.is_avcd else VANILLA_DIR
        vl = _import_videollama2(backend)
        self._vl = vl

        model_path = cfg.get("models.videollama2_av.local_path")
        if not model_path or "UNKNOWN" in str(model_path):
            model_path = cfg.get("models.videollama2_av.hf_id")

        if self.is_avcd:
            # 공식 fork 재현: fp16 (fork 내부 하드코딩), fork 자체가 manual eager attention
            self.model, self.processor, self.tokenizer = vl.model_init(
                model_path, torch_dtype=torch.float16)
        else:
            dtype = torch.bfloat16 if cfg.get("experiment.dtype") == "bfloat16" else torch.float16
            self.model, self.processor, self.tokenizer = vl.model_init(
                model_path, torch_dtype=dtype,
                attn_implementation=cfg.get("experiment.attn_implementation"))
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id
        self._dtype = torch.float16 if self.is_avcd else (
            torch.bfloat16 if cfg.get("experiment.dtype") == "bfloat16" else torch.float16)
        self._warned_no_audio = False

        from videollama2.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_VIDEO_TOKEN
        self.VIDEO_TOKEN = DEFAULT_VIDEO_TOKEN
        self.AUDIO_TOKEN = DEFAULT_AUDIO_TOKEN

    # ------------------------------------------------------------ 입력 준비

    def prepare(self, sample, question_with_suffix: str) -> dict:
        """미디어 텐서 4-branch 세트 + 질문.

        오디오 규칙 (2026-07-16 OURS vl2_contam._build_inputs 대조 확정):
        - AVHBench: mp4에 오디오 먹싱 → VA = process_video(va=True),
          audio-only branch는 mp4에서 process_audio_from_video(mp4, 0)로 추출
        - CMM AV: mp4는 무음 + 별도 wav → VA = {video: va=False, audio: process_audio_from_video(wav, 0)}
          (⚠ va=True를 mp4에 쓰면 무음이 들어가 baseline 붕괴 — 선배가 실측한 함정.
           process_audio_file(연속 30s)은 오디오가 과강해져 dominance 왜곡 → 사용 금지)
        - CMM video-only(Language Dom): VA = V (오디오 자체가 없음), 'a' branch는 텍스트 대체
        """
        from videollama2.mm_utils import process_audio_from_video

        tensors = {"va": None, "v": None, "a": None, "t": None}
        audio_in_video = bool(sample.extra.get("audio_in_video", False))
        if sample.video_path:
            tensors["v"] = self.processor["video"](sample.video_path, va=False)
            if sample.audio_path:                       # CMM AV: 별도 wav
                audio_t = process_audio_from_video(sample.audio_path, 0)
                tensors["va"] = {"video": tensors["v"], "audio": audio_t}
                tensors["a"] = audio_t
            elif audio_in_video:                        # AVHBench: 먹싱
                tensors["va"] = self.processor["video"](sample.video_path, va=True)
                tensors["a"] = process_audio_from_video(sample.video_path, 0)
            else:                                       # video-only (CMM Language Dom)
                tensors["va"] = tensors["v"]
        elif sample.audio_path:                         # audio-only (본 매트릭스엔 없음)
            tensors["a"] = process_audio_from_video(sample.audio_path, 0)

        if tensors["a"] is None and not self._warned_no_audio:
            logger.warning("audio 없음(%s 등) → 'a' branch는 텍스트만으로 대체", sample.sample_id)
            self._warned_no_audio = True
        return {"sample_id": sample.sample_id, "question": question_with_suffix,
                "tensors": tensors}

    def _to_device(self, tensor, modal: str):
        if tensor is None:
            return None
        if isinstance(tensor, dict):
            t = {k: v.to(dtype=self._dtype).cuda() for k, v in tensor.items()}
        else:
            t = tensor.to(dtype=self._dtype).cuda()
        return [(t, modal)]

    def _prompt_ids(self, content: str, modal_token: str | None):
        """chat template 적용 + multimodal token 치환 (MAD mm_contrast_decode 이식)."""
        from videollama2.mm_utils import tokenizer_multimodal_token
        message = [{"role": "user", "content": content}]
        # VideoLLaMA2.1(qwen2 backbone)은 별도 system 블록 없음 (docs/code_analysis.md §1.5)
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False,
                                                    add_generation_prompt=True)
        ids = tokenizer_multimodal_token(prompt, self.tokenizer, modal_token,
                                         return_tensors="pt")
        return ids.unsqueeze(0).long().cuda()

    def _branch_inputs(self, ctx: dict):
        """branch별 (input_ids, images). audio 부재 시 'a'는 텍스트만."""
        q = ctx["question"]
        t = ctx["tensors"]
        out = []
        for b in BRANCHES:
            if b == "va" and t["va"] is not None:
                out.append((self._prompt_ids(f"{self.VIDEO_TOKEN}\n{q}", self.VIDEO_TOKEN),
                            self._to_device(t["va"], "video")))
            elif b == "v" and t["v"] is not None:
                out.append((self._prompt_ids(f"{self.VIDEO_TOKEN}\n{q}", self.VIDEO_TOKEN),
                            self._to_device(t["v"], "video")))
            elif b == "a" and t["a"] is not None:
                out.append((self._prompt_ids(f"{self.AUDIO_TOKEN}\n{q}", self.AUDIO_TOKEN),
                            self._to_device(t["a"], "audio")))
            else:  # 't' 또는 미디어 부재 branch → 텍스트만
                out.append((self._prompt_ids(q, None), None))
        return out

    def decode_tokens(self, token_ids):
        return self.tokenizer.decode(
            [t for t in token_ids if t != self.eos_token_id], skip_special_tokens=True).strip()

    # ------------------------------------------------------------ base

    def greedy_generate(self, ctx: dict, max_new_tokens: int) -> str:
        t = ctx["tensors"]
        if t["va"] is not None:
            ids = self._prompt_ids(f"{self.VIDEO_TOKEN}\n{ctx['question']}", self.VIDEO_TOKEN)
            images = self._to_device(t["va"], "video")
        else:
            ids = self._prompt_ids(ctx["question"], None)
            images = None
        attn = ids.ne(self.tokenizer.pad_token_id).long().cuda()
        with torch.inference_mode():
            out = self.model.generate(
                ids, attention_mask=attn, images=images,
                do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens,
                use_cache=True, pad_token_id=self.eos_token_id)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------ 4-branch CD

    def branch_prefill(self, ctx: dict):
        states, logits = [], []
        with torch.inference_mode():
            for ids, images in self._branch_inputs(ctx):
                attn = ids.ne(self.tokenizer.pad_token_id).long().cuda()
                out = self.model(ids, attention_mask=attn, images=images,
                                 use_cache=True, pad_token_id=self.eos_token_id)
                out = out[0] if isinstance(out, tuple) else out  # AVCD fork 호환
                states.append({"past": out.past_key_values})
                logits.append(out.logits[0, -1, :].float().cpu())
        return logits, {"branches": states, "device": "cuda"}

    def branch_step(self, state, token_id: int):
        tok = torch.tensor([[token_id]], device="cuda", dtype=torch.long)
        logits = []
        with torch.inference_mode():
            for b in state["branches"]:
                out = self.model(tok, images=None, use_cache=True,
                                 past_key_values=b["past"], pad_token_id=self.eos_token_id)
                out = out[0] if isinstance(out, tuple) else out
                b["past"] = out.past_key_values
                logits.append(out.logits[0, -1, :].float().cpu())
        return logits

    # ------------------------------------------------------------ MAD

    def modality_query_probs(self, ctx: dict, query_prompt: str):
        """head 프롬프트 1회 forward → (p_audio, p_video, p_both).
        (MAD mm_contrast_decode L365-388 이식 — head는 VA 입력)"""
        ids = self._prompt_ids(f"{self.VIDEO_TOKEN}\n{query_prompt}", self.VIDEO_TOKEN)
        attn = ids.ne(self.tokenizer.pad_token_id).long().cuda()
        images = self._to_device(ctx["tensors"]["va"], "video") \
            if ctx["tensors"]["va"] is not None else None
        with torch.inference_mode():
            out = self.model(ids, attention_mask=attn, images=images,
                             use_cache=False, pad_token_id=self.eos_token_id)
            out = out[0] if isinstance(out, tuple) else out
        z = []
        for word in ("audio", "video", "both"):
            tok = self.tokenizer.encode(word)[0]
            z.append(out.logits[0, -1, tok].float().cpu())
        p = torch.softmax(torch.stack(z), dim=0).tolist()
        return p[0], p[1], p[2]

    # ------------------------------------------------------------ AVCD (공식 fork 위임)

    def _avcd_ids(self, ctx: dict, generated_ids):
        ids = self._prompt_ids(f"{self.VIDEO_TOKEN}\n{ctx['question']}", self.VIDEO_TOKEN)
        if generated_ids:
            gen = torch.tensor([list(generated_ids)], device=ids.device, dtype=torch.long)
            ids = torch.cat([ids, gen], dim=1)
        return ids

    def avcd_orig_forward(self, ctx: dict, generated_ids):
        assert self.is_avcd, "avcd 메서드 전용 백엔드가 아닙니다"
        ids = self._avcd_ids(ctx, generated_ids)
        attn = ids.ne(self.tokenizer.pad_token_id).long().cuda()
        images = self._to_device(ctx["tensors"]["va"], "video")
        with torch.no_grad():
            out, dominance, threshold = self.model(
                input_ids=ids, attention_mask=attn, images=images, return_dict=True)
        return out.logits[0, -1, :].float().cpu(), dominance, threshold

    def avcd_masked_forward(self, ctx: dict, generated_ids, mask_spec: str, threshold: float):
        ids = self._avcd_ids(ctx, generated_ids)
        attn = ids.ne(self.tokenizer.pad_token_id).long().cuda()
        images = self._to_device(ctx["tensors"]["va"], "video")
        with torch.no_grad():
            out, _, _ = self.model(input_ids=ids, attention_mask=attn, images=images,
                                   return_dict=True, modality=mask_spec, threshold=threshold)
        return out.logits[0, -1, :].float().cpu()
