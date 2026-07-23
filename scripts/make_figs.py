#!/usr/bin/env python3
"""MAD.pdf Fig.9-10 스타일 정성 비교 figure 생성 (Base/AVCD/MAD, HTML)."""
import base64, html
from pathlib import Path

FIGROOT = Path("/Users/hansangmin/Hallucination/results/runs/qualitative/figfinal")
OUT = Path("/Users/hansangmin/Hallucination/docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

def frames_b64(vid, idxs=(0, 2, 3, 5)):
    out = []
    for i in idxs:
        p = FIGROOT / vid / f"frame_{i}.jpg"
        if p.exists():
            out.append(base64.b64encode(p.read_bytes()).decode())
    return out

# hl(text, [(substr, 'red'|'blue'), ...])
def hl(text, spans=()):
    t = html.escape(text)
    for sub, cls in spans:
        t = t.replace(html.escape(sub), f'<span class="{cls}">{html.escape(sub)}</span>')
    return t

# ── Ours 슬롯 (선배님이 직접 채우는 자리) ──
OURS_SLOT = ('<span class="slot">[ Ours 출력을 여기에 붙여넣으세요 — '
             '환각 없이 올바른 서술이면 핵심 구절을 &lt;span class="blue"&gt;…&lt;/span&gt;로 감싸세요 ]</span>')
OURS_SLOT_YN = ('<span class="slot">[ Ours의 Yes/No — 정답이면 '
                '&lt;span class="blue"&gt;No&lt;/span&gt; 형태로 ]</span>')

# ── figure 정의 ──
FIGS = {
"fig9_videollama2": {
  "caption": "Figure 9. Qualitative Results in VideoLLaMA2-AV",
  "blocks": [
    {"vid": "00001",
     "video": "A man holds a microphone to his mouth at a nighttime performance.",
     "sound": "Ambient/background audio — the man is not the source of vocal sound.",
     "question": "Please describe what you can hear and see in detail.",
     "tag": "*Vision-driven audio hallucination",
     "methods": [
       ("Base", hl("A man is singing into a microphone.",
                   [("singing into a microphone", "red")])),
       ("AVCD", hl("A yellow screen with black words stating 'african symbol sound 3186 live and direct damali beach moddy jama' is shown.",
                   [("A yellow screen with black words", "blue")])),
       ("MAD", hl("A man singing into a microphone.", [("singing", "red")])),
       ("Ours", OURS_SLOT),
     ]},
    {"vid": "00103",
     "video": "A person dries a black-and-white baby goat with a white towel.",
     "sound": "The sound of a goat bleating.",
     "question": "Is the towel making sound in the audio?",
     "tag": "*Vision-driven audio hallucination",
     "yn": True,
     "methods": [
       ("Base", hl("Yes", [("Yes", "red")])),
       ("AVCD", hl("Yes", [("Yes", "red")])),
       ("MAD", hl("Yes", [("Yes", "red")])),
       ("Ours", OURS_SLOT_YN),
     ]},
  ]},
"fig10_qwen": {
  "caption": "Figure 10. Qualitative Results in Qwen2.5-Omni",
  "blocks": [
    {"vid": "00092",
     "video": "A person stitches fabric with a sewing machine.",
     "sound": "The rhythmic hum of the sewing machine's motor.",
     "question": "Please describe what you can hear and see in detail.",
     "tag": "*Vision-driven audio hallucination",
     "methods": [
       ("Base", hl("Well, I can see a sewing machine in action. There's a piece of fabric under the needle, and a person's hands are guiding it. The machine is making that rhythmic clicking sound as it sews.",
                   [("The machine is making that rhythmic clicking sound", "blue")])),
       ("AVCD", hl("Well, I see someone using a sewing machine. There's this piece of fabric under the needle, and they're adjusting some part of the machine. And then you hear the sewing machine whirring and clicking as it sews the fabric.",
                   [("you hear the sewing machine whirring and clicking", "blue")])),
       ("MAD", hl("Well, it seems like you're describing a sewing machine in action. You can hear the rhythmic sound of the needle going up and down, and the fabric being fed through. There's also the sound of the motor running.",
                  [("the fabric being fed through", "red")])),
       ("Ours", OURS_SLOT),
     ]},
    {"vid": "00127",
     "video": "A hand handles a large ball of pink cotton candy in a metal bowl.",
     "sound": "The sound of a machine clicking.",
     "question": "Is the cotton candy making sound in the audio?",
     "tag": "*Vision-driven audio hallucination",
     "yn": True,
     "methods": [
       ("Base", hl("Yes", [("Yes", "red")])),
       ("AVCD", hl("Yes", [("Yes", "red")])),
       ("MAD", hl("Yes", [("Yes", "red")])),
       ("Ours", OURS_SLOT_YN),
     ]},
  ]},
}

CSS = """
<style>
* { box-sizing: border-box; }
body { font-family: 'Times New Roman', Times, serif; background:#fff; color:#111;
       margin:0; padding:22px; width:900px; }
.frames { display:flex; gap:3px; margin:0 0 3px 0; }
.frames img { height:132px; width:auto; object-fit:cover; flex:1 1 0; min-width:0; }
.box { border:1px solid #a8a8a8; border-radius:9px; padding:10px 14px; margin:5px 0; }
.ctx b { font-weight:bold; }
.ctx div { margin:3px 0; font-size:16px; line-height:1.35; }
.ans { background:#f2f2f2; position:relative; }
.ans .m { font-weight:bold; font-size:16px; margin-bottom:2px; }
.ans .t { font-size:15.5px; line-height:1.4; }
.tag { position:absolute; top:10px; right:14px; color:#d00; font-style:italic; font-size:14px; }
.red { color:#d40000; }
.blue { color:#1560d0; }
.blk { margin-bottom:20px; }
.cap { text-align:center; font-size:16px; margin-top:14px; }
.slot { color:#9a6b00; background:#fff8e6; padding:2px 6px; border-radius:4px;
        border:1px dashed #d9a441; font-size:14px; }
.ans.ours { background:#eef4ff; border-color:#4a6fb5; border-width:1.5px; }
.editnote { font-family:-apple-system,sans-serif; font-size:12.5px; color:#555;
            background:#f5f7fa; border-left:3px solid #4a6fb5; padding:9px 13px;
            margin-bottom:14px; line-height:1.5; }
</style>
"""

EDITNOTE = ("<div class=\"editnote\"><b>편집 안내</b> — 이 파일은 텍스트 에디터로 바로 수정할 수 있습니다. "
            "노란 <span class=\"slot\">[ … ]</span> 자리에 Ours 출력을 붙여넣으면 됩니다. "
            "강조는 <code>&lt;span class=\"red\"&gt;환각 구절&lt;/span&gt;</code> / "
            "<code>&lt;span class=\"blue\"&gt;올바른 구절&lt;/span&gt;</code>. "
            "브라우저로 열어 확인 후 인쇄(PDF)하거나 캡처해 논문에 넣으시면 됩니다. "
            "(이 안내 문단은 최종본에서 지우세요)</div>")

def render(fig, editable=True):
    parts = [CSS] + ([EDITNOTE] if editable else [])
    for b in fig["blocks"]:
        imgs = "".join(f'<img src="data:image/jpeg;base64,{d}">' for d in frames_b64(b["vid"]))
        parts.append(f'<div class="blk"><div class="frames">{imgs}</div>')
        parts.append('<div class="box ctx">'
                     f'<div><b>Video</b>: {html.escape(b["video"])}</div>'
                     f'<div><b>Sound</b>: {html.escape(b["sound"])}</div>'
                     f'<div><b>Question</b>: {html.escape(b["question"])}</div></div>')
        for j, (m, t) in enumerate(b["methods"]):
            tag = f'<div class="tag">{html.escape(b["tag"])}</div>' if j == 0 else ""
            cls = "box ans ours" if m == "Ours" else "box ans"
            parts.append(f'<div class="{cls}">{tag}<div class="m">{m}</div><div class="t">{t}</div></div>')
        parts.append('</div>')
    parts.append(f'<div class="cap">{html.escape(fig["caption"])}</div>')
    return "\n".join(parts)

for name, fig in FIGS.items():
    (OUT / f"{name}.html").write_text(render(fig), encoding="utf-8")
    print("작성:", OUT / f"{name}.html")

# 검수용 결합 파일
combined = CSS + '<div style="font-size:13px;color:#888;margin-bottom:8px;">[검수용 결합 — fig9 / fig10]</div>'
for name in ("fig9_videollama2", "fig10_qwen"):
    combined += render(FIGS[name]) + '<hr style="margin:30px 0;border:none;border-top:2px dashed #ccc;">'
(OUT / "_review_combined.html").write_text(combined, encoding="utf-8")
print("결합 검수:", OUT / "_review_combined.html")
