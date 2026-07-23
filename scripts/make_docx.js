const d = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
        ShadingType, HeadingLevel, AlignmentType, BorderStyle, ExternalHyperlink } = d;
const fs = require("fs");

const NAVY = "1E2761", GREEN = "1E8449", RED = "C0392B", AMBER = "9A6B00", MUT = "6B7280";
const HDBG = "1E2761", GBG = "E8F5E9", ABG = "FDF3E2", ZBG = "F4F7FC";
const F = "맑은 고딕";

const W = [1500, 950, 880, 880, 950, 1000, 900, 900, 1066]; // 합 9026 DXA
const TW = W.reduce((a, b) => a + b, 0);

function p(text, o = {}) {
  return new Paragraph({
    spacing: { before: o.before ?? 0, after: o.after ?? 100 },
    alignment: o.align,
    children: [new TextRun({ text, font: F, size: o.size ?? 20, bold: o.bold,
                             color: o.color, italics: o.italics })],
  });
}
function rich(runs, o = {}) {
  return new Paragraph({
    spacing: { before: o.before ?? 0, after: o.after ?? 100 },
    children: runs.map(r => new TextRun({ text: r.t, font: F, size: r.size ?? 20,
                                          bold: r.b, color: r.c, italics: r.i })),
  });
}
function h(text, lvl) {
  return new Paragraph({
    heading: lvl, spacing: { before: 260, after: 130 },
    children: [new TextRun({ text, font: F, bold: true, size: lvl === HeadingLevel.HEADING_1 ? 28 : 24, color: NAVY })],
  });
}
function cell(text, o = {}) {
  return new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.bg ? { type: ShadingType.CLEAR, fill: o.bg, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({
      alignment: o.align ?? AlignmentType.CENTER, spacing: { before: 0, after: 0 },
      children: [new TextRun({ text, font: F, size: o.size ?? 16, bold: o.bold, color: o.color })],
    })],
  });
}
function bullet(label, body) {
  return new Paragraph({
    spacing: { after: 90 }, bullet: { level: 0 },
    children: [
      new TextRun({ text: label, font: F, size: 20, bold: true, color: NAVY }),
      new TextRun({ text: body, font: F, size: 20 }),
    ],
  });
}
function link(label, url) {
  return new ExternalHyperlink({
    link: url,
    children: [new TextRun({ text: label, font: F, size: 19, color: "1560D0", underline: {} })],
  });
}

// ── 수치표 ──
const COLS = ["Model", "Method", "Visual", "Audio", "Language", "CMM Overall", "Video-Driven", "Audio-Driven", "AVH Overall"];
const ROWS = [
  ["VideoLLaMA2-AV", "Baseline", "72.5", "79.2", "68.8", "73.5", "75.6", "78.8", "76.7", GBG],
  ["", "+ VCD", "69.0", "83.0", "77.3", "76.4", "69.6", "78.5", "72.5", GBG],
  ["", "+ AVCD", "71.8", "81.8", "70.5", "74.7", "76.3", "79.2", "77.2", GBG],
  ["", "+ MAD", "82.3", "83.3", "76.5", "80.7", "78.1", "78.4", "78.2", GBG],
  ["Qwen2.5-Omni-7B", "Baseline", "69.0", "70.0", "81.2", "73.4", "73.1", "80.3", "75.5", GBG],
  ["", "+ VCD", "64.5", "69.5", "82.2", "72.1", "67.4", "74.7", "69.8", ABG],
  ["", "+ AVCD", "70.8", "70.8", "82.2", "74.6", "71.4", "80.3", "74.3", ABG],
  ["", "+ MAD", "76.8", "84.2", "82.0", "81.0", "78.7", "84.7", "80.7", GBG],
];
const table = new Table({
  columnWidths: W, width: { size: TW, type: WidthType.DXA },
  rows: [
    new TableRow({ tableHeader: true, children: COLS.map((c, i) =>
      cell(c, { w: W[i], bg: HDBG, color: "FFFFFF", bold: true, size: 15 })) }),
    ...ROWS.map(r => new TableRow({ children: [
      cell(r[0], { w: W[0], bg: r[9], bold: true, align: AlignmentType.LEFT, size: 15 }),
      cell(r[1], { w: W[1], bg: r[9], bold: true, align: AlignmentType.LEFT, size: 15 }),
      ...r.slice(2, 9).map((v, i) => cell(v, { w: W[i + 2], bg: r[9], size: 15 })),
    ] })),
  ],
});

// ── 세팅표 ──
const SW = [1900, 7126];
const SET = [
  ["실행 환경", "RTX 3090 24GB ×1 (연구실 클러스터) · conda 2종 = 선배님 환경 복제본 (Qwen: transformers 4.52.3/torch 2.5.1, VideoLLaMA2: 4.42.3/2.2.0)"],
  ["모델 체크포인트", "VideoLLaMA2.1-7B-AV, Qwen2.5-Omni-7B — 선배님 HF 캐시 스냅샷 그대로 사용 (bf16, Qwen talker 미생성)"],
  ["벤치마크", "AVHBench 3,419문항 (Video-driven 2,287 + Audio-driven 1,132, MAD 프로토콜) · CMM 1,200문항 (Visual/Audio/Language Dominance 각 400)"],
  ["디코딩·채점", "greedy (do_sample=False), max_new_tokens=1, seed 42 · 채점은 MAD score.py의 extract_answer/normalize_answer 그대로"],
  ["오디오 입력", "AVHBench: mp4 muxed 트랙 · CMM: 무음 mp4 + 별도 wav — 전 branch에 실제 오디오가 공급되도록 구성"],
  ["하이퍼파라미터", "MAD γ=2.5 · VCD-ext α=0.5 · AVCD α (AVH 2.5 / CMM: VL2 2.5, Qwen 0.5), EAD τ=0.6 — 200문항 게이트 그리드로 확정"],
];
const setTable = new Table({
  columnWidths: SW, width: { size: SW[0] + SW[1], type: WidthType.DXA },
  rows: SET.map(([k, v]) => new TableRow({ children: [
    cell(k, { w: SW[0], bg: HDBG, color: "FFFFFF", bold: true, size: 17 }),
    cell(v, { w: SW[1], bg: ZBG, align: AlignmentType.LEFT, size: 17 }),
  ] })),
});

// ── 코드 경로표 ──
const CW = [3100, 1250, 4676];
const CODE = [
  ["src/adapters/qwen_omni.py", "신규", "Qwen2.5-Omni 어댑터 — AVCD용 4-branch/마스킹 forward를 새로 포팅 (원 구현 비공개)", true],
  ["src/adapters/attn_patch.py", "신규", "Qwen thinker attention 패치 — AVCD dominance 기록·attentive masking 삽입, 장문 대비 head/q-block 청킹", true],
  ["src/adapters/common.py", "신규", "AVCD 공통 수식 — modality dominance 계산, attentive row masking(재정규화 포함)", true],
  ["src/methods/vcd_ext.py", "신규", "VCD-extended — MAD repo에 dead code만 존재해 논문 Eq.10 기준으로 새로 구현", true],
  ["src/methods/avcd.py", "신규", "AVCD 디코딩 플러그인 — EAD/plausibility 포함, 두 모델 공통 인터페이스", true],
  ["src/methods/mad.py", "이식", "MAD 4-branch 대비 디코딩 (논문 Eq.9 가중식)", false],
  ["src/adapters/videollama2_av.py", "이식", "VideoLLaMA2-AV 어댑터 (MAD/AVCD 공식 fork 백엔드 연결)", false],
  ["configs/unified_settings.yaml", "설정", "모든 실험 세팅의 단일 소스 — 하드코딩 없음", false],
  ["src/runner.py · score.py · aggregate.py", "공통", "실행/채점/집계 파이프라인 (per-sample JSONL, 중단-재개 지원)", false],
];
const codeTable = new Table({
  columnWidths: CW, width: { size: CW[0] + CW[1] + CW[2], type: WidthType.DXA },
  rows: [
    new TableRow({ tableHeader: true, children: [
      cell("파일 경로", { w: CW[0], bg: HDBG, color: "FFFFFF", bold: true, size: 16 }),
      cell("구분", { w: CW[1], bg: HDBG, color: "FFFFFF", bold: true, size: 16 }),
      cell("역할", { w: CW[2], bg: HDBG, color: "FFFFFF", bold: true, size: 16 }),
    ] }),
    ...CODE.map(([f, kind, desc, isNew]) => new TableRow({ children: [
      cell(f, { w: CW[0], bg: isNew ? ABG : ZBG, align: AlignmentType.LEFT, size: 15 }),
      cell(kind, { w: CW[1], bg: isNew ? ABG : ZBG, size: 15, bold: isNew, color: isNew ? AMBER : undefined }),
      cell(desc, { w: CW[2], bg: isNew ? ABG : ZBG, align: AlignmentType.LEFT, size: 15 }),
    ] })),
  ],
});

const doc = new Document({
  sections: [{
    properties: { page: { margin: { top: 1000, bottom: 1000, left: 1100, right: 1100 } } },
    children: [
      p("Omni-Steering 비교군 재실험 — 최종 결과 보고", { size: 32, bold: true, color: NAVY, after: 60 }),
      p("VideoLLaMA2-AV · Qwen2.5-Omni-7B  ×  Base / VCD / AVCD / MAD  ×  CMM · AVHBench", { size: 19, color: MUT, after: 240 }),

      h("1. 개요", HeadingLevel.HEADING_1),
      p("Ours와 동일한 조건(동일 체크포인트·디코딩·채점·입력 규약)에서 비교군 4종을 전량 재실험하여, Ours 결과와 직접 비교 가능한 수치를 확보했습니다. 총 18,476건의 예측을 수행했습니다.", { after: 80 }),
      rich([
        { t: "핵심: ", b: true, c: GREEN },
        { t: "Baseline은 선배님 Ours(Base) 실측과 전 지표 ±1.4%p 이내로 일치하며, 동일 샘플 단위 예측 일치율은 98~99.5%입니다. 즉 이 표의 모든 행은 Ours와 같은 기준에서 측정된 값입니다." },
      ], { after: 160 }),

      h("2. 최종 수치표 (Accuracy, %)", HeadingLevel.HEADING_1),
      table,
      p("", { after: 60 }),
      rich([
        { t: "MAD 논문 Table 1 대비: ", b: true },
        { t: "Baseline·MAD는 논문값과 ±1.3%p 이내로 정합하고(구현 정확성 확인), VCD·AVCD는 원 구현이 불완전/비공개여서 새로 구현한 부분이라 2~4%p 차이가 있습니다(노란 배경). VideoLLaMA2의 CMM VCD/MAD는 전 branch에 실제 오디오가 공급되도록 입력 구성을 교정한 결과 논문값(76.4 / 81.3)과 사실상 일치합니다(76.4 / 80.7).", size: 18 },
      ], { after: 200 }),

      h("3. 실험 세팅", HeadingLevel.HEADING_1),
      setTable,
      p("", { after: 120 }),
      p("모든 세팅은 configs/unified_settings.yaml 한 곳에서 관리되며, 실행마다 seed와 config_hash가 결과 파일에 기록됩니다(재현성 보장).", { size: 18, color: MUT, after: 200 }),

      h("4. 코드 (GitHub)", HeadingLevel.HEADING_1),
      new Paragraph({ spacing: { after: 140 }, children: [
        new TextRun({ text: "저장소: ", font: F, size: 20, bold: true }),
        link("https://github.com/hsmai/Hallucination", "https://github.com/hsmai/Hallucination"),
      ] }),
      rich([{ t: "노란 배경 = 이번에 새로 구현한 부분입니다. ", b: true, c: AMBER },
            { t: "특히 AVCD × Qwen2.5-Omni는 원저자 구현이 공개되어 있지 않고, VCD-extended는 MAD repo에 미사용 코드만 있어 논문 수식 기준으로 새로 작성했습니다. 두 구역 모두 수식 단위테스트 → 로컬 통합테스트 → 서버 스모크 → 게이트 수치 대조의 4단 검증을 거쳤습니다.", size: 18 }],
        { after: 140 }),
      codeTable,
      p("", { after: 140 }),

      h("5. 재현 방법", HeadingLevel.HEADING_1),
      p("git clone 후, 아래 형식으로 임의의 (모델 × 방법 × 벤치) 조합을 재실행할 수 있습니다.", { after: 80 }),
      new Paragraph({
        spacing: { after: 60 },
        shading: { type: ShadingType.CLEAR, fill: "F2F4F8", color: "auto" },
        border: { left: { style: BorderStyle.SINGLE, size: 12, color: NAVY, space: 6 } },
        children: [new TextRun({
          text: "python -m src.runner --model videollama2_av --method mad --benchmark cmm",
          font: "Courier New", size: 18 })],
      }),
      new Paragraph({
        spacing: { after: 140 },
        shading: { type: ShadingType.CLEAR, fill: "F2F4F8", color: "auto" },
        border: { left: { style: BorderStyle.SINGLE, size: 12, color: NAVY, space: 6 } },
        children: [new TextRun({
          text: "python -m src.score --jsonl 'results/runs/*/*.jsonl' && python -m src.aggregate --results results/runs",
          font: "Courier New", size: 18 })],
      }),
      p("산출물: results/runs/{benchmark}/{model}__{method}.jsonl (문항별 예측·정오·중간값) → results/runs/tables/main_table.{md,csv} (위 표).", { size: 18, color: MUT, after: 200 }),

      h("6. 참고 사항", HeadingLevel.HEADING_1),
      bullet("정성 비교 자료: ", "MAD 논문 Fig.9-10 형식의 비교 figure를 별도 HTML로 제공합니다(모델별 1개). Ours 출력을 붙여 넣을 수 있도록 빈 칸을 만들어 두었으며, 텍스트 편집기로 바로 수정 가능합니다."),
      bullet("속도 비교: ", "Ours와 비교군의 실행 환경(GPU)이 달라 지연시간(latency) 수치는 직접 비교에 적합하지 않습니다. 정확도 비교에는 영향이 없습니다."),
      bullet("제외 문항: ", "CMM에서 디코더 오류를 일으키는 손상 mp4 8건은 재인코딩본으로 대체했습니다(Ours와 동일 처리)."),
    ],
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("/Users/hansangmin/Hallucination/docs/선배보고_최종수치표.docx", b);
  console.log("작성 완료");
});
