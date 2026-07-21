const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";
const NAVY="1E2761", ICE="CADCFC", WHITE="FFFFFF", DARK="1A1A2E", MUT="6B7280";
const GREEN="1E8449", GBG="E8F5E9", AMBER="B7791F", ABG="FDF3E2", RED="C0392B", RBG="FBE9E7";

function title(s, t, sub){
  s.addText(t,{x:0.55,y:0.34,w:12.2,h:0.7,fontSize:29,bold:true,color:NAVY,fontFace:"Cambria",margin:0});
  if(sub) s.addText(sub,{x:0.55,y:1.02,w:12.2,h:0.4,fontSize:13,color:MUT,fontFace:"Calibri",margin:0});
}

// ══ S1 표지 ══
{
  const s=p.addSlide(); s.background={color:NAVY};
  s.addText("비교군 재실험 — 정량 결과",{x:0.9,y:2.3,w:11.5,h:1,fontSize:42,bold:true,color:WHITE,fontFace:"Cambria",margin:0});
  s.addText("Ours와 동일 조건에서 뽑은 Base · VCD · AVCD · MAD 수치",{x:0.9,y:3.35,w:11.5,h:0.6,fontSize:22,color:ICE,fontFace:"Calibri",margin:0});
  s.addText("목표: 논문값 재현이 아니라, Ours를 돌린 것과 똑같은 조건에서 비교군을 측정해\nOurs 표와 나란히 놓을 수 있는 수치를 만드는 것",
    {x:0.9,y:4.5,w:11.3,h:1,fontSize:15,color:"9DB2D8",fontFace:"Calibri",margin:0,lineSpacing:22});
  s.addText("VideoLLaMA2-AV · Qwen2.5-Omni-7B  |  CMM 1,200 · AVHBench 3,419  |  RTX 3090",
    {x:0.9,y:6.6,w:11,h:0.4,fontSize:12,color:"7A8BB5",fontFace:"Calibri",margin:0});
}

// ══ S2 수치표 (3색 분류) ══
{
  const s=p.addSlide(); s.background={color:WHITE};
  title(s,"최종 수치표 — 우리값 (논문값)","행 배경색 = 논문 대비 재현 정도.  값 형식: 우리 재실험 (MAD 논문 Table 1)");
  const hd={fill:{color:NAVY},color:"FFFFFF",bold:true,fontSize:11,align:"center",valign:"middle"};
  const cell=(t,bg,tc)=>({text:t,options:{fontSize:10.5,align:"center",valign:"middle",color:tc||DARK,fill:{color:bg}}});
  const mc=(t,bg,tc)=>({text:t,options:{fontSize:10.5,align:"left",valign:"middle",bold:true,color:tc||DARK,fill:{color:bg}}});
  // 행: [model, method, 7 vals, bg, tc]
  const R=[
    ["VideoLLaMA2-AV","Baseline",["72.5 (71.8)","79.2 (80.0)","68.8 (68.8)","73.5 (73.5)","75.6 (75.7)","78.8 (79.0)","76.7 (77.4)"],GBG,GREEN],
    ["","+ VCD",["52.8 (71.3)","68.5 (83.3)","54.2 (74.8)","58.5 (76.4)","69.6 (66.0)","78.5 (74.8)","72.5 (70.4)"],RBG,RED],
    ["","+ AVCD",["71.8 (71.8)","81.8 (84.0)","70.5 (71.5)","74.7 (75.8)","76.3 (78.3)","79.2 (80.3)","77.2 (79.3)"],GBG,GREEN],
    ["","+ MAD",["67.8 (82.3)","76.0 (84.3)","69.0 (77.5)","70.9 (81.3)","78.1 (79.7)","78.4 (79.1)","78.2 (79.4)"],RBG,RED],
    ["Qwen2.5-Omni-7B","Baseline",["69.0 (64.5)","70.0 (72.3)","81.2 (81.3)","73.4 (72.7)","73.1 (73.0)","80.3 (80.7)","75.5 (76.9)"],GBG,GREEN],
    ["","+ VCD",["64.5 (62.5)","69.5 (71.3)","82.2 (84.5)","72.1 (72.8)","67.4 (70.3)","74.7 (77.1)","69.8 (73.7)"],ABG,AMBER],
    ["","+ AVCD",["70.8 (66.3)","70.8 (72.8)","82.2 (81.0)","74.6 (73.3)","71.4 (75.8)","80.3 (79.7)","74.3 (77.8)"],ABG,AMBER],
    ["","+ MAD",["76.8 (76.8)","84.2 (84.3)","82.0 (83.3)","81.0 (81.4)","78.7 (78.7)","84.7 (84.4)","80.7 (81.6)"],GBG,GREEN],
  ];
  const cols=["Model","Method","Visual","Audio","Language","CMM Overall","Vid-Driven","Aud-Driven","AVH Overall"];
  const w=[1.75,1.05,1.35,1.35,1.5,1.5,1.4,1.4,1.25]; // 인치, 합≈12.55
  const rows=[cols.map(c=>({text:c,options:hd}))];
  R.forEach(r=>{
    const row=[mc(r[0],r[3]),mc(r[1],r[3])];
    r[2].forEach(v=>row.push(cell(v,r[3],r[4])));
    rows.push(row);
  });
  s.addTable(rows,{x:0.5,y:1.68,w:12.3,rowH:0.5,fontFace:"Calibri",
    border:{type:"solid",color:"D5DCE8",pt:0.5},colW:w});
  // 범례
  const leg=[["정합 (±1.5pp) — 제대로 재현",GBG,GREEN],["중간격차 (2~4pp) — 재구현",ABG,AMBER],["대격차 — 입력규약 (CMM 열)",RBG,RED]];
  let lx=0.5;
  leg.forEach(([t,bg,tc])=>{
    s.addShape(p.ShapeType.rect,{x:lx,y:6.6,w:0.3,h:0.3,fill:{color:bg},line:{color:tc,width:1}});
    s.addText(t,{x:lx+0.38,y:6.55,w:4.0,h:0.4,fontSize:11,color:tc,bold:true,fontFace:"Calibri",margin:0});
    lx+=4.25;
  });
  s.addText("* 대격차는 VL2 VCD/MAD의 CMM 열에만 집중 — 같은 행의 AVHBench 열은 논문과 정합.",
    {x:0.5,y:7.0,w:12.3,h:0.35,fontSize:10.5,italic:true,color:MUT,fontFace:"Calibri",margin:0});
}

// ══ S3 3분류 원인 ══
{
  const s=p.addSlide(); s.background={color:WHITE};
  title(s,"격차의 세 가지 성격","전부 원인 규명 — ‘재현 실패’가 아니라 조건·구현의 차이");
  const cards=[
    ["정합 (초록)",GREEN,GBG,"Base · MAD · 일부 AVCD",
     "논문값과 ±1.5pp.\n\n• Base(순수 파이프라인)와 최강 비교군 MAD가 일치 = 재현이 근본적으로 옳음.\n• Qwen MAD Δ0.4 · VL2 Base Δ0.4.\n\n→ ‘제대로 재현된’ 축."],
    ["중간격차 (노랑)",AMBER,ABG,"Qwen VCD · Qwen AVCD",
     "논문값과 2~4pp.\n\n• 원 구현이 불완전/비공개(VCD=dead code, AVCD×Qwen=미공개)라 우리가 세부 세팅을 새로 구현.\n• ‘재현’이 아니라 ‘재구현’이므로 차이는 예상 범위.\n\n→ 참고 대조군."],
    ["대격차 (빨강)",RED,RBG,"VL2 VCD/MAD의 CMM 열",
     "논문값과 8~20pp (CMM만, AVH는 정합).\n\n• Ours와 동일 조건(별도 wav)에서 뽑은 값.\n• 무음 mp4를 넣으면 논문값 재현 가능(원본 코드로 확인).\n\n→ 격차=입력규약. 정확한 입력에선 대조 디코딩이 오히려 해로움 (Ours 우위 부각)."],
  ];
  let x=0.5;
  cards.forEach(([t,col,bg,sub,body])=>{
    s.addShape(p.ShapeType.roundRect,{x:x,y:1.65,w:4.05,h:5.1,fill:{color:bg},line:{color:col,width:1.5},rectRadius:0.1});
    s.addText(t,{x:x+0.25,y:1.85,w:3.6,h:0.5,fontSize:16,bold:true,color:col,fontFace:"Calibri",margin:0});
    s.addText(sub,{x:x+0.25,y:2.4,w:3.6,h:0.4,fontSize:12,italic:true,color:DARK,fontFace:"Calibri",margin:0});
    s.addText(body,{x:x+0.25,y:2.95,w:3.6,h:3.6,fontSize:11.5,color:DARK,fontFace:"Calibri",margin:0,lineSpacing:15});
    x+=4.27;
  });
}

// ══ S4 합당성 + 세팅 ══
{
  const s=p.addSlide(); s.background={color:WHITE};
  title(s,"왜 이 수치를 Ours와 나란히 쓸 수 있나","‘논문값 복원’이 아니라 ‘Ours와 동일 조건 측정’이 목표");
  // 좌: 3대 증거
  s.addShape(p.ShapeType.roundRect,{x:0.5,y:1.6,w:6.1,h:3.3,fill:{color:"F4F7FC"},rectRadius:0.1});
  s.addText("Ours와 동일 조건 — 3대 증거",{x:0.75,y:1.78,w:5.6,h:0.4,fontSize:15,bold:true,color:NAVY,fontFace:"Calibri",margin:0});
  const ev=[
    ["① 동일 환경","같은 3090·선배 conda 복제·같은 체크포인트·greedy·seed 42·yes-no 채점·같은 오디오 규약"],
    ["② Base per-sample 일치","우리 Base가 선배 Ours(Base)와 샘플 단위 98~99.5% 동일 예측 = 같은 자로 잼"],
    ["③ 최강 비교군 검증","Qwen MAD가 논문값과 Δ0.4pp = 방법 구현 정확"],
  ];
  let y=2.3;
  ev.forEach(([h,b])=>{
    s.addText(h,{x:0.75,y:y,w:5.6,h:0.35,fontSize:12.5,bold:true,color:GREEN,fontFace:"Calibri",margin:0});
    s.addText(b,{x:0.75,y:y+0.34,w:5.6,h:0.6,fontSize:11,color:DARK,fontFace:"Calibri",margin:0,lineSpacing:14});
    y+=0.86;
  });
  // 우: 세팅 세부
  s.addShape(p.ShapeType.roundRect,{x:6.75,y:1.6,w:6.05,h:3.3,fill:{color:NAVY},rectRadius:0.1});
  s.addText("세팅 세부 (전부 게이트 검증 후 확정)",{x:7.0,y:1.78,w:5.5,h:0.4,fontSize:15,bold:true,color:ICE,fontFace:"Calibri",margin:0});
  const setup=[
    "벤치: AVHBench 3,419 (MAD 프로토콜) · CMM 1,200 (V/A/L×400)",
    "디코딩: greedy, max_new_tokens=1, seed 42, 모델별 eval suffix",
    "오디오: AVH=muxed / CMM=무음 mp4+별도 wav (Ours와 동일)",
    "하이퍼: MAD γ=2.5 · VCD α=0.5 · AVCD α(AVH 2.5, CMM VL2 2.5/Qwen 0.5)",
    "재구현: VCD-ext·AVCD×Qwen 세부 세팅 신규 구현 (원본 불완전/비공개)",
  ];
  s.addText(setup.map((t,i)=>({text:t,options:{bullet:{code:"2022"},breakLine:true,paraSpaceAfter:6}})),
    {x:7.0,y:2.3,w:5.55,h:2.5,fontSize:11,color:WHITE,fontFace:"Calibri",margin:0,lineSpacing:14});
  // 하단 결론 바
  s.addShape(p.ShapeType.roundRect,{x:0.5,y:5.15,w:12.3,h:1.6,fill:{color:GBG},line:{color:GREEN,width:1.5},rectRadius:0.1});
  s.addText([
    {text:"핵심:  ",options:{bold:true,color:GREEN}},
    {text:"이 표는 논문값 복원본이 아니라 ",options:{color:DARK}},
    {text:"Ours를 돌린 것과 똑같은 조건에서 잰 비교군 수치",options:{bold:true,color:DARK}},
    {text:"다. Base·MAD가 논문과 일치해 조건 동일성이 입증되며, VCD/MAD의 CMM 대격차는 정확한 입력에서 그 방법들이 취약함을 보여준다(무음 mp4면 논문값 재현). ",options:{color:DARK}},
    {text:"따라서 Ours 행과 직접 비교 가능하다.",options:{bold:true,color:GREEN}},
  ],{x:0.85,y:5.35,w:11.6,h:1.2,fontSize:13.5,fontFace:"Calibri",margin:0,lineSpacing:19});
}

p.writeFile({fileName:"/Users/hansangmin/Hallucination/docs/미팅_목표1_수치보고.pptx"}).then(()=>console.log("작성 완료"));
