"""One-shot codemod: viewer phase B (zoom/pan, casualties, inspection, polish)."""
from pathlib import Path

# ---------- constants relocation ----------
text = Path("muster/sim/constants.py").read_text()
old = "INFLUENCE_FIXED_SCALE = 1 << 20"
assert text.count(old) == 1
text = text.replace(
    old,
    "INFLUENCE_FIXED_SCALE = 1 << 20\n\n# Per-soldier entity perception (used by the RL observation builder).\nENTITY_NEIGHBORS = 16\nENTITY_RADIUS = 5.0",
)
Path("muster/sim/constants.py").write_text(text)

text = Path("muster/rl/env.py").read_text()
old = "ENTITY_NEIGHBORS = 16\nENTITY_RADIUS = 5.0"
assert text.count(old) == 1
text = text.replace(old, "from muster.sim.constants import ENTITY_NEIGHBORS, ENTITY_RADIUS  # noqa: E402")
Path("muster/rl/env.py").write_text(text)

text = Path("muster/viewer/replay.py").read_text()
old = """from muster.sim.constants import (
    STRONGPOINT_CELLS,"""
assert text.count(old) == 1
text = text.replace(
    old,
    """from muster.sim.constants import (
    ENTITY_RADIUS,
    STRONGPOINT_CELLS,""",
)
old = '''                "strongpoint_weight": STRONGPOINT_WEIGHT,'''
assert text.count(old) >= 1
text = text.replace(old, old + '\n                "entity_radius": ENTITY_RADIUS,', 1)
Path("muster/viewer/replay.py").write_text(text)

# rollout.py replay config block also embeds a config dict
text = Path("muster/rl/rollout.py").read_text()
old = """from muster.sim.constants import (
    STRONGPOINT_CELLS,"""
assert text.count(old) == 1
text = text.replace(
    old,
    """from muster.sim.constants import (
    ENTITY_RADIUS,
    STRONGPOINT_CELLS,""",
)
old = '''                "strongpoint_weight": STRONGPOINT_WEIGHT,'''
assert text.count(old) == 1
text = text.replace(old, old + '\n                "entity_radius": ENTITY_RADIUS,', 1)
Path("muster/rl/rollout.py").write_text(text)

# ---------- template ----------
text = Path("muster/viewer/template.html").read_text()

# CSS + HTML: inspect overlay, trails button, hide extras on mobile
old = """  #update,#matchup,#status,#territoryStatus { white-space:nowrap; }"""
assert text.count(old) == 1
text = text.replace(
    old,
    """  #update,#matchup,#status,#territoryStatus { white-space:nowrap; }
  #inspect { position:fixed; right:10px; top:52px; background:rgba(10,14,18,.85); border:1px solid #444; padding:6px 9px; border-radius:4px; pointer-events:none; z-index:5; }""",
)
old = """    #status,#territoryStatus { display:none; }"""
assert text.count(old) == 1
text = text.replace(old, """    #status,#territoryStatus,#trails { display:none; }""")
old = """  <button id="territory">territory on (T)</button>"""
assert text.count(old) == 1
text = text.replace(
    old,
    """  <button id="territory">territory on (T)</button>
  <button id="trails">trails off (R)</button>""",
)
old = """<canvas id="world"></canvas>"""
assert text.count(old) == 1
text = text.replace(old, """<canvas id="world"></canvas>
<div id="inspect" hidden></div>""")

# state + death precompute (after territoryFrame block)
old = """const screenState=new Float32Array(soldierCount*4);
let cursor=0, playing=lastFrame>0, showTerritory=true, previous=performance.now();"""
assert text.count(old) == 1
text = text.replace(
    old,
    """const screenState=new Float32Array(soldierCount*4);
const deathFrame=new Uint16Array(soldierCount).fill(65535);
const deathX=new Float32Array(soldierCount), deathY=new Float32Array(soldierCount);
for (let i=0;i<soldierCount;i++) for (let f=1;f<frameCount;f++) {
  if (healths[f*soldierCount+i]===0 && healths[(f-1)*soldierCount+i]>0) {
    deathFrame[i]=f;
    deathX[i]=positions[(f*soldierCount+i)*2]*cfg.world_width/quantizedMax;
    deathY[i]=positions[(f*soldierCount+i)*2+1]*cfg.world_height/quantizedMax;
    break;
  }
}
const perceptionRadius=cfg.entity_radius||5;
let zoom=1, camX=cfg.world_width/2, camY=cfg.world_height/2;
let showTrails=false, selected=-1;
let viewLeft=0, viewBottom=0, viewScale=1, viewBase=1, viewDpr=1;
const trailsButton=document.querySelector("#trails");
const inspect=document.querySelector("#inspect");
let cursor=0, playing=lastFrame>0, showTerritory=true, previous=performance.now();""",
)

# matchup: mode badge
old = """  const roles=[opponent,opponent]; roles[replay.learner_team]="learner";
  const text=mobile?"B "+roles[0]+" · R "+roles[1]:"blue "+roles[0]+" vs red "+roles[1];"""
assert text.count(old) == 1
text = text.replace(
    old,
    """  const roles=[opponent,opponent]; roles[replay.learner_team]="learner";
  if (Number.isInteger(replay.learner_mode)) roles[replay.learner_team]+=mobile?"·m"+replay.learner_mode:" (mode "+replay.learner_mode+")";
  const text=mobile?"B "+roles[0]+" · R "+roles[1]:"blue "+roles[0]+" vs red "+roles[1];""",
)

# draw(): camera transform + capped territory resolution
old = """  const margin=12*dpr, scale=Math.min((width-2*margin)/cfg.world_width,(height-2*margin)/cfg.world_height);
  const left=(width-cfg.world_width*scale)/2, bottom=(height+cfg.world_height*scale)/2, top=bottom-cfg.world_height*scale;
  const worldWidth=Math.max(1,Math.round(cfg.world_width*scale));
  const worldHeight=Math.max(1,Math.round(cfg.world_height*scale));
  if (hexRadius) prepareTerritoryGeometry(worldWidth,worldHeight);"""
assert text.count(old) == 1
text = text.replace(
    old,
    """  const margin=12*dpr, baseScale=Math.min((width-2*margin)/cfg.world_width,(height-2*margin)/cfg.world_height);
  const scale=baseScale*zoom;
  camX=Math.max(0,Math.min(cfg.world_width,camX)); camY=Math.max(0,Math.min(cfg.world_height,camY));
  const left=width/2-camX*scale, bottom=height/2+camY*scale, top=bottom-cfg.world_height*scale;
  viewLeft=left; viewBottom=bottom; viewScale=scale; viewBase=baseScale; viewDpr=dpr;
  const territoryScale=Math.min(scale,2048/cfg.world_width), geometryRatio=scale/territoryScale;
  const worldWidth=Math.max(1,Math.round(cfg.world_width*territoryScale));
  const worldHeight=Math.max(1,Math.round(cfg.world_height*territoryScale));
  if (hexRadius) prepareTerritoryGeometry(worldWidth,worldHeight);""",
)

# clip block: apply ratio for geometry built at territoryScale
old = """  if (hexRadius&&territoryBoard) {
    ctx.translate(left,top); ctx.clip(territoryBoard); ctx.translate(-left,-top);
  }"""
assert text.count(old) == 1
text = text.replace(
    old,
    """  if (hexRadius&&territoryBoard) {
    ctx.translate(left,top); ctx.scale(geometryRatio,geometryRatio); ctx.clip(territoryBoard);
    ctx.scale(1/geometryRatio,1/geometryRatio); ctx.translate(-left,-top);
  }""",
)

# strongpoints with ratio transform
old = """  if (hexRadius&&strongpointCells.length) {
    ctx.save(); ctx.translate(left,top);
    ctx.fillStyle="rgba(255,196,64,.10)"; ctx.strokeStyle="#ffc440"; ctx.lineWidth=Math.max(1.5*dpr,2);"""
assert text.count(old) == 1
text = text.replace(
    old,
    """  if (hexRadius&&strongpointCells.length) {
    ctx.save(); ctx.translate(left,top); ctx.scale(geometryRatio,geometryRatio);
    ctx.fillStyle="rgba(255,196,64,.10)"; ctx.strokeStyle="#ffc440"; ctx.lineWidth=Math.max(1.5*dpr,2)/geometryRatio;""",
)

# trails + death markers before soldier fill; alive counters in interpolation loop
old = """  const radius=Math.max(2*dpr,cfg.soldier_radius*scale);
  const lowerBase=lower*soldierCount, upperBase=upper*soldierCount;
  for (let i=0;i<soldierCount;i++) {"""
assert text.count(old) == 1
text = text.replace(
    old,
    """  const radius=Math.max(2*dpr,cfg.soldier_radius*scale);
  if (showTrails) {
    ctx.lineWidth=Math.max(1,dpr*.8);
    const back=Math.max(0,lower-10);
    for (let team=0;team<2;team++) {
      ctx.strokeStyle=team===0?"rgba(53,167,255,.3)":"rgba(255,77,95,.3)";
      ctx.beginPath();
      for (let i=0;i<soldierCount;i++) if (teams[i]===team && healths[lower*soldierCount+i]>0) {
        let p=(back*soldierCount+i)*2;
        ctx.moveTo(left+positions[p]*cfg.world_width*scale/quantizedMax,bottom-positions[p+1]*cfg.world_height*scale/quantizedMax);
        for (let f=back+2;f<=lower;f+=2) {
          p=(f*soldierCount+i)*2;
          ctx.lineTo(left+positions[p]*cfg.world_width*scale/quantizedMax,bottom-positions[p+1]*cfg.world_height*scale/quantizedMax);
        }
      }
      ctx.stroke();
    }
  }
  const fadeFrames=15;
  ctx.lineWidth=Math.max(1,dpr);
  for (let i=0;i<soldierCount;i++) {
    const df=deathFrame[i];
    if (df===65535||df>lower||lower-df>=fadeFrames) continue;
    const age=(cursor-df)/fadeFrames, alpha=Math.max(0,.7*(1-age));
    if (alpha<=0) continue;
    ctx.strokeStyle=(teams[i]===0?"rgba(53,167,255,":"rgba(255,77,95,")+alpha.toFixed(3)+")";
    ctx.beginPath(); ctx.arc(left+deathX[i]*scale,bottom-deathY[i]*scale,radius*(1+age*1.5),0,2*Math.PI); ctx.stroke();
  }
  let alive0=0, alive1=0;
  const lowerBase=lower*soldierCount, upperBase=upper*soldierCount;
  for (let i=0;i<soldierCount;i++) {""",
)
old = """    screenState[output+3]=health;
    if (health<=0) continue;"""
assert text.count(old) == 1
text = text.replace(
    old,
    """    screenState[output+3]=health;
    if (health<=0) continue;
    if (teams[i]===0) alive0++; else alive1++;""",
)

# selection highlight + inspect overlay + status/banner
old = """  slider.value=Math.min(cursor,lastFrame);
  const ending=lower===lastFrame&&Number.isInteger(replay.winner)?(replay.winner<0?" | draw":" | "+(replay.winner===0?"blue":"red")+" wins"):"";
  setText(status,"step "+lower+" / "+lastFrame+ending);
}"""
assert text.count(old) == 1
text = text.replace(
    old,
    """  if (selected>=0) {
    const health=screenState[selected*4+3];
    if (health>0) {
      const x=screenState[selected*4], y=screenState[selected*4+1];
      ctx.strokeStyle="#ffffff"; ctx.lineWidth=Math.max(1.2*dpr,radius*.2);
      ctx.beginPath(); ctx.arc(x,y,radius*1.45,0,2*Math.PI); ctx.stroke();
      ctx.setLineDash([6*dpr,5*dpr]); ctx.strokeStyle="rgba(255,255,255,.45)"; ctx.lineWidth=Math.max(1,dpr);
      ctx.beginPath(); ctx.arc(x,y,perceptionRadius*scale,0,2*Math.PI); ctx.stroke(); ctx.setLineDash([]);
      const facing=-screenState[selected*4+2];
      ctx.fillStyle=teams[selected]===0?"rgba(53,167,255,.22)":"rgba(255,77,95,.22)";
      ctx.beginPath(); ctx.moveTo(x,y); ctx.arc(x,y,perceptionRadius*scale*.5,facing-Math.PI/3,facing+Math.PI/3); ctx.closePath(); ctx.fill();
      const degrees=((Math.round(screenState[selected*4+2]*180/Math.PI)%360)+360)%360;
      inspect.textContent=(teams[selected]===0?"blue":"red")+" #"+selected+" · hp "+Math.round(health*100)+"% · facing "+degrees+"°";
      inspect.hidden=false;
    } else {
      inspect.textContent=(teams[selected]===0?"blue":"red")+" #"+selected+" · down";
      inspect.hidden=false;
    }
  } else inspect.hidden=true;
  slider.value=Math.min(cursor,lastFrame);
  const ending=lower===lastFrame&&Number.isInteger(replay.winner)?(replay.winner<0?" | draw":" | "+(replay.winner===0?"blue":"red")+" wins"):"";
  setText(status,"t="+(lower*cfg.decision_dt).toFixed(1)+"s · step "+lower+"/"+lastFrame+" · blue "+alive0+" · red "+alive1+ending);
  if (lower===lastFrame&&Number.isInteger(replay.winner)) {
    const verdict=replay.winner<0?"DRAW":replay.winner===0?"BLUE WINS":"RED WINS";
    ctx.font="bold "+Math.round(26*dpr)+"px monospace"; ctx.textAlign="center";
    ctx.fillStyle=replay.winner===0?"#35a7ff":replay.winner===1?"#ff4d5f":"#dddddd";
    ctx.globalAlpha=.92; ctx.fillText(verdict,width/2,Math.max(top,20*dpr)+34*dpr);
    ctx.globalAlpha=1; ctx.textAlign="left";
  }
}""",
)

# input: trails toggle, reset, wheel zoom, pointer pan/pinch/select
old = """play.onclick=toggle;"""
assert text.count(old) == 1
text = text.replace(
    old,
    """function toggleTrails() {
  showTrails=!showTrails;
  setText(trailsButton,showTrails?"trails on (R)":"trails off (R)");
  invalidate();
}
function resetView() { zoom=1; camX=cfg.world_width/2; camY=cfg.world_height/2; invalidate(); }
function zoomAt(screenX,screenY,factor) {
  const worldX=(screenX-viewLeft)/viewScale, worldY=(viewBottom-screenY)/viewScale;
  zoom=Math.max(1,Math.min(16,zoom*factor));
  const scale=viewBase*zoom;
  camX=worldX-(screenX-canvas.width/2)/scale;
  camY=worldY+(screenY-canvas.height/2)/scale;
  if (zoom===1) { camX=cfg.world_width/2; camY=cfg.world_height/2; }
  invalidate();
}
canvas.addEventListener("wheel",e=>{
  e.preventDefault();
  const rect=canvas.getBoundingClientRect();
  zoomAt((e.clientX-rect.left)*viewDpr,(e.clientY-rect.top)*viewDpr,Math.exp(-e.deltaY*.0015));
},{passive:false});
const pointers=new Map();
let dragDistance=0, pinchDistance=0;
canvas.addEventListener("pointerdown",e=>{
  canvas.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
  dragDistance=0;
  if (pointers.size===2) {
    const p=[...pointers.values()];
    pinchDistance=Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y);
  }
});
canvas.addEventListener("pointermove",e=>{
  const entry=pointers.get(e.pointerId); if (!entry) return;
  const dx=e.clientX-entry.x, dy=e.clientY-entry.y;
  entry.x=e.clientX; entry.y=e.clientY;
  if (pointers.size===1) {
    dragDistance+=Math.hypot(dx,dy);
    camX-=dx*viewDpr/viewScale; camY+=dy*viewDpr/viewScale;
    invalidate();
  } else if (pointers.size===2) {
    dragDistance=100;
    const p=[...pointers.values()];
    const distance=Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y);
    if (pinchDistance>0) {
      const rect=canvas.getBoundingClientRect();
      zoomAt(((p[0].x+p[1].x)/2-rect.left)*viewDpr,((p[0].y+p[1].y)/2-rect.top)*viewDpr,distance/pinchDistance);
    }
    pinchDistance=distance;
  }
});
function endPointer(e) {
  if (!pointers.has(e.pointerId)) return;
  pointers.delete(e.pointerId);
  pinchDistance=0;
  if (pointers.size===0 && dragDistance<5) {
    const rect=canvas.getBoundingClientRect();
    const screenX=(e.clientX-rect.left)*viewDpr, screenY=(e.clientY-rect.top)*viewDpr;
    let best=-1, bestDistance=Math.max(24*viewDpr,cfg.soldier_radius*viewScale*2.5);
    for (let i=0;i<soldierCount;i++) if (screenState[i*4+3]>0) {
      const separation=Math.hypot(screenState[i*4]-screenX,screenState[i*4+1]-screenY);
      if (separation<bestDistance) { best=i; bestDistance=separation; }
    }
    selected=best;
    invalidate();
  }
}
canvas.addEventListener("pointerup",endPointer);
canvas.addEventListener("pointercancel",endPointer);
trailsButton.onclick=toggleTrails;
play.onclick=toggle;""",
)
old = """  else if(e.key.toLowerCase()==="t")toggleTerritory();"""
assert text.count(old) == 1
text = text.replace(
    old,
    """  else if(e.key.toLowerCase()==="t")toggleTerritory();
  else if(e.key.toLowerCase()==="r")toggleTrails();
  else if(e.key==="0")resetView();""",
)

Path("muster/viewer/template.html").write_text(text)
print("viewer phase B applied")
