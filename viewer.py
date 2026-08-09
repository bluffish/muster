"""Minimal pygame viewer.

Use ``Viewer.draw(sim.snapshot(0))`` inside any training loop, or run this file
directly for a nearest-enemy-policy demo.  Drawing is read-only and entirely
optional; call it infrequently during high-throughput training.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from simulator import (
    STRONGPOINT_CELLS,
    STRONGPOINT_WEIGHT,
    TERRITORY_CELLS,
    TERRITORY_COORDINATES,
    TERRITORY_RADIUS,
    Config,
    CpuSimulator,
    strongpoint_world_centers,
    territory_centers,
    territory_indices,
)

Snapshot = Mapping[str, object]
Policy = Callable[[Snapshot, Config], np.ndarray]

_REPLAY_HTML = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simulator replay</title>
<style>
  html,body { width:100%; height:100%; margin:0; overflow:hidden; overscroll-behavior:none; background:#141414; color:#ddd; font:14px monospace; }
  #tools { height:42px; display:flex; align-items:center; gap:10px; padding:0 12px; }
  #frame { flex:1; min-width:60px; }
  button,select,input { font:inherit; }
  #update,#matchup,#status,#territoryStatus { white-space:nowrap; }
  canvas { display:block; width:100%; height:calc(100vh - 42px); height:calc(100dvh - 42px); touch-action:none; }
  @media (max-width:700px), (pointer:coarse) {
    #tools { gap:5px; padding:0 6px; }
    #frame { min-width:24px; }
    #status,#territoryStatus { display:none; }
    button,select { padding:3px 5px; font-size:12px; }
  }
</style>
<div id="tools">
  <button id="current">load current</button>
  <button id="play">pause</button>
  <button id="territory">territory on (T)</button>
  <span id="update"><label for="updates">update</label> <select id="updates"></select></span>
  <span id="matchup"></span>
  <input id="frame" type="range" min="0" value="0" step="any">
  <select id="speed"><option value=".25">0.25x</option><option value=".5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select>
  <span id="status"></span>
  <span id="territoryStatus"></span>
</div>
<canvas id="world"></canvas>
<script>
const replay=__REPLAY__, cfg=replay.config, teams=replay.team;
const mobile=matchMedia("(pointer:coarse)").matches||innerWidth<=700;
const maxDpr=mobile?1:2, targetFrameMs=1000/(mobile?20:30);
function decode(text,Type) {
  const raw=atob(text), bytes=new Uint8Array(raw.length);
  for (let i=0;i<raw.length;i++) bytes[i]=raw.charCodeAt(i);
  return new Type(bytes.buffer);
}
const positions=decode(replay.position_u16,Uint16Array);
const angles=decode(replay.angle_i16,Int16Array);
const healths=decode(replay.health_u16,Uint16Array);
const territoryOwners=replay.territory_i8?decode(replay.territory_i8,Int8Array):null;
const controlShares=replay.control_u8?decode(replay.control_u8,Uint8Array):null;
delete replay.position_u16; delete replay.angle_i16; delete replay.health_u16; delete replay.territory_i8; delete replay.control_u8;
const frameCount=replay.frame_count, soldierCount=teams.length, lastFrame=frameCount-1;
const quantizedMax=65535, angleScale=Math.PI/32767;
const canvas=document.querySelector("#world"), ctx=canvas.getContext("2d",{alpha:false,desynchronized:true});
const current=document.querySelector("#current");
const slider=document.querySelector("#frame"), play=document.querySelector("#play");
const speed=document.querySelector("#speed"), status=document.querySelector("#status");
const update=document.querySelector("#update"), updates=document.querySelector("#updates");
const matchup=document.querySelector("#matchup");
const territoryButton=document.querySelector("#territory"), territoryStatus=document.querySelector("#territoryStatus");
const hexRadius=cfg.territory_radius||0, hexArena=cfg.arena_shape==="hex";
const territoryWidth=cfg.territory_width||32, territoryHeight=cfg.territory_height||18;
const territoryCells=hexRadius?cfg.territory_cells:territoryWidth*territoryHeight;
const strongpointCells=Array.isArray(cfg.strongpoint_cells)?cfg.strongpoint_cells:[];
const strongpointSet=new Set(strongpointCells), strongpointWeight=cfg.strongpoint_weight||1;
const territoryTotalWeight=territoryCells+strongpointCells.length*(strongpointWeight-1);
const territoryCellWeight=cell=>strongpointSet.has(cell)?strongpointWeight:1;
const territoryHexes=[];
if (hexRadius) for (let q=-hexRadius;q<=hexRadius;q++) {
  const first=Math.max(-hexRadius,-q-hexRadius), last=Math.min(hexRadius,-q+hexRadius);
  for (let r=first;r<=last;r++) territoryHexes.push([q,r]);
}
const territoryDensity=new Float32Array(2*territoryCells);
const territoryCanvas=document.createElement("canvas"), territoryContext=territoryCanvas.getContext("2d",{desynchronized:true});
const territoryFrame=new Uint16Array(frameCount);
if (controlShares) for (let frame=1;frame<frameCount;frame++) {
  let changed=false, base=frame*territoryCells*2, previousBase=base-territoryCells*2;
  for (let i=0;i<territoryCells*2&&!changed;i++) changed=controlShares[base+i]!==controlShares[previousBase+i];
  territoryFrame[frame]=changed?frame:territoryFrame[frame-1];
}
else if (territoryOwners) for (let frame=1;frame<frameCount;frame++) {
  let changed=false, base=frame*territoryCells, previousBase=base-territoryCells;
  for (let cell=0;cell<territoryCells&&!changed;cell++) changed=territoryOwners[base+cell]!==territoryOwners[previousBase+cell];
  territoryFrame[frame]=changed?frame:territoryFrame[frame-1];
}
const screenState=new Float32Array(soldierCount*4);
let cursor=0, playing=lastFrame>0, showTerritory=true, previous=performance.now();
let dirty=true, lastDraw=0, animationRequest=0, territoryKey="", territoryText="";
let territoryGeometryKey="", territoryBoard=null, territoryPaths=[];
slider.max=lastFrame;

function setText(element,value) { if (element.textContent!==value) element.textContent=value; }

function setMatchup() {
  if (!Number.isInteger(replay.learner_team)||!replay.opponent_mode) { matchup.hidden=true; return; }
  const opponent={nearest:"nearest-charge",pool:"policy-pool",self:"current-policy"}[replay.opponent_mode]||replay.opponent_mode;
  const roles=[opponent,opponent]; roles[replay.learner_team]="learner";
  const text=mobile?"B "+roles[0]+" · R "+roles[1]:"blue "+roles[0]+" vs red "+roles[1];
  setText(matchup,text); document.title="Muster · "+text;
}
setMatchup();

function setUpdateChoices(values) {
  if (!Number.isInteger(replay.update)) { update.hidden=true; return; }
  values=[...new Set(values.filter(Number.isInteger).concat(replay.update))].sort((a,b)=>b-a);
  updates.replaceChildren(...values.map(value=>{
    const option=document.createElement("option"); option.value=value; option.textContent=value;
    option.selected=value===replay.update; return option;
  }));
}
setUpdateChoices([]);
function refreshUpdates() {
  fetch("/replays/manifest.json",{cache:"no-store"})
    .then(response=>response.ok?response.json():Promise.reject())
    .then(manifest=>{
      setUpdateChoices(Array.isArray(manifest.updates)?manifest.updates:[]);
      if (Number.isInteger(manifest.current)&&manifest.current!==replay.update)
        setText(current,(mobile?"current ":"load current ")+manifest.current);
    })
    .catch(()=>{});
}
refreshUpdates();
setInterval(refreshUpdates,10000);

function arenaPath(context,left,top,width,height) {
  context.beginPath(); context.moveTo(left+width*.5,top);
  context.lineTo(left+width,top+height*.25); context.lineTo(left+width,top+height*.75);
  context.lineTo(left+width*.5,top+height); context.lineTo(left,top+height*.75);
  context.lineTo(left,top+height*.25); context.closePath();
}

function appendTerritoryHex(context,hex,left,top,pixelWidth,pixelHeight) {
  const q=hex[0], r=hex[1];
  const size=cfg.world_width/(3*hexRadius+2);
  const x=left+(cfg.world_width*.5+1.5*size*q)*pixelWidth/cfg.world_width;
  const y=top+pixelHeight-(cfg.world_height*.5+Math.sqrt(3)*size*(r+.5*q))*pixelHeight/cfg.world_height;
  const rx=size*pixelWidth/cfg.world_width, ry=size*pixelHeight/cfg.world_height;
  for (let side=0;side<6;side++) {
    const angle=side*Math.PI/3, px=x+rx*Math.cos(angle), py=y-ry*Math.sin(angle);
    if (side===0) context.moveTo(px,py); else context.lineTo(px,py);
  }
  context.closePath();
}

function territoryHexPath(context,hex,pixelWidth,pixelHeight) {
  context.beginPath(); appendTerritoryHex(context,hex,0,0,pixelWidth,pixelHeight);
}

function territoryBoardPath(context,left,top,pixelWidth,pixelHeight) {
  context.beginPath();
  for (const hex of territoryHexes) appendTerritoryHex(context,hex,left,top,pixelWidth,pixelHeight);
}

function prepareTerritoryGeometry(pixelWidth,pixelHeight) {
  const key=pixelWidth+":"+pixelHeight;
  if (key===territoryGeometryKey) return;
  territoryGeometryKey=key; territoryBoard=null; territoryPaths=[];
  if (!hexRadius||typeof Path2D==="undefined") return;
  territoryBoard=new Path2D();
  for (const hex of territoryHexes) {
    appendTerritoryHex(territoryBoard,hex,0,0,pixelWidth,pixelHeight);
    const path=new Path2D(); appendTerritoryHex(path,hex,0,0,pixelWidth,pixelHeight);
    territoryPaths.push(path);
  }
}

function updateTerritory(frameIndex,pixelWidth,pixelHeight) {
  frameIndex=(controlShares||territoryOwners)?territoryFrame[frameIndex]:frameIndex;
  const key=frameIndex+":"+pixelWidth+":"+pixelHeight;
  if (key===territoryKey) return;
  territoryKey=key;
  territoryCanvas.width=pixelWidth; territoryCanvas.height=pixelHeight;
  prepareTerritoryGeometry(pixelWidth,pixelHeight);
  territoryContext.clearRect(0,0,pixelWidth,pixelHeight);
  territoryContext.save();
  if (hexRadius) {
    if (territoryBoard) territoryContext.clip(territoryBoard);
    else { territoryBoardPath(territoryContext,0,0,pixelWidth,pixelHeight); territoryContext.clip(); }
  }
  else if (hexArena) { arenaPath(territoryContext,0,0,pixelWidth,pixelHeight); territoryContext.clip(); }
  const cellWidth=pixelWidth/territoryWidth, cellHeight=pixelHeight/territoryHeight;
  if (controlShares) {
    let territory0=0, territory1=0;
    const base=frameIndex*territoryCells*2;
    for (let cell=0;cell<territoryCells;cell++) {
      const s0=controlShares[base+cell*2]/255, s1=controlShares[base+cell*2+1]/255;
      territory0+=territoryCellWeight(cell)*s0; territory1+=territoryCellWeight(cell)*s1;
      const claimed=s0+s1;
      if (claimed<0.02) continue;
      const r=Math.round((53*s0+255*s1)/claimed), g=Math.round((167*s0+77*s1)/claimed), b=Math.round((255*s0+95*s1)/claimed);
      const alpha=Math.min(0.42,0.05+0.4*Math.max(s0,s1)).toFixed(3);
      territoryContext.fillStyle="rgba("+r+","+g+","+b+","+alpha+")";
      if (hexRadius) {
        if (territoryPaths.length) territoryContext.fill(territoryPaths[cell]);
        else { territoryHexPath(territoryContext,territoryHexes[cell],pixelWidth,pixelHeight); territoryContext.fill(); }
      }
      else {
        const x=cell%territoryWidth, y=Math.floor(cell/territoryWidth);
        territoryContext.fillRect(x*cellWidth,pixelHeight-(y+1)*cellHeight,cellWidth+.5,cellHeight+.5);
      }
    }
    const advantage=(territory0-territory1)/territoryTotalWeight;
    territoryText="blue "+(territory0/territoryTotalWeight).toFixed(3)+" | red "+(territory1/territoryTotalWeight).toFixed(3)+" | delta "+(advantage>=0?"+":"")+advantage.toFixed(3);
    territoryContext.restore();
    return;
  }
  if (territoryOwners) {
    let territory0=0, territory1=0;
    const base=frameIndex*territoryCells;
    for (let cell=0;cell<territoryCells;cell++) {
      const owner=territoryOwners[base+cell];
      if (owner<0) continue;
      if (owner===0) territory0+=territoryCellWeight(cell); else territory1+=territoryCellWeight(cell);
      if (!hexRadius) {
        territoryContext.fillStyle=owner===0?"rgba(53,167,255,.28)":"rgba(255,77,95,.28)";
        const x=cell%territoryWidth, y=Math.floor(cell/territoryWidth);
        territoryContext.fillRect(x*cellWidth,pixelHeight-(y+1)*cellHeight,cellWidth+.5,cellHeight+.5);
      }
    }
    if (hexRadius) for (let cell=0;cell<territoryCells;cell++) {
      const owner=territoryOwners[base+cell]; if (owner<0) continue;
      territoryContext.fillStyle=owner===0?"rgba(53,167,255,.28)":"rgba(255,77,95,.28)";
      if (territoryPaths.length) territoryContext.fill(territoryPaths[cell]);
      else { territoryHexPath(territoryContext,territoryHexes[cell],pixelWidth,pixelHeight); territoryContext.fill(); }
    }
    const advantage=(territory0-territory1)/territoryTotalWeight;
    territoryText="blue "+(territory0/territoryTotalWeight).toFixed(3)+" | red "+(territory1/territoryTotalWeight).toFixed(3)+" | delta "+(advantage>=0?"+":"")+advantage.toFixed(3);
    territoryContext.restore();
    return;
  }
  territoryDensity.fill(0);
  const frameBase=frameIndex*soldierCount;
  for (let i=0;i<soldierCount;i++) {
    const index=frameBase+i, strength=healths[index]/quantizedMax;
    if (strength<=0) continue;
    const positionX=positions[index*2]*cfg.world_width/quantizedMax;
    const positionY=positions[index*2+1]*cfg.world_height/quantizedMax;
    const gridX=Math.max(0,Math.min(territoryWidth-1,positionX*territoryWidth/cfg.world_width-.5));
    const gridY=Math.max(0,Math.min(territoryHeight-1,positionY*territoryHeight/cfg.world_height-.5));
    const x0=Math.floor(gridX), y0=Math.floor(gridY);
    const x1=Math.min(x0+1,territoryWidth-1), y1=Math.min(y0+1,territoryHeight-1);
    const fractionX=gridX-x0, fractionY=gridY-y0;
    const base=teams[i]*territoryCells;
    territoryDensity[base+y0*territoryWidth+x0]+=strength*(1-fractionX)*(1-fractionY);
    territoryDensity[base+y0*territoryWidth+x1]+=strength*fractionX*(1-fractionY);
    territoryDensity[base+y1*territoryWidth+x0]+=strength*(1-fractionX)*fractionY;
    territoryDensity[base+y1*territoryWidth+x1]+=strength*fractionX*fractionY;
  }

  let territory0=0, territory1=0;
  for (let y=0;y<territoryHeight;y++) for (let x=0;x<territoryWidth;x++) {
    let influence0=0, influence1=0;
    for (let offsetY=-1;offsetY<=1;offsetY++) {
      const otherY=y+offsetY; if (otherY<0 || otherY>=territoryHeight) continue;
      for (let offsetX=-1;offsetX<=1;offsetX++) {
        const otherX=x+offsetX; if (otherX<0 || otherX>=territoryWidth) continue;
        const weight=offsetX===0&&offsetY===0?4:offsetX===0||offsetY===0?2:1;
        const cell=otherY*territoryWidth+otherX;
        influence0+=weight*territoryDensity[cell];
        influence1+=weight*territoryDensity[territoryCells+cell];
      }
    }
    influence0*=.0625; influence1*=.0625;
    const presence=1-Math.exp(-(influence0+influence1)/.15);
    const difference=Math.max(-20,Math.min(20,(influence0-influence1)/.1));
    const share0=1/(1+Math.exp(-difference));
    const strategic=Math.sin(Math.PI*(x+.5)/territoryWidth)**2;
    const contribution=strategic*presence/(territoryCells*.5);
    territory0+=contribution*share0;
    territory1+=contribution*(1-share0);
    const alpha=.55*presence*strategic;
    if (alpha>.002) {
      const red=Math.round(53*share0+255*(1-share0));
      const green=Math.round(167*share0+77*(1-share0));
      const blue=Math.round(255*share0+95*(1-share0));
      territoryContext.fillStyle="rgba("+red+","+green+","+blue+","+alpha+")";
      territoryContext.fillRect(x*cellWidth,pixelHeight-(y+1)*cellHeight,cellWidth+.5,cellHeight+.5);
    }
  }
  const advantage=territory0-territory1;
  territoryText="blue "+territory0.toFixed(3)+" | red "+territory1.toFixed(3)+" | delta "+(advantage>=0?"+":"")+advantage.toFixed(3);
  territoryContext.restore();
}

function draw() {
  const dpr=Math.min(devicePixelRatio||1,maxDpr);
  const width=Math.max(1,Math.round(canvas.clientWidth*dpr));
  const height=Math.max(1,Math.round(canvas.clientHeight*dpr));
  if (canvas.width!==width || canvas.height!==height) {
    canvas.width=width; canvas.height=height; ctx.imageSmoothingEnabled=false;
  }
  const margin=12*dpr, scale=Math.min((width-2*margin)/cfg.world_width,(height-2*margin)/cfg.world_height);
  const left=(width-cfg.world_width*scale)/2, bottom=(height+cfg.world_height*scale)/2, top=bottom-cfg.world_height*scale;
  const worldWidth=Math.max(1,Math.round(cfg.world_width*scale));
  const worldHeight=Math.max(1,Math.round(cfg.world_height*scale));
  if (hexRadius) prepareTerritoryGeometry(worldWidth,worldHeight);
  const lower=Math.min(Math.floor(cursor),lastFrame), upper=Math.min(lower+1,lastFrame), amount=cursor-lower;
  ctx.fillStyle="#141414"; ctx.fillRect(0,0,width,height);
  ctx.save();
  if (hexRadius&&territoryBoard) {
    ctx.translate(left,top); ctx.clip(territoryBoard); ctx.translate(-left,-top);
  }
  else if (hexRadius) territoryBoardPath(ctx,left,top,cfg.world_width*scale,cfg.world_height*scale);
  else if (hexArena) arenaPath(ctx,left,top,cfg.world_width*scale,cfg.world_height*scale);
  else { ctx.beginPath(); ctx.rect(left,top,cfg.world_width*scale,cfg.world_height*scale); }
  if (!(hexRadius&&territoryBoard)) ctx.clip();
  ctx.fillStyle="#2b2b2b"; ctx.fillRect(left,top,cfg.world_width*scale,cfg.world_height*scale);
  if (showTerritory) {
    updateTerritory(lower,worldWidth,worldHeight);
    ctx.drawImage(territoryCanvas,left,top,cfg.world_width*scale,cfg.world_height*scale);
    setText(territoryStatus,territoryText);
  } else setText(territoryStatus,"");
  if (cfg.river_width>0) {
    const x=left+(cfg.world_width-cfg.river_width)*.5*scale, y0=(cfg.world_height-cfg.bridge_width)*.5;
    ctx.fillStyle="#1e465f"; ctx.fillRect(x,bottom-y0*scale,cfg.river_width*scale,y0*scale);
    ctx.fillRect(x,top,cfg.river_width*scale,y0*scale);
  }
  ctx.restore();
  if (hexRadius&&strongpointCells.length) {
    ctx.save(); ctx.translate(left,top);
    ctx.fillStyle="rgba(255,196,64,.10)"; ctx.strokeStyle="#ffc440"; ctx.lineWidth=Math.max(1.5*dpr,2);
    for (const cell of strongpointCells) {
      if (territoryPaths.length) { ctx.fill(territoryPaths[cell]); ctx.stroke(territoryPaths[cell]); }
      else { territoryHexPath(ctx,territoryHexes[cell],worldWidth,worldHeight); ctx.fill(); ctx.stroke(); }
    }
    ctx.restore();
  }
  ctx.strokeStyle="#969696"; ctx.lineWidth=Math.max(1,dpr);
  if (!hexRadius) {
    if (hexArena) arenaPath(ctx,left,top,cfg.world_width*scale,cfg.world_height*scale);
    else { ctx.beginPath(); ctx.rect(left,top,cfg.world_width*scale,cfg.world_height*scale); }
    ctx.stroke();
  }
  const radius=Math.max(2*dpr,cfg.soldier_radius*scale);
  const lowerBase=lower*soldierCount, upperBase=upper*soldierCount;
  for (let i=0;i<soldierCount;i++) {
    const lowerIndex=lowerBase+i, upperIndex=upperBase+i, output=i*4;
    const health=(healths[lowerIndex]+(healths[upperIndex]-healths[lowerIndex])*amount)/quantizedMax;
    screenState[output+3]=health;
    if (health<=0) continue;
    const p0=lowerIndex*2, p1=upperIndex*2;
    screenState[output]=left+(positions[p0]+(positions[p1]-positions[p0])*amount)*cfg.world_width*scale/quantizedMax;
    screenState[output+1]=bottom-(positions[p0+1]+(positions[p1+1]-positions[p0+1])*amount)*cfg.world_height*scale/quantizedMax;
    const angle0=angles[lowerIndex]*angleScale, angle1=angles[upperIndex]*angleScale;
    screenState[output+2]=angle0+Math.atan2(Math.sin(angle1-angle0),Math.cos(angle1-angle0))*amount;
  }
  ctx.strokeStyle="#071116"; ctx.lineWidth=Math.max(1,radius*.14);
  for (let team=0;team<2;team++) {
    ctx.fillStyle=team===0?"#35a7ff":"#ff4d5f"; ctx.beginPath();
    for (let i=0;i<soldierCount;i++) if (teams[i]===team && screenState[i*4+3]>0) {
      const x=screenState[i*4], y=screenState[i*4+1];
      ctx.moveTo(x+radius,y); ctx.arc(x,y,radius,0,2*Math.PI);
    }
    ctx.fill(); ctx.stroke();
  }
  const attackRadius=radius*1.17;
  ctx.strokeStyle="#ffe08a"; ctx.lineWidth=Math.max(1.2*dpr,radius*.25); ctx.beginPath();
  for (let i=0;i<soldierCount;i++) if (screenState[i*4+3]>0) {
    const x=screenState[i*4], y=screenState[i*4+1], start=-screenState[i*4+2]-Math.PI/3;
    ctx.moveTo(x+Math.cos(start)*attackRadius,y+Math.sin(start)*attackRadius);
    ctx.arc(x,y,attackRadius,start,start+2*Math.PI/3);
  }
  ctx.stroke();
  const healthRadius=radius*.72;
  ctx.strokeStyle="#d9f2dc"; ctx.lineWidth=Math.max(1*dpr,radius*.15); ctx.beginPath();
  for (let i=0;i<soldierCount;i++) {
    const health=screenState[i*4+3]; if (health<=0 || health>=.99999) continue;
    const x=screenState[i*4], y=screenState[i*4+1];
    ctx.moveTo(x+healthRadius,y); ctx.arc(x,y,healthRadius,0,2*Math.PI*health);
  }
  ctx.stroke();
  slider.value=Math.min(cursor,lastFrame);
  const ending=lower===lastFrame&&Number.isInteger(replay.winner)?(replay.winner<0?" | draw":" | "+(replay.winner===0?"blue":"red")+" wins"):"";
  setText(status,"step "+lower+" / "+lastFrame+ending);
}

function animate(now) {
  animationRequest=0;
  if (document.hidden) return;
  const elapsed=Math.min(now-previous,100); previous=now;
  if (playing) {
    cursor+=elapsed*Number(speed.value)/1000/cfg.decision_dt;
    if (cursor>=lastFrame) { cursor=lastFrame; setPlaying(false); }
    dirty=true;
  }
  if (dirty && (!playing || now-lastDraw>=targetFrameMs)) { draw(); dirty=false; lastDraw=now; }
  if (playing||dirty) scheduleDraw();
}
function scheduleDraw() {
  if (!animationRequest&&!document.hidden) animationRequest=requestAnimationFrame(animate);
}
function invalidate() { dirty=true; scheduleDraw(); }
function setPlaying(value) { playing=value && lastFrame>0; setText(play,playing?"pause":"play"); if (playing) scheduleDraw(); }
function toggle() { if (!playing && cursor>=lastFrame) cursor=0; setPlaying(!playing); previous=performance.now(); invalidate(); }
function toggleTerritory() {
  showTerritory=!showTerritory;
  setText(territoryButton,mobile?(showTerritory?"map on":"map off"):(showTerritory?"territory on (T)":"territory off (T)"));
  invalidate();
}
play.onclick=toggle;
current.onclick=()=>location.replace("/?current="+Date.now());
updates.onchange=()=>location.replace("/replays/update-"+updates.value+".html");
territoryButton.onclick=toggleTerritory;
slider.oninput=()=>{ cursor=Number(slider.value); setPlaying(false); invalidate(); };
document.onkeydown=e=>{
  if(e.key===" "){e.preventDefault();toggle();}
  else if(e.key.toLowerCase()==="t")toggleTerritory();
  else if(e.key==="ArrowLeft"||e.key==="ArrowRight") {
    e.preventDefault(); setPlaying(false);
    cursor=Math.max(0,Math.min(lastFrame,Math.floor(cursor)+(e.key==="ArrowLeft"?-1:1))); invalidate();
  }
};
document.addEventListener("visibilitychange",()=>{
  previous=performance.now();
  if (document.hidden&&animationRequest) { cancelAnimationFrame(animationRequest); animationRequest=0; }
  else invalidate();
});
const resized=()=>{ territoryKey=""; territoryGeometryKey=""; invalidate(); };
if (window.ResizeObserver) new ResizeObserver(resized).observe(canvas);
else window.addEventListener("resize",resized);
if (mobile) { setText(current,"current"); setText(territoryButton,"map on"); }
setPlaying(playing);
invalidate();
</script>
"""


def nearest_enemy_policy(state: Snapshot, config: Config) -> np.ndarray:
    """Move toward and aim at each living soldier's nearest living enemy.

    With no living enemy, occupy the nearest strongpoint instead of halting.
    """
    position = np.asarray(state["position"])
    team = np.asarray(state["team"])
    alive = np.asarray(state["alive"], dtype=bool)
    strongpoints = strongpoint_world_centers(config)
    actions = np.zeros((len(position), 4), np.float32)
    for soldier in np.flatnonzero(alive):
        enemies = np.flatnonzero(alive & (team != team[soldier]))
        if len(enemies) > 0:
            delta = position[enemies] - position[soldier]
            direction = delta[int(np.argmin(np.sum(delta * delta, axis=1)))]
            minimum_length = 1e-6
        else:
            delta = strongpoints - position[soldier]
            direction = delta[int(np.argmin(np.sum(delta * delta, axis=1)))]
            minimum_length = 1.0
        length = float(np.linalg.norm(direction))
        if length > minimum_length:
            direction = direction / length
            actions[soldier, :2] = direction
            actions[soldier, 2:] = direction
    return actions


def local_march_policy(
    state: Snapshot, config: Config, vision_radius: int = 2
) -> np.ndarray:
    """March forward, engaging only enemies visible within nearby hex tiles."""
    position = np.asarray(state["position"])
    team = np.asarray(state["team"])
    alive = np.asarray(state["alive"], dtype=bool)
    axial = TERRITORY_COORDINATES[territory_indices(position, config)]
    actions = np.zeros((len(position), 4), np.float32)
    actions[alive, 0] = np.where(team[alive] == 0, 1.0, -1.0)
    actions[alive, 2] = actions[alive, 0]
    for soldier in np.flatnonzero(alive):
        enemies = np.flatnonzero(alive & (team != team[soldier]))
        delta_hex = axial[enemies] - axial[soldier]
        distance = np.max(
            np.stack(
                (np.abs(delta_hex[:, 0]), np.abs(delta_hex[:, 1]), np.abs(delta_hex.sum(1)))
            ),
            axis=0,
        )
        visible = enemies[distance <= vision_radius]
        if len(visible) == 0:
            continue
        delta = position[visible] - position[soldier]
        direction = delta[np.argmin(np.sum(delta * delta, axis=1))]
        length = float(np.linalg.norm(direction))
        if length > 1e-6:
            direction = direction / length
            actions[soldier, :2] = direction
            actions[soldier, 2:] = direction
    return actions


def record_episode(simulator: object, policy: Policy = nearest_enemy_policy) -> dict[str, object]:
    """Run the current one-environment episode and return compact replay data."""
    if simulator.num_envs != 1:
        raise ValueError("episode recording expects exactly one environment")
    config = simulator.config
    state = simulator.snapshot(0)
    started = time.perf_counter()
    steps = 0
    replay: dict[str, object] = {
        "config": {
            "world_width": config.world_width,
            "world_height": config.world_height,
            "soldier_radius": config.soldier_radius,
            "initial_health": config.initial_health,
            "decision_dt": config.decision_dt,
            "river_width": config.river_width,
            "bridge_width": config.bridge_width,
            "arena_shape": "hex",
            "territory_radius": TERRITORY_RADIUS,
            "territory_cells": TERRITORY_CELLS,
            "strongpoint_cells": STRONGPOINT_CELLS.tolist(),
            "strongpoint_weight": STRONGPOINT_WEIGHT,
        },
        "team": np.asarray(state["team"], dtype=int).tolist(),
        "frames": [],
    }
    frames = replay["frames"]
    for _ in range(config.maximum_decision_steps + 1):
        frames.append(
            {
                "p": np.asarray(state["position"]).round(4).tolist(),
                "a": np.asarray(state["attack_angle"]).round(4).tolist(),
                "h": np.asarray(state["health"]).round(3).tolist(),
                "o": np.asarray(state["territory_owner"], dtype=np.int8).tolist(),
                "c": np.asarray(state["control_share"], dtype=np.float32),
            }
        )
        if state["done"]:
            elapsed = time.perf_counter() - started
            replay["winner"] = int(state["winner"])
            replay["statistics"] = {
                "decision_steps": steps,
                "simulated_seconds": steps * config.decision_dt,
                "recording_wall_seconds": elapsed,
                "environment_decisions_per_second": steps / max(elapsed, 1e-9),
                "soldier_decisions_per_second": steps * simulator.num_soldiers / max(elapsed, 1e-9),
            }
            return replay
        simulator.step(policy(state, config))
        steps += 1
        state = simulator.snapshot(0)
    raise RuntimeError("episode did not finish within its configured limit")


def _encode_array(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _pack_replay(replay: Mapping[str, object]) -> dict[str, object]:
    """Quantize frame data into compact browser-native typed arrays."""
    frames = list(replay["frames"])
    if not frames:
        raise ValueError("replay needs at least one frame")
    config = replay["config"]
    positions = np.asarray([frame["p"] for frame in frames], dtype=np.float32)
    angles = np.asarray([frame["a"] for frame in frames], dtype=np.float32)
    health = np.asarray([frame["h"] for frame in frames], dtype=np.float32)
    territory_owner = (
        np.asarray([frame["o"] for frame in frames], dtype=np.int8)
        if all("o" in frame for frame in frames)
        else None
    )
    control_share = (
        np.asarray([frame["c"] for frame in frames], dtype=np.float32)
        if all("c" in frame for frame in frames)
        else None
    )
    expected = (len(frames), len(replay["team"]))
    if positions.shape != (*expected, 2) or angles.shape != expected or health.shape != expected:
        raise ValueError("replay frame shapes do not match its team array")
    if territory_owner is not None:
        territory_shape = (
            (len(frames), int(config["territory_cells"]))
            if config.get("territory_radius")
            else (
                len(frames),
                int(config["territory_height"]),
                int(config["territory_width"]),
            )
        )
        if territory_owner.shape != territory_shape:
            raise ValueError(f"expected territory frames with shape {territory_shape}")

    world_size = np.array(
        [float(config["world_width"]), float(config["world_height"])], dtype=np.float32
    )
    position_u16 = np.rint(np.clip(positions / world_size, 0, 1) * 65535).astype("<u2")
    wrapped_angles = (angles + math.pi) % (2 * math.pi) - math.pi
    angle_i16 = np.rint(wrapped_angles * (32767 / math.pi)).astype("<i2")
    initial_health = max(float(config["initial_health"]), 1e-6)
    health_u16 = np.rint(np.clip(health / initial_health, 0, 1) * 65535).astype("<u2")

    packed = {key: value for key, value in replay.items() if key != "frames"}
    packed.update(
        frame_count=len(frames),
        position_u16=_encode_array(position_u16),
        angle_i16=_encode_array(angle_i16),
        health_u16=_encode_array(health_u16),
    )
    if territory_owner is not None:
        packed["territory_i8"] = _encode_array(territory_owner)
    if control_share is not None:
        quantized = np.rint(np.clip(control_share, 0, 1) * 255).astype(np.uint8)
        packed["control_u8"] = _encode_array(quantized)
    return packed


def write_replay(replay: Mapping[str, object], path: str | Path) -> None:
    """Write a compact, self-contained browser replay."""
    payload = json.dumps(_pack_replay(replay), separators=(",", ":"))
    Path(path).write_text(_REPLAY_HTML.replace("__REPLAY__", payload), encoding="utf-8")


class Viewer:
    """A read-only renderer. ``draw`` returns false after quit or Escape."""

    def __init__(self, config: Config, width: int = 1000, title: str = "Simulator"):
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("install the viewer extra: pip install -e '.[viewer]'") from error
        self.pg = pygame
        self.config = config
        world_ratio = config.world_height / config.world_width
        self.size = width, max(300, int(width * world_ratio) + 50)
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        try:
            self.font = pygame.font.Font(None, 22)
        except (ImportError, NotImplementedError):
            self.font = None
        self.clock = pygame.time.Clock()
        self.closed = False

    def _point(self, position: np.ndarray, scale: float, left: float, bottom: float) -> tuple[int, int]:
        return round(left + position[0] * scale), round(bottom - position[1] * scale)

    def draw(self, state: Snapshot, caption: str = "") -> bool:
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.closed = True
        if self.closed:
            return False

        c = self.config
        margin, header = 16, 30
        scale = min((self.size[0] - 2 * margin) / c.world_width, (self.size[1] - header - margin) / c.world_height)
        left = (self.size[0] - c.world_width * scale) / 2
        top = header
        bottom = top + c.world_height * scale
        self.screen.fill((20, 20, 20))
        angles = np.arange(6) * math.pi / 3
        offsets = c.territory_tile_size * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        arena = [
            [self._point(point, scale, left, bottom) for point in center + offsets]
            for center in territory_centers(c)
        ]
        strongpoints = set(STRONGPOINT_CELLS.tolist())
        for cell, tile in enumerate(arena):
            color = (72, 59, 27) if cell in strongpoints else (43, 43, 43)
            pg.draw.polygon(self.screen, color, tile)

        if c.river_width > 0:
            river_x = left + (c.world_width - c.river_width) * 0.5 * scale
            river_w = c.river_width * scale
            bridge_lo = c.world_height * 0.5 - c.bridge_width * 0.5
            bridge_hi = c.world_height * 0.5 + c.bridge_width * 0.5
            pg.draw.rect(self.screen, (30, 70, 95), (river_x, bottom - bridge_lo * scale, river_w, bridge_lo * scale))
            pg.draw.rect(self.screen, (30, 70, 95), (river_x, top, river_w, (c.world_height - bridge_hi) * scale))

        position = np.asarray(state["position"])
        team = np.asarray(state["team"])
        angle = np.asarray(state["attack_angle"])
        health = np.asarray(state["health"])
        alive = np.asarray(state["alive"], dtype=bool)
        radius = max(2, round(c.soldier_radius * scale))
        colors = ((53, 167, 255), (255, 77, 95))
        for soldier in np.flatnonzero(alive):
            center = self._point(position[soldier], scale, left, bottom)
            color = colors[int(team[soldier])] if int(team[soldier]) in (0, 1) else (190, 190, 190)
            pg.draw.circle(self.screen, color, center, radius)
            pg.draw.circle(self.screen, (7, 17, 22), center, radius, max(1, round(radius * 0.14)))
            arc = float(angle[soldier]) + np.linspace(-math.pi / 3, math.pi / 3, 17)
            arc_position = position[soldier] + np.stack((np.cos(arc), np.sin(arc)), axis=-1) * c.soldier_radius * 1.17
            arc_points = [self._point(point, scale, left, bottom) for point in arc_position]
            pg.draw.lines(self.screen, (255, 224, 138), False, arc_points, max(1, round(radius * 0.25)))
            fraction = max(0.0, min(1.0, float(health[soldier]) / c.initial_health))
            if fraction < 1:
                health_arc = np.linspace(0, 2 * math.pi * fraction, max(2, math.ceil(25 * fraction)))
                health_position = position[soldier] + np.stack((np.cos(health_arc), np.sin(health_arc)), axis=-1) * c.soldier_radius * 0.72
                health_points = [self._point(point, scale, left, bottom) for point in health_position]
                pg.draw.lines(self.screen, (217, 242, 220), False, health_points, max(1, round(radius * 0.15)))

        status = f"step {state.get('step_count', '?')}"
        if state.get("done", False):
            winner = state.get("winner", -1)
            status += "  draw" if winner == -1 else f"  team {winner} wins"
        if caption:
            status += f"  {caption}"
        if self.font is not None:
            self.screen.blit(self.font.render(status, True, (225, 225, 225)), (margin, 7))
        for cell, tile in enumerate(arena):
            color = (255, 196, 64) if cell in strongpoints else (60, 60, 60)
            pg.draw.polygon(self.screen, color, tile, 2 if cell in strongpoints else 1)
        pg.display.flip()
        return True

    def close(self) -> None:
        if not self.closed:
            self.pg.quit()
            self.closed = True


def run(simulator: object, policy: Policy = nearest_enemy_policy, fps: int = 20) -> None:
    """Run a one-environment simulator with any ``policy(snapshot, config)`` callback."""
    if simulator.num_envs != 1:
        raise ValueError("standalone viewer run expects exactly one environment")
    viewer = Viewer(simulator.config)
    state = simulator.snapshot(0)
    try:
        while viewer.draw(state):
            if state["done"]:
                simulator.reset()
            else:
                simulator.step(policy(state, simulator.config))
            state = simulator.snapshot(0)
            viewer.clock.tick(fps)
    finally:
        viewer.close()


def load_checkpoint_policy(
    simulator: object, checkpoint: Mapping[str, object]
) -> Policy:
    """Build a deterministic snapshot callback from a training checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install the RL extra: pip install -e '.[rl]'") from error
    from policy import CHECKPOINT_VERSION, Policy as NeuralPolicy
    from rl_env import RLEnv

    env = RLEnv(simulator=simulator)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint predates the local hex policy and cannot be viewed")
    model = NeuralPolicy(**checkpoint.get("model_kwargs", {})).to(env.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    opponent = None
    pool = checkpoint.get("opponent_pool")
    if pool is not None:
        opponent = NeuralPolicy(**checkpoint.get("model_kwargs", {})).to(env.device)
        opponent.load_state_dict(pool["models"][int(pool["latest_slot"])])
        opponent.eval()

    def act(_state: Snapshot, _config: Config):
        state = env.observe()
        with torch.no_grad():
            actions, _, _ = model.act(state, deterministic=True)
            if opponent is not None:
                actions[:, 1] = opponent.actor_actions(state, deterministic=True)[:, 1]
        world_actions = env.actions_to_sim(actions)
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        return world_actions

    return act


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--soldiers", type=int, default=256, help="soldiers per team")
    parser.add_argument("--record", metavar="HTML", help="record one episode as a browser replay")
    parser.add_argument("--checkpoint", metavar="PT", help="run a deterministic neural policy")
    args = parser.parse_args()
    checkpoint = None
    if args.checkpoint:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("install the RL extra: pip install -e '.[rl]'") from error
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = Config(**checkpoint["sim_config"])
    else:
        config = Config(soldiers_per_team=args.soldiers)
    if args.device == "cpu" and checkpoint is None:
        simulator = CpuSimulator(config)
    else:
        from simulator_gpu import GpuSimulator

        simulator = GpuSimulator(config, device=args.device)
    selected_policy = (
        load_checkpoint_policy(simulator, checkpoint) if checkpoint is not None else nearest_enemy_policy
    )
    if args.record:
        replay = record_episode(simulator, selected_policy)
        write_replay(replay, args.record)
        print(args.record)
        statistics = replay["statistics"]
        print(
            f"{statistics['environment_decisions_per_second']:,.1f} recorded environment decisions/s; "
            f"{statistics['soldier_decisions_per_second']:,.0f} soldier decisions/s"
        )
    else:
        run(simulator, selected_policy, fps=args.fps)


if __name__ == "__main__":
    main()
