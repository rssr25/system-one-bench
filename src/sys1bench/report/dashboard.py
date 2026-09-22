"""Self-contained HTML dashboard: one file, data inlined, no network. Views: overview (sortable, hosted/local apart),
per-question reliability + risk-coverage overlays, framing heatmap, sweep curves, audit chips."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics.selective import risk_coverage
from .results_doc import _load_dir, _main_rows, _model_meta
from .scorecard import framing_scorecard, group_rows, scorecard


def _reliability_bins(rows, bins=10):
    ok = [r for r in rows if r.error is None and r.correct is not None]
    if len(ok) < 20:
        return []
    conf = np.array([max(r.probs) for r in ok])
    corr = np.array([float(r.correct) for r in ok])
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            out.append({"x": float(conf[m].mean()), "y": float(corr[m].mean()), "n": int(m.sum())})
    return out


def _risk_curve(rows, points=60):
    ok = [r for r in rows if r.error is None and r.correct is not None]
    if len(ok) < 20:
        return []
    cov, risk = risk_coverage(np.array([max(r.probs) for r in ok]), np.array([float(r.correct) for r in ok]))
    sel = np.linspace(0, len(cov) - 1, min(points, len(cov))).astype(int)
    return [{"x": float(cov[i]), "y": float(risk[i])} for i in sel]


def collect(dirs: list[Path]) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for d in dirs:
        data = _load_dir(d)
        if not data["preds"]:
            continue
        all_rows = [r for rs in data["preds"].values() for r in rs]
        meta = _model_meta(all_rows)
        entry: dict[str, Any] = {"dir": d.name, **meta, "questions": {}, "audits": [], "sweeps": {}}
        for mname in ("tickets", "phish"):
            rows = data["preds"].get(mname, [])
            for (qk,), rs in sorted(group_rows(_main_rows(rows), "question_key").items()):
                if qk.startswith("sub_"):
                    continue
                c = scorecard(rs, floor_resamples=50)
                canon = [r for r in rs if r.permutation_id == "p0" and r.framing_id == "f0"]
                fs = framing_scorecard(rs)
                entry["questions"][f"{mname}.{qk}"] = {
                    "primitive": rs[0].primitive, "K": rs[0].cardinality, "n": c.get("calibration", {}).get("n"),
                    "acc": c.get("accuracy"), "acc_min": c.get("accuracy_range", [None, None])[0], "acc_max": c.get("accuracy_range", [None, None])[1],
                    "ece": c.get("calibration", {}).get("ece_width_15"), "ece_floor": c.get("calibration", {}).get("ece_floor_mean"),
                    "ece_over_floor": c.get("calibration", {}).get("ece_over_floor"), "brier": c.get("calibration", {}).get("brier"),
                    "T": c.get("temperature", {}).get("T"), "T_dir": c.get("temperature", {}).get("direction"),
                    "aurc": c.get("selective", {}).get("aurc"), "cov5": c.get("selective", {}).get("coverage_at_risk_5"),
                    "perm_jsd": c.get("permutation", {}).get("mean_jsd"), "flips": c.get("permutation", {}).get("argmax_flip_rate"),
                    "p50": c.get("latency", {}).get("p50"), "p95": c.get("latency", {}).get("p95"), "renorm": c.get("renormalised_rate"),
                    "adv_min": c.get("framing", {}).get("accuracy_adversarial_min"),
                    "ordinal": c.get("ordinal"), "framings": fs["accuracy_by_framing"],
                    "reliability": _reliability_bins(canon), "risk": _risk_curve(canon),
                }
        # audits
        for mname in ("tickets", "phish"):
            rows = data["preds"].get(mname, [])
            full = [r for r in rows if r.arm == "main" and r.permutation_id == "p0" and r.framing_id == "f0" and r.primitive == "choice"]
            so = [r for r in rows if r.arm == "state_only"]
            oo = [r for r in rows if r.arm == "options_only"]
            for qk in sorted({r.question_key for r in full}):
                f_ = [r for r in full if r.question_key == qk]
                truths = [r.ground_truth for r in f_]
                prior = max(truths.count(t) for t in set(truths)) / len(truths) if truths else float("nan")
                a_so = np.mean([r.correct for r in so if r.question_key == qk and r.correct is not None]) if so else float("nan")
                a_oo = np.mean([r.correct for r in oo if r.question_key == qk and r.correct is not None]) if oo else float("nan")
                entry["audits"].append({"name": f"{mname}.{qk} state-only", "value": float(a_so), "flag": bool(a_so > prior + 0.10), "note": f"prior {prior:.2f}"})
                entry["audits"].append({"name": f"{mname}.{qk} options-only", "value": float(a_oo), "flag": bool(a_oo > prior + 0.10), "note": f"prior {prior:.2f}"})
        sd = d.parent / f"{d.name}_sweeps"
        if sd.exists():
            for f in sd.glob("*.json"):
                try:
                    entry["sweeps"][f.stem] = json.loads(f.read_text())
                except Exception:
                    pass
        models.append(entry)
    return {"models": models}


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--fg:#111827;--mut:#6b7280;--line:#e5e7eb;--acc:#4f46e5;--acc2:#06b6d4;--warn:#f59e0b;--bad:#dc2626;--ok:#16a34a;--card:#f9fafb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b0f19;--fg:#e5e7eb;--mut:#9ca3af;--line:#1f2937;--card:#111827}}
:root[data-theme=dark]{--bg:#0b0f19;--fg:#e5e7eb;--mut:#9ca3af;--line:#1f2937;--card:#111827}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 Inter,system-ui,sans-serif;padding:16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px}.mut{color:var(--mut)}
nav{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}nav button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:6px 12px;border-radius:999px;cursor:pointer}
nav button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{cursor:pointer;color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--bg)}
.wrap{overflow-x:auto}.bar{display:inline-block;height:8px;background:var(--acc);border-radius:4px;vertical-align:middle;opacity:.35}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--acc);vertical-align:middle;margin-left:-8px}
.chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin:2px;border:1px solid var(--line)}.chip.flag{background:var(--warn);color:#111;border-color:var(--warn)}.chip.ok{color:var(--ok)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
canvas{width:100%;height:240px}select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:4px 8px}
.heat td{text-align:center;color:#fff;font-size:12px}.legend span{display:inline-flex;align-items:center;gap:4px;margin-right:10px;font-size:12px}.sw{width:12px;height:3px;display:inline-block}
.sect h3{margin:0 0 6px;font-size:14px}.bad{color:var(--bad)}
</style></head><body>
<h1>__TITLE__</h1><div class="mut">sys1bench dashboard · hosted and local models are grouped separately and their latencies are not comparable · accuracy is the median over framings with the range as a bar</div>
<nav id="nav"></nav><main id="main"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);const M=D.models;
const COLORS=['#4f46e5','#06b6d4','#f59e0b','#dc2626','#16a34a','#a855f7','#0ea5e9','#84cc16'];
const f=(v,d=3)=>v==null||Number.isNaN(v)?'–':(typeof v==='number'?(Math.abs(v)>=100?v.toFixed(0):v.toFixed(d)):String(v));
const views={overview:Overview,questions:Questions,framing:Framing,sweeps:Sweeps,audits:Audits};
let view='overview',sortKey='acc',sortDir=-1;
function nav(){const n=document.getElementById('nav');n.innerHTML='';for(const k of Object.keys(views)){const b=document.createElement('button');b.textContent=k[0].toUpperCase()+k.slice(1);b.className=k===view?'on':'';b.onclick=()=>{view=k;render()};n.appendChild(b)}}
function render(){nav();document.getElementById('main').innerHTML='';views[view](document.getElementById('main'))}
function Overview(el){const cols=[['question','question'],['prim','primitive'],['K','K'],['n','n'],['acc (median)','acc'],['range','range'],['adv. min','adv_min'],['ECE/floor','ece_over_floor'],['Brier','brier'],['T','T'],['AURC','aurc'],['cov@5%','cov5'],['perm JSD','perm_jsd'],['p50 ms','p50'],['renorm','renorm']];
 for(const dep of ['hosted','local']){const ms=M.filter(m=>m.deployment===dep);if(!ms.length)continue;const h=document.createElement('h2');h.textContent=dep==='hosted'?'Hosted models (client-observed latency)':'Local models (compute latency on stated hardware)';el.appendChild(h);
 for(const m of ms){const c=document.createElement('div');c.className='card';c.innerHTML=`<b>${m.model_id}</b> <span class="mut">· ${m.adapter} · ${m.version_hash||''} · ${m.hardware||''}</span>`;
 let rows=Object.entries(m.questions).map(([q,v])=>({question:q,...v}));rows.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*sortDir});
 const t=document.createElement('table');t.innerHTML='<thead><tr>'+cols.map(([h,k])=>`<th data-k="${k}">${h}${sortKey===k?(sortDir<0?' ▼':' ▲'):''}</th>`).join('')+'</tr></thead>';
 const tb=document.createElement('tbody');for(const r of rows){const tr=document.createElement('tr');tr.innerHTML=cols.map(([h,k])=>{if(k==='range'){const lo=r.acc_min??r.acc,hi=r.acc_max??r.acc;return `<td title="${f(lo)}..${f(hi)}"><span class="bar" style="width:${Math.max(2,(hi-lo)*120)}px;margin-left:${lo*120}px"></span><span class="dot" style="margin-left:${(r.acc-hi)*120-8}px"></span></td>`}
 if(k==='ece_over_floor'){const v=r[k];return `<td class="${v>5?'bad':''}">${f(v,2)}</td>`}if(k==='T'){return `<td title="${r.T_dir||''}">${r.T_dir&&r.T_dir.startsWith('degenerate')?'–':f(r.T,2)}</td>`}return `<td>${f(r[k])}</td>`}).join('');tb.appendChild(tr)}
 t.appendChild(tb);t.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=-1}render()});const w=document.createElement('div');w.className='wrap';w.appendChild(t);c.appendChild(w);el.appendChild(c)}}}
function qlist(){return [...new Set(M.flatMap(m=>Object.keys(m.questions)))].sort()}
function Questions(el){const qs=qlist();const sel=document.createElement('select');qs.forEach(q=>{const o=document.createElement('option');o.value=q;o.textContent=q;sel.appendChild(o)});sel.value=window._q||qs[0];sel.onchange=()=>{window._q=sel.value;render()};el.appendChild(sel);
 const q=sel.value;const g=document.createElement('div');g.className='grid';el.appendChild(g);
 const leg=document.createElement('div');leg.className='legend';M.forEach((m,i)=>{if(m.questions[q])leg.innerHTML+=`<span><i class="sw" style="background:${COLORS[i%COLORS.length]}"></i>${m.model_id}</span>`});el.insertBefore(leg,g);
 const c1=card(g,'Reliability (canonical framing): accuracy in bin vs confidence');const c2=card(g,'Risk–coverage: error rate among answered vs coverage');const c3=card(g,'Ordinal (score questions)');
 draw(c1,M.map((m,i)=>({pts:(m.questions[q]||{}).reliability||[],c:COLORS[i%COLORS.length]})),true);draw(c2,M.map((m,i)=>({pts:(m.questions[q]||{}).risk||[],c:COLORS[i%COLORS.length]})),false);
 const o=M.filter(m=>m.questions[q]&&m.questions[q].ordinal);if(o.length){c3.innerHTML+='<table><tr><th>model</th><th>exact</th><th>off-by-one</th><th>MAE</th><th>QWK</th><th>RPS</th><th>Spearman</th></tr>'+o.map(m=>{const x=m.questions[q].ordinal;return `<tr><td>${m.model_id}</td><td>${f(x.exact_accuracy_argmax)}</td><td>${f(x.off_by_one_accuracy)}</td><td>${f(x.mae_expected)}</td><td>${f(x.qwk_argmax)}</td><td>${f(x.rps)}</td><td>${f(x.spearman_expected)}</td></tr>`}).join('')+'</table>'}else c3.innerHTML+='<span class="mut">not a score question</span>'}
function card(g,title){const c=document.createElement('div');c.className='card sect';c.innerHTML=`<h3>${title}</h3>`;const cv=document.createElement('canvas');c.appendChild(cv);g.appendChild(c);return c}
function draw(cardEl,series,diag){const cv=cardEl.querySelector('canvas');if(!cv)return;const r=cv.getBoundingClientRect();cv.width=r.width*2||1200;cv.height=480;const ctx=cv.getContext('2d');const W=cv.width,H=cv.height,p=50;ctx.scale(1,1);
 ctx.strokeStyle=getComputedStyle(document.body).getPropertyValue('--line');ctx.lineWidth=2;ctx.strokeRect(p,p/2,W-1.5*p,H-1.5*p);ctx.fillStyle=getComputedStyle(document.body).getPropertyValue('--mut');ctx.font='20px system-ui';
 for(const t of [0,0.5,1]){ctx.fillText(t.toFixed(1),p+(W-1.5*p)*t-12,H-p/2+26);ctx.fillText(t.toFixed(1),8,H-p+2-(H-1.5*p)*t+8)}
 if(diag){ctx.setLineDash([6,6]);ctx.beginPath();ctx.moveTo(p,H-p);ctx.lineTo(W-p/2,p/2);ctx.stroke();ctx.setLineDash([])}
 for(const s of series){if(!s.pts.length)continue;ctx.strokeStyle=s.c;ctx.fillStyle=s.c;ctx.lineWidth=3;ctx.beginPath();s.pts.forEach((pt,i)=>{const x=p+(W-1.5*p)*pt.x,y=H-p-(H-1.5*p)*pt.y;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();if(diag)for(const pt of s.pts){ctx.beginPath();ctx.arc(p+(W-1.5*p)*pt.x,H-p-(H-1.5*p)*pt.y,5,0,7);ctx.fill()}}}
function Framing(el){const qs=qlist();for(const q of qs){const fr=[...new Set(M.flatMap(m=>Object.keys((m.questions[q]||{}).framings||{})))];if(!fr.length)continue;const order=['f0',...fr.filter(x=>x.startsWith('para')).sort(),...fr.filter(x=>x.startsWith('adv')).sort(),...fr.filter(x=>x.startsWith('crit')).sort()].filter((x,i,a)=>fr.includes(x)&&a.indexOf(x)===i);
 const c=document.createElement('div');c.className='card';c.innerHTML=`<h3>${q}</h3><div class="mut">accuracy by framing; f0 canonical, para paraphrases, adv adversarial, crit criteria variants (swapped = corruption control)</div>`;const t=document.createElement('table');t.className='heat';
 t.innerHTML='<tr><th>model</th>'+order.map(o=>`<th>${o.replace('crit_','')}</th>`).join('')+'</tr>'+M.filter(m=>m.questions[q]).map(m=>'<tr><td style="text-align:left;color:var(--fg)">'+m.model_id+'</td>'+order.map(o=>{const v=m.questions[q].framings[o];if(v==null)return '<td></td>';const h=Math.round(v*120);return `<td style="background:hsl(${h} 60% 40%)">${v.toFixed(2)}</td>`}).join('')+'</tr>').join('');
 const w=document.createElement('div');w.className='wrap';w.appendChild(t);c.appendChild(w);el.appendChild(c)}}
function Sweeps(el){const g=document.createElement('div');g.className='grid';el.appendChild(g);
 const line=(title,ser,xlog)=>{const c=card(g,title);const cv=c.querySelector('canvas');const r=cv.getBoundingClientRect();cv.width=r.width*2||1200;cv.height=480;const ctx=cv.getContext('2d');const W=cv.width,H=cv.height,p=60;const xs=ser.flatMap(s=>s.pts.map(q=>q.x)),ys=ser.flatMap(s=>s.pts.map(q=>q.y)).filter(v=>Number.isFinite(v));if(!xs.length)return;const xm=Math.min(...xs),xM=Math.max(...xs),ym=Math.min(0,...ys),yM=Math.max(...ys)*1.05||1;const X=x=>p+(W-1.5*p)*((xlog?Math.log(x)-Math.log(xm):x-xm)/((xlog?Math.log(xM)-Math.log(xm):xM-xm)||1)),Y=y=>H-p-(H-1.5*p)*((y-ym)/((yM-ym)||1));
 ctx.strokeStyle=getComputedStyle(document.body).getPropertyValue('--line');ctx.lineWidth=2;ctx.strokeRect(p,p/2,W-1.5*p,H-1.5*p);ctx.fillStyle=getComputedStyle(document.body).getPropertyValue('--mut');ctx.font='20px system-ui';ctx.fillText(f(xm,0),p,H-8);ctx.fillText(f(xM,0),W-p-30,H-8);ctx.fillText(f(yM,2),4,p/2+16);ctx.fillText(f(ym,2),4,H-p);
 ser.forEach((s,i)=>{ctx.strokeStyle=COLORS[i%COLORS.length];ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=3;ctx.beginPath();s.pts.filter(q=>Number.isFinite(q.y)).forEach((q,j)=>{j?ctx.lineTo(X(q.x),Y(q.y)):ctx.moveTo(X(q.x),Y(q.y))});ctx.stroke();ctx.fillText(s.name,p+10,p/2+22+22*i)})};
 const pick=(sw,name,metric,qk)=>{const d=sw[name];if(!d)return[];return Object.entries(d).filter(([k])=>!isNaN(+k)).map(([k,v])=>({x:+k,y:qk?((v.by_question||{})[qk]||{})[metric]:v[metric]})).sort((a,b)=>a.x-b.x)};
 line('Cardinality: accuracy vs K (rag best_passage)',M.map(m=>({name:m.model_id,pts:pick(m.sweeps,'cardinality_rag','accuracy')})),true);
 line('Cardinality: ECE/floor vs K',M.map(m=>({name:m.model_id,pts:pick(m.sweeps,'cardinality_rag','ece_over_floor')})),true);
 line('State length: priority accuracy vs target tokens',M.map(m=>({name:m.model_id,pts:pick(m.sweeps,'length_tickets','accuracy','priority')})),true);
 line('State length: p50 latency (ms) vs target tokens',M.map(m=>({name:m.model_id,pts:pick(m.sweeps,'length_tickets','latency','queue').map(q=>({x:q.x,y:q.y?q.y.p50:NaN}))})),true);
 for(const tgt of ['queue','priority'])line(`Interference (${tgt}): p50 ms vs extra questions`,M.map(m=>{const d=m.sweeps['interference_'+tgt];if(!d)return{name:m.model_id,pts:[]};const kind=d.by_kind.irrelevant||d.by_kind.relevant||{};return{name:m.model_id,pts:Object.entries(kind).map(([q,v])=>({x:+q,y:v.latency_p50_ms})).sort((a,b)=>a.x-b.x)}}),false);
 line('Option budget (Laya): accuracy vs head_max_len at K=12',M.map(m=>({name:m.model_id,pts:pick(m.sweeps,'budget_K12','accuracy')})),false);}
function Audits(el){for(const m of M){const c=document.createElement('div');c.className='card';c.innerHTML=`<h3>${m.model_id}</h3>`;
 for(const a of m.audits)c.innerHTML+=`<span class="chip ${a.flag?'flag':'ok'}" title="${a.note}">${a.name}: ${f(a.value)}${a.flag?' ⚠':''}</span>`;
 for(const [q,v] of Object.entries(m.questions)){if(v.renorm>0)c.innerHTML+=`<span class="chip">${q} renormalised ${f(v.renorm)}</span>`;if(v.ece_over_floor>5)c.innerHTML+=`<span class="chip flag">${q} ECE ${f(v.ece_over_floor,1)}× floor</span>`;if(v.T&&v.T>1.5&&!(v.T_dir||'').startsWith('deg'))c.innerHTML+=`<span class="chip flag">${q} over-confident T=${f(v.T,2)}</span>`;if(v.adv_min!=null&&v.acc!=null&&v.acc-v.adv_min>0.05)c.innerHTML+=`<span class="chip flag">${q} adversarial wording −${f(v.acc-v.adv_min,2)}</span>`}
 const rb=(m.sweeps.robustness_tickets||{});if(rb.none_of_the_above){const w=rb.none_of_the_above.without_abstain_option||{};c.innerHTML+=`<span class="chip ${w['frac_ood_conf_over_0.7']>0.3?'flag':'ok'}">none-of-the-above: ${f(w['frac_ood_conf_over_0.7'])} of out-of-scope items above 0.7 conf (AUROC ${f(w.auroc_confidence,2)})</span>`}
 el.appendChild(c)}}
render();
</script></body></html>"""


def build_dashboard(dirs: list[Path], title: str = "sys1bench results") -> str:
    data = collect(dirs)
    payload = json.dumps(data, default=lambda o: None if isinstance(o, float) and o != o else str(o)).replace("</", "<\\/")
    payload = payload.replace("NaN", "null")
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)
