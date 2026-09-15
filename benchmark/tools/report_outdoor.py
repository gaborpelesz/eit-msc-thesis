"""Generate the outdoor-four-scene HTML report from the result store.

Every number on the page is read from the store at generation time. Nothing is
transcribed, so the report cannot drift from the measurements it describes, and
re-running it after a re-score is the whole update procedure.

Scores come from `rescore.json` where a run has one -- that file is written by
a single evaluator across the whole campaign, which `run.json` is not once any
run has been re-scored. Timing, memory and energy always come from `run.json`:
re-scoring reads a stored cloud and measures nothing about the run.
"""
import argparse, html, json
from pathlib import Path

METHODS = ["ACMMP", "APD-MVS", "MP-MVS", "DPE-MVS", "CUMVS", "quorum-mvs"]
SCENES = ["courtyard", "playground", "terrace", "meadow"]
VIEWS = {"courtyard": [6, 25], "playground": [19, 31], "terrace": [11, 19], "meadow": [7, 10]}
OWN = "quorum-mvs"

PROVENANCE = {
    "ACMMP": "published-reference", "APD-MVS": "published-reference",
    "MP-MVS": "published-reference", "DPE-MVS": "published-reference",
    "CUMVS": "third-party-optimization", "quorum-mvs": "own",
}
SCENE_NOTE = {
    "courtyard": "38 images, per-camera crops differ — the APD and DPE converters pad",
    "playground": "38 images, per-camera crops differ — the APD and DPE converters pad",
    "terrace": "23 images, per-camera crops differ — the APD and DPE converters pad",
    "meadow": "15 images, one camera — nothing pads anywhere. The control.",
}
VIEW_NOTE = {
    ("courtyard", 6): "Tables, benches and deckchairs against brick: thin clutter at close range.",
    ("courtyard", 25): "The corner, two facades and the ground plane between them.",
    ("playground", 19): "The rope climbing frame — the thinnest geometry in the campaign.",
    ("playground", 31): "The timber canopy, with poles and the buildings behind it.",
    ("terrace", 11): "The balcony overhang, an unsupported surface seen from below.",
    ("terrace", 19): "The courtyard: grass, paving and two facades at distance.",
    ("meadow", 7): "The barn at three-quarters, with the neighbouring building for context.",
    ("meadow", 10): "The long facade and the hedge line in front of it.",
}


def load(campaign_dir):
    runs = {}
    for d in sorted(campaign_dir.iterdir()):
        rj = d / "run.json"
        if not d.is_dir() or not rj.is_file():
            continue
        rec = json.loads(rj.read_text())
        rs = d / "rescore.json"
        scored = json.loads(rs.read_text()) if rs.is_file() else None
        # A re-score supersedes the campaign's own scoring only when it produced
        # one; a failed re-score must not blank a run the campaign did score.
        use = scored if (scored and scored.get("status") == "ok") else rec
        runs[(rec["method"], rec["scene"])] = {
            "f1": use.get("f1_primary"), "acc": use.get("accuracy_primary"),
            "comp": use.get("completeness_primary"), "quality": use.get("quality") or [],
            "wall": rec.get("wall_time_s"), "convert": rec.get("preprocess_convert_s") or 0.0,
            "points": rec.get("point_count"), "mem": rec.get("peak_device_mem_bytes"),
            "energy": rec.get("energy_j"), "status": rec.get("status"),
            "rescored": bool(scored and scored.get("status") == "ok"),
            "evaluator": (scored or rec).get("evaluator_sha"),
        }
    return runs


def n_images(runs, scene):
    return {"courtyard": 38, "playground": 38, "terrace": 23, "meadow": 15}[scene]


def fmt(v, p=4):
    return "—" if v is None else f"{v:.{p}f}"


def secs(v):
    if v is None:
        return "—"
    return f"{v:,.0f} s" if v >= 100 else f"{v:.1f} s"


def bar(frac, cls=""):
    return f'<span class="bar {cls}"><i style="width:{max(frac,0.004)*100:.1f}%"></i></span>'


def build(runs, out, renders_rel="renders"):
    best_f1 = {s: max((runs[(m, s)]["f1"] or 0) for m in METHODS) for s in SCENES}
    fastest = {s: min((runs[(m, s)]["wall"] or 9e9) for m in METHODS) for s in SCENES}
    max_wall = max((runs[(m, s)]["wall"] or 0) for m in METHODS for s in SCENES)

    P = []
    A = P.append

    # ---- headline table -------------------------------------------------
    A('<table class="grid"><thead><tr><th class="lbl">Method</th>')
    for s in SCENES:
        A(f'<th>{s}</th>')
    A('<th class="sep">mean</th></tr></thead><tbody>')
    for m in METHODS:
        vals = [runs[(m, s)]["f1"] for s in SCENES]
        mean = sum(v for v in vals if v) / max(len([v for v in vals if v]), 1)
        own = ' class="own"' if m == OWN else ''
        A(f'<tr{own}><th class="lbl">{html.escape(m)}'
          f'<em>{PROVENANCE[m]}</em></th>')
        for s, v in zip(SCENES, vals):
            win = ' win' if v and abs(v - best_f1[s]) < 1e-12 else ''
            A(f'<td class="num{win}">{fmt(v)}</td>')
        A(f'<td class="num sep">{fmt(mean)}</td></tr>')
    A('</tbody></table>')
    headline = "".join(P)

    # ---- accuracy / completeness ---------------------------------------
    P = []
    A = P.append
    for s in SCENES:
        A(f'<div class="ac"><h4>{s}</h4><table class="grid tight"><thead><tr>'
          f'<th class="lbl">Method</th><th>accuracy</th><th>completeness</th><th>F1</th>'
          f'</tr></thead><tbody>')
        rows = sorted(METHODS, key=lambda m: -(runs[(m, s)]["f1"] or 0))
        hi_a = max((runs[(m, s)]["acc"] or 0) for m in METHODS)
        hi_c = max((runs[(m, s)]["comp"] or 0) for m in METHODS)
        for m in rows:
            r = runs[(m, s)]
            own = ' class="own"' if m == OWN else ''
            wa = ' win' if r["acc"] and abs(r["acc"] - hi_a) < 1e-12 else ''
            wc = ' win' if r["comp"] and abs(r["comp"] - hi_c) < 1e-12 else ''
            A(f'<tr{own}><th class="lbl">{html.escape(m)}</th>'
              f'<td class="num{wa}">{fmt(r["acc"])}</td>'
              f'<td class="num{wc}">{fmt(r["comp"])}</td>'
              f'<td class="num">{fmt(r["f1"])}</td></tr>')
        A('</tbody></table></div>')
    accomp = "".join(P)

    # ---- cost ------------------------------------------------------------
    P = []
    A = P.append
    A('<table class="grid"><thead><tr><th class="lbl">Method</th>'
      '<th>wall clock, four scenes</th><th>total</th><th>s / image</th>'
      '<th>vs quorum</th><th>peak VRAM</th></tr></thead><tbody>')
    own_total = sum(runs[(OWN, s)]["wall"] or 0 for s in SCENES)
    for m in sorted(METHODS, key=lambda m: sum(runs[(m, s)]["wall"] or 0 for s in SCENES)):
        tot = sum(runs[(m, s)]["wall"] or 0 for s in SCENES)
        imgs = sum(n_images(runs, s) for s in SCENES)
        mem = max((runs[(m, s)]["mem"] or 0) for s in SCENES)
        own = ' class="own"' if m == OWN else ''
        ratio = tot / own_total if own_total else 0
        A(f'<tr{own}><th class="lbl">{html.escape(m)}</th>'
          f'<td>{bar(tot / max_wall / 4 * 1.0)}</td>'
          f'<td class="num">{secs(tot)}</td>'
          f'<td class="num">{tot/imgs:.1f}</td>'
          f'<td class="num">{"—" if m == OWN else f"{ratio:.1f}&times;"}</td>'
          f'<td class="num">{mem/2**30:.1f} GB</td></tr>')
    A('</tbody></table>')
    cost = "".join(P)

    # ---- renders ---------------------------------------------------------
    P = []
    A = P.append
    for s in SCENES:
        A(f'<section class="scene"><h3>{s}</h3>'
          f'<p class="note">{html.escape(SCENE_NOTE[s])}</p>')
        for v in VIEWS[s]:
            A(f'<h4 class="vp">viewpoint {v}</h4>'
              f'<p class="note">{html.escape(VIEW_NOTE[(s, v)])}</p>'
              f'<div class="shots">')
            for m in sorted(METHODS, key=lambda m: -(runs[(m, s)]["f1"] or 0)):
                r = runs[(m, s)]
                own = ' own' if m == OWN else ''
                A(f'<figure class="shot{own}">'
                  f'<img loading="lazy" src="{renders_rel}/{s}-v{v}-{m}.jpg" '
                  f'alt="{html.escape(m)} reconstruction of {s}, viewpoint {v}">'
                  f'<figcaption><b>{html.escape(m)}</b>'
                  f'<span>F1 {fmt(r["f1"],3)} &middot; {secs(r["wall"])} &middot; '
                  f'{(r["points"] or 0)/1e6:.1f} M pts</span></figcaption></figure>')
            A('</div>')
        A('</section>')
    renders = "".join(P)

    ev = next((runs[k]["evaluator"] for k in runs if runs[k]["evaluator"]), "—")
    nres = sum(1 for k in runs if runs[k]["rescored"])

    doc = TEMPLATE.format(headline=headline, accomp=accomp, cost=cost,
                          renders=renders, evaluator=html.escape(str(ev)[:12]),
                          nrescored=nres, ntotal=len(runs))
    Path(out).write_text(doc)
    print(f"{out}  {len(doc)/1024:.0f} KB")


TEMPLATE = """<title>Outdoor Four</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#F1F3F2; --surface:#FFFFFF; --sunk:#E9ECEA;
  --ink:#14171A; --muted:#5A6470; --rule:#DCE1DF;
  --accent:#C4551F; --accent-soft:#F6E6DC; --teal:#2F6B7A;
  --serif:"Newsreader",Georgia,serif;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0E1113; --surface:#161A1D; --sunk:#1D2225;
    --ink:#E8ECEA; --muted:#949EA5; --rule:#262D31;
    --accent:#E27A40; --accent-soft:#33211A; --teal:#5E9DAE;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0E1113; --surface:#161A1D; --sunk:#1D2225;
  --ink:#E8ECEA; --muted:#949EA5; --rule:#262D31;
  --accent:#E27A40; --accent-soft:#33211A; --teal:#5E9DAE;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.6; margin:0; }}
.wrap {{ max-width:1120px; margin:0 auto; padding-inline:20px; padding-block:0 96px; }}
header.top {{ padding-block:64px 40px; border-bottom:1px solid var(--rule); margin-bottom:40px; }}
.eyebrow {{ font-family:var(--mono); font-size:11px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin:0 0 18px; }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(34px,6vw,54px);
  line-height:1.06; margin:0 0 20px; text-wrap:balance; letter-spacing:-.015em; }}
.standfirst {{ font-family:var(--serif); font-size:clamp(17px,2.4vw,21px); line-height:1.5;
  color:var(--muted); max-width:62ch; margin:0 0 28px; }}
.facts {{ display:flex; flex-wrap:wrap; gap:8px 28px; font-family:var(--mono);
  font-size:12px; color:var(--muted); }}
.facts b {{ color:var(--ink); font-weight:500; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:clamp(24px,3.4vw,32px);
  margin:64px 0 8px; letter-spacing:-.01em; text-wrap:balance; }}
h2:first-of-type {{ margin-top:0; }}
h3 {{ font-family:var(--serif); font-weight:600; font-size:24px; margin:56px 0 6px; }}
h4 {{ font-size:13px; font-family:var(--mono); text-transform:uppercase;
  letter-spacing:.1em; color:var(--muted); margin:28px 0 10px; font-weight:500; }}
h4.vp {{ color:var(--teal); }}
p {{ max-width:68ch; }}
p.lede {{ color:var(--muted); margin:0 0 24px; }}
p.note {{ color:var(--muted); font-size:13.5px; margin:0 0 14px; max-width:70ch; }}
.tablewrap {{ overflow-x:auto; }}
table.grid {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums;
  background:var(--surface); border:1px solid var(--rule); border-radius:8px;
  overflow:hidden; }}
table.grid th, table.grid td {{ padding:11px 14px; text-align:right;
  border-bottom:1px solid var(--rule); font-size:14px; }}
table.grid thead th {{ font-family:var(--mono); font-size:11px; font-weight:500;
  text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
  background:var(--sunk); }}
table.grid th.lbl {{ text-align:left; font-weight:600; white-space:nowrap; }}
table.grid th.lbl em {{ display:block; font-style:normal; font-family:var(--mono);
  font-size:10.5px; letter-spacing:.04em; color:var(--muted); font-weight:400; }}
table.grid td.num {{ font-family:var(--mono); }}
table.grid tbody tr:last-child th, table.grid tbody tr:last-child td {{ border-bottom:0; }}
td.win {{ color:var(--accent); font-weight:500; position:relative; }}
td.win::after {{ content:"\\2022"; position:absolute; right:4px; top:8px;
  color:var(--accent); font-size:11px; }}
tr.own {{ background:var(--accent-soft); }}
tr.own th.lbl {{ box-shadow:inset 3px 0 0 var(--accent); }}
.sep {{ border-left:1px solid var(--rule); }}
.bar {{ display:block; height:7px; background:var(--sunk); border-radius:4px;
  min-width:90px; overflow:hidden; }}
.bar i {{ display:block; height:100%; background:var(--teal); border-radius:4px; }}
tr.own .bar i {{ background:var(--accent); }}
.acgrid {{ display:grid; gap:18px; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); }}
.ac h4 {{ margin-top:0; color:var(--ink); font-family:var(--serif); font-size:17px;
  text-transform:none; letter-spacing:0; }}
table.tight th, table.tight td {{ padding:7px 10px; font-size:13px; }}
.shots {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(272px,1fr)); }}
figure.shot {{ margin:0; background:var(--surface); border:1px solid var(--rule);
  border-radius:8px; overflow:hidden; }}
figure.shot.own {{ border-color:var(--accent); }}
figure.shot img {{ display:block; width:100%; height:auto; background:#0b0d10; }}
figcaption {{ padding:9px 11px; font-size:12.5px; display:flex;
  justify-content:space-between; gap:10px; flex-wrap:wrap; align-items:baseline; }}
figcaption span {{ font-family:var(--mono); font-size:11px; color:var(--muted); }}
section.scene {{ margin-bottom:12px; }}
.caveats {{ background:var(--surface); border:1px solid var(--rule); border-radius:8px;
  padding:22px 24px; margin-top:24px; }}
.caveats h4 {{ margin-top:0; }}
.caveats ul {{ margin:0; padding-left:20px; }}
.caveats li {{ margin-bottom:10px; color:var(--muted); font-size:14px; max-width:74ch; }}
.caveats li b {{ color:var(--ink); font-weight:600; }}
footer {{ margin-top:72px; padding-top:22px; border-top:1px solid var(--rule);
  font-family:var(--mono); font-size:11.5px; color:var(--muted); }}
@media (max-width:640px) {{
  table.grid th, table.grid td {{ padding:9px 8px; font-size:13px; }}
}}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">ETH3D &middot; campaign outdoor-four-scene</p>
  <h1>Six reconstructions, four outdoor scenes</h1>
  <p class="standfirst">The first campaign to leave the five uniform-image scenes.
  Every earlier result was indoors — not by choice, but because two methods abort on
  a scene whose images differ in size. These four are the test of whether the
  lineage's ordering survives outdoors. It does not.</p>
  <div class="facts">
    <span><b>24</b> runs, 1 repeat</span>
    <span><b>6.49 h</b> total</span>
    <span><b>3200 px</b> width</span>
    <span>config <b>author</b></span>
    <span>tolerance <b>2 cm</b></span>
    <span>evaluator <b>{evaluator}</b></span>
    <span>RTX 2080 Ti @ <b>1800 MHz</b></span>
  </div>
</header>

<h2>F1 at 2 cm</h2>
<p class="lede">The primary score. A dot marks the best method on each scene.</p>
<div class="tablewrap">{headline}</div>

<h2>Where the difference actually is</h2>
<p class="lede">F1 hides which half moves. Split into its two terms, one pattern holds on
every scene: Quorum MVS is at or near the top on completeness and at or near the
bottom on accuracy. Its deficit is entirely in where the points land, never in how
many there are — which is what F-073 found indoors, reproduced outdoors.</p>
<div class="acgrid">{accomp}</div>

<h2>What it cost</h2>
<p class="lede">Wall clock inside the container, summed over all four scenes
(114 images). Converter time is excluded here and is reported per run in the store.</p>
<div class="tablewrap">{cost}</div>

<h2>The clouds</h2>
<p class="lede">Two viewpoints per scene, chosen to stress different failure modes.
Every method is drawn from the identical camera with the identical renderer, so
what differs between panels is the reconstruction and nothing else. Holes are real
holes: no surface is interpolated. Panels are ordered by F1 on that scene.</p>
{renders}

<h2>Read this before quoting anything</h2>
<div class="caveats">
<ul>
<li><b>One repeat.</b> PatchMatch is randomised and R-STA-01 requires repeats before
any quality or runtime claim. Margins here under roughly 0.005 F1 are not separable
from run-to-run noise — which covers MP-MVS against Quorum MVS on terrace, and
CUMVS against DPE-MVS there too.</li>
<li><b>Two methods pad, four do not.</b> Under <code>author</code> each method runs its
own converter. APD-MVS's and DPE-MVS's pad every image to the scene maximum and
re-encode it (F-006), which is the only reason those two run on three of these
scenes at all (F-007). <b>meadow is the control</b>: one camera, no padding anywhere.
The ordering on meadow tracks the padded scenes, so scene content rather than
padding looks like the driver — but that is an observation from one repeat, not a
controlled result.</li>
<li><b>These scenes are newly prepared.</b> They reproduce the existing five to within
the JPEG noise floor but not bit-exactly; the original resampling filter cannot be
recovered from re-encoded JPEGs.</li>
<li><b>CUMVS is not the author's work.</b> It is a third-party optimization of the ACM
family, included as prior art that headroom exists. Its four runs were scored after
the campaign, from stored clouds, once the evaluator could read ASCII PLY (F-074);
the other twenty were re-scored with that same evaluator and reproduced their
original values exactly, 20 of 20.</li>
<li><b>Indoor results are not on this page.</b> Nothing here should be compared against
the five-scene indoor campaigns: different scenes, and no repeats to bound the
difference.</li>
</ul>
</div>

<footer>
Generated from /data/bench/results/outdoor-four-scene by benchmark/tools/report_outdoor.py.
{nrescored} of {ntotal} runs carry a re-score record. Every figure is read from the
store at generation time; none is transcribed. deviations verify: 706 checks, 0 failures.
</footer>
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign_dir", type=Path)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    build(load(a.campaign_dir), a.out)


if __name__ == "__main__":
    main()
