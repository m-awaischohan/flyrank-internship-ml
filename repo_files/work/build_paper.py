"""Builds docs/index.html (the deployed paper) from work/outputs/capstone_metrics.json.
Every number on the page is read from that file, which capstone.ipynb writes.
Run:  python work/build_paper.py   (from the repo root, after running the capstone notebook)"""
import json, os, shutil, html

M = json.load(open("work/outputs/capstone_metrics.json"))
d, main, des, pf, lk, er, ar, dc, co, mo, env = (M[k] for k in ("data", "main", "designs", "per_fold", "leakage", "errors", "archetypes", "decay", "cost", "monitoring", "environment"))
REPO = "https://github.com/m-awaischohan/flyrank-internship-ml"
NB = REPO + "/blob/main/work/notebooks/"
f2 = lambda x: f"{x:.2f}"
pc = lambda x: f"{round(100 * x):d}%"
n0 = lambda x: f"{int(round(x)):,}"
Q, R, LR, GB = main["average of both (the queue)"], main["week-4 rule"], main["logistic regression"], main["gradient boosting"]
BASE = d["base_rate"]
pf_min = min(x["queue_p50"] for x in pf); pf_max = max(x["queue_p50"] for x in pf)
rule_min = min(x["rule_p50"] for x in pf)
A, B, C = (des[k] for k in des)
below3000 = er["low_later_pages_below_rank_3000"] / er["low_later_pages_total"]
batch = co["batches"]
Qb, Rb, Cb = batch["this queue, top 50"], batch["week-4 rule top 50"], batch["this queue, capped batch (max 15 per client)"]

def table(head, rows, cls="", hl=None):
    h = "".join(f"<th scope='col'>{html.escape(str(c))}</th>" for c in head)
    body = ""
    for i, r in enumerate(rows):
        tr = " class='hl'" if hl is not None and i == hl else ""
        body += f"<tr{tr}>" + "".join(f"<td>{c}</td>" if j else f"<th scope='row'>{c}</th>" for j, c in enumerate(r)) + "</tr>"
    return f"<div class='tw'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>"

def fig(src, alt, cap, num):
    return (f"<figure id='fig{num}'><img src='img/{src}.png' alt='{html.escape(alt)}' loading='lazy' width='1480' height='840'>"
            f"<figcaption><strong>Figure {num}.</strong> {cap}</figcaption></figure>")

res_rows = [[k, f2(v["p20"]), f2(v["p50"]), f2(v["p100"]), f2(v["auc"])] for k, v in main.items()]
res_tbl = table(["Strategy (same pages, same label)", "precision@20", "precision@50", "precision@100", "AUC"], res_rows, hl=5)

des_rows = []
for m_, lbl in (("rule", "week-4 rule"), ("logreg", "logistic regression"), ("gboost", "gradient boosting")):
    des_rows.append([lbl] + [f"{f2(v[m_]['p50'])} / {v[m_]['auc']:.3f}" for v in (A, B, C)])
des_tbl = table(["Model", "A. random split, norms from all pages", "B. random split, norms from training only", "C. grouped by client, norms from training only"], des_rows)

pf_tbl = table(["Fold", "Clients held out", "Pages", "Base rate", "Queue precision@50", "Rule precision@50"],
               [[x["fold"], x["clients"], n0(x["pages"]), f2(x["base_rate"]), f2(x["queue_p50"]), f2(x["rule_p50"])] for x in pf])

leak_expl = {"final features (reference)": "The reference score.",
             "drop avg_position": "A 90-day average that overlaps the later window. Barely matters.",
             "drop band-norm features (ratio, gap, miss)": "The position-relative features are not carrying the result alone.",
             "add days_since_last_update": "Excluded on timing grounds; adding it does not raise the score.",
             "add ctr_last (deliberate leak)": "The label's own ingredient. The score jumps to 1.0, so the test can detect a leak.",
             "shuffled labels": "Labels randomly permuted. About chance, as it should be."}
leak_tbl = table(["Test (gradient boosting, grouped by client)", "AUC", "precision@50", "What it shows"],
                 [[k, f"{v['auc']:.3f}", f2(v["p50"]), leak_expl[k]] for k, v in lk.items()])

arch_rows = [[k, n0(v["labelled"]), f2(v["low_later_rate"]), v["in_top100"], html.escape(v["first_action"])] for k, v in ar.items()]
arch_tbl = table(["Archetype", "Labelled pages", "Still low a month later", "In queue's top 100", "First review action"], arch_rows)

cost_rows = [[k, f2(v["precision50"]), f"{v['found']:g}", f"{v['hours_per_page_found']:g}", n0(v["clicks_at_stake"]), n0(v["clicks_at_stake_still_low"])] for k, v in batch.items()]
cost_tbl = table(["50-page weekly batch", "precision@50", "Pages found that stayed low", "Reviewer hours per page found", "Clicks at stake (earlier 30 days)", "...of which on pages that stayed low"], cost_rows, hl=3)

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Which pages should a reviewer open first? Ranking low-CTR search pages and testing the ranking honestly</title>
<meta name="description" content="A research write-up on ranking already-visible pages whose click-through rate is far below their position's norm, validated on held-out clients. Observational, decision-support only.">
<meta property="og:title" content="Which pages should a reviewer open first?">
<meta property="og:description" content="A baseline-vs-model study on anonymized FlyRank internship data, validated on held-out clients, with leakage tests and honest limits.">
<style>
:root{{--bg:#fbfaf7;--ink:#1b1b19;--muted:#5a5954;--line:#dedcd4;--card:#f2f0ea;--accent:#1f5fb4;--accent-ink:#ffffff;--warn:#8a4b08;--warn-bg:#fbf0e0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151513;--ink:#ecebe6;--muted:#aaa8a0;--line:#383731;--card:#1f1e1b;--accent:#7ab0f5;--accent-ink:#0b0b0b;--warn:#f0b968;--warn-bg:#2a2216}}}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth;-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--bg);color:var(--ink);font:clamp(16px,1.05vw + 12px,18px)/1.65 Charter,'Iowan Old Style','Palatino Linotype',Georgia,serif}}
a{{color:var(--accent)}}
.skip{{position:absolute;left:-999px}}.skip:focus{{left:12px;top:12px;background:var(--accent);color:var(--accent-ink);padding:8px 12px;z-index:9}}
.wrap{{max-width:1120px;margin:0 auto;padding:0 20px;display:grid;grid-template-columns:minmax(0,1fr);gap:0 48px}}
@media (min-width:1000px){{.wrap{{grid-template-columns:210px minmax(0,44rem)}}}}
nav.toc{{font:14px/1.4 system-ui,-apple-system,'Segoe UI',sans-serif;padding:20px 0}}
@media (min-width:1000px){{nav.toc{{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding-top:56px}}}}
nav.toc b{{display:block;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-size:12px;margin-bottom:10px}}
nav.toc ol{{margin:0;padding:0;list-style:none}}nav.toc li{{margin:0}}
nav.toc a{{display:block;padding:5px 0;color:var(--muted);text-decoration:none;border-left:2px solid transparent;padding-left:10px}}
nav.toc a:hover,nav.toc a:focus{{color:var(--ink);border-left-color:var(--accent)}}
@media (max-width:999px){{nav.toc ol{{display:flex;flex-wrap:wrap;gap:4px 14px}}nav.toc a{{border-left:0;padding-left:0}}}}
main{{padding:8px 0 80px;min-width:0}}
header.title{{padding:28px 0 8px}}
.kicker{{font:600 13px/1 system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}}
h1{{font-size:clamp(1.7rem,3.4vw + .6rem,2.5rem);line-height:1.15;margin:.5rem 0 .75rem;letter-spacing:-.01em}}
.byline{{font:15px/1.5 system-ui,sans-serif;color:var(--muted)}}
h2{{font-size:1.5rem;line-height:1.25;margin:3rem 0 .6rem;padding-top:1rem;border-top:1px solid var(--line)}}
h2 .n{{color:var(--muted);font-weight:400;margin-right:.4rem}}
h3{{font-size:1.1rem;margin:1.8rem 0 .3rem}}
p,ul,ol{{margin:.7rem 0}}li{{margin:.3rem 0}}
.abstract{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 22px;margin:20px 0}}
.abstract h2{{border:0;margin:0 0 .3rem;padding:0;font-size:1.1rem;font-family:system-ui,sans-serif;text-transform:uppercase;letter-spacing:.06em}}
.glance{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:18px 0}}
.glance div{{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--card)}}
.glance .big{{display:block;font:700 1.7rem/1.1 system-ui,sans-serif}}
.glance .sm{{display:block;font:13px/1.35 system-ui,sans-serif;color:var(--muted);margin-top:4px}}
.callout{{background:var(--warn-bg);border-left:4px solid var(--warn);padding:10px 16px;border-radius:4px;font-size:.96em}}
figure{{margin:1.6rem 0}}figure img{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fcfcfb}}
figcaption{{font:14px/1.5 system-ui,sans-serif;color:var(--muted);margin-top:8px}}figcaption strong{{color:var(--ink)}}
.tw{{overflow-x:auto;margin:1rem 0;-webkit-overflow-scrolling:touch}}
table{{border-collapse:collapse;width:100%;font:14.5px/1.4 system-ui,sans-serif;min-width:520px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
thead th{{font-size:12.5px;text-transform:none;color:var(--muted);border-bottom:2px solid var(--line)}}
tbody th{{font-weight:600}}tr.hl td,tr.hl th{{background:var(--card)}}
code{{font:.88em ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--card);padding:1px 5px;border-radius:4px}}
pre{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow:auto;font:13.5px/1.5 ui-monospace,Menlo,monospace}}
.nogo li::marker{{color:var(--warn)}}
footer{{border-top:1px solid var(--line);margin-top:3rem;padding-top:1rem;color:var(--muted);font:14px/1.5 system-ui,sans-serif}}
@media print{{nav.toc{{display:none}}.wrap{{display:block}}a{{color:inherit}}figure,table{{break-inside:avoid}}}}
</style>
</head>
<body>
<a class="skip" href="#abstract">Skip to the abstract</a>
<div class="wrap">
<nav class="toc" aria-label="Sections"><b>Sections</b><ol>
<li><a href="#abstract">Abstract</a></li><li><a href="#introduction">1 Introduction</a></li><li><a href="#data">2 Data</a></li>
<li><a href="#methodology">3 Methodology</a></li><li><a href="#results">4 Results</a></li><li><a href="#limitations">5 Limitations</a></li>
<li><a href="#recommendations">6 Ranked recommendations</a></li><li><a href="#reproducibility">7 Reproducibility</a></li><li><a href="#acknowledgments">8 Acknowledgments</a></li></ol></nav>
<main>
<header class="title">
<div class="kicker">Research write-up · FlyRank ML Internship · Machine Learning track</div>
<h1>Which pages should a reviewer open first? Ranking low-CTR search pages, and testing the ranking honestly</h1>
<div class="byline">By Tayyab · November 2026 · Independent intern project, not an official FlyRank publication</div>
</header>

<section id="abstract" class="abstract" aria-labelledby="abs-h"><h2 id="abs-h">Abstract</h2>
<p><strong>Question.</strong> FlyRank-style content work has a practical version of a scaling problem: an editor cannot re-read every page that already appears in search, so which visible pages under-capture clicks (a much lower click-through rate, CTR, than their position normally earns), and are likely to stay that way, deserve a metadata, content or monitoring review first?
<strong>Data.</strong> {n0(d['labelled_pages'])} pages from {d['clients_labelled']} anonymized clients in the FlyRank ML Internship starter slice, with two 30-day windows (an earlier one for inputs, a later one for the label).
<strong>Method.</strong> I compared a one-line rule built before any modelling against logistic regression and gradient boosting, validating on whole clients held out from training, and attacked my own pipeline with leakage tests.
<strong>Result.</strong> On clients it had not seen, the average of the two models put {round(Q['p50'] * 50)} of its top 50 pages ({pc(Q['p50'])}) among pages that stayed far below their position's typical CTR, against {pc(R['p50'])} for the rule and a base rate of {pc(BASE)}, but one client supplies {er['top100_from_largest_client']} of the top 100 and the result ranges from {f2(pf_min)} to {f2(pf_max)} across held-out folds.
<strong>Use.</strong> A ranked review queue with plain reason codes for a human reviewer; the data are observational, so the paper makes no claim that any edit raises CTR.</p></section>

<div class="glance" role="list" aria-label="Findings at a glance">
<div role="listitem"><span class="big">{f2(Q['p50'])} vs {f2(R['p50'])}</span><span class="sm">precision@50, queue vs my own rule (base rate {f2(BASE)}), clients held out</span></div>
<div role="listitem"><span class="big">{C['gboost']['auc']:.3f} vs {A['gboost']['auc']:.3f}</span><span class="sm">gradient-boosting AUC, grouped by client vs random split: the models barely moved</span></div>
<div role="listitem"><span class="big">{er['top100_from_largest_client']} of 100</span><span class="sm">top picks come from one client, which is the main caution on this result</span></div>
<div role="listitem"><span class="big">{pc(dc['still_low_later'])}</span><span class="sm">of flagged pages were still far below norm a month later, with no known action taken</span></div>
</div>

<section id="introduction"><h2><span class="n">1</span>Introduction</h2>
<p>Picture a content reviewer on Monday morning. There are thousands of pages that already appear in search, and time to read about fifty. Some of those pages earn clicks at about the rate their position suggests. Others sit at the same position but get far fewer clicks, which can point to a weak search snippet, a mismatched query, or something on the results page that has nothing to do with the page. Reading the right fifty first is the decision this work supports.</p>
<h3>The content problem this comes from (the case study)</h3>
<p>This work started from a lane brief in the FlyRank ML Internship, which poses the problem this way: <em>which visible pages under-capture clicks or engagement and deserve metadata, content, or monitoring review?</em> The same program's data comes from content-refresh work, where a page that already ranks can still lose clicks because its search snippet, its match to the query, or its content has drifted. FlyRank's own March 2026 research report names refresh timing as a lever (its &ldquo;Freshness Multiplier&rdquo; finding). This paper does not test that lever. It addresses the step before it: <strong>where should a human look first?</strong> Section 6 returns to the case and says what the findings mean for it.</p>
<p>The obvious approach is a rule: compare each page's CTR with the typical CTR for its position and flag the big gaps. That rule is the baseline here. The question is whether a model, trained on the same information and tested on clients it never saw, ranks the pages that stay low better than the rule does, and how much of any improvement survives an honest test.</p>
<p>This paper is deliberately modest in what it claims. It describes a <em>ranking</em> that helps a person decide what to read first. It does not show that changing a page raises its CTR; the data cannot show that.</p>
<p class="callout"><strong>How to read the claim words.</strong> <em>Observed</em>: a pattern in this data. <em>Measured</em>: a comparison between groups. <em>Directional</em>: a result with known weaknesses, such as concentration in one client. <em>Decision-support</em>: an ordering for a human reviewer to use, not an instruction.</p></section>

<section id="data"><h2><span class="n">2</span>Data</h2>
<p><strong>Source.</strong> The anonymized starter file of the FlyRank ML Internship dataset: {n0(d['rows'])} content pages across {d['clients_total']} pseudonymized clients, {d['columns']} columns, with trailing-90-day search and analytics totals plus an earlier-30-day and a later-30-day split. Page and client identifiers are pseudonyms and are used only for grouping. <strong>Every number in this paper comes from this file.</strong> The larger hosted warehouse release exists, and I documented its contract in a separate notebook, but I did not use it for any result here.</p>
<p><strong>What I kept.</strong> Pages with at least 100 impressions in the earlier 30 days and a real average position between 1 and 20 ({n0(d['no_position_rows'])} rows had no position data). That leaves {n0(d['eligible_pages'])} pages. The label also needs at least 100 impressions in the later 30 days, which leaves <strong>{n0(d['labelled_pages'])} pages from {d['clients_labelled']} clients</strong>.</p>
<p><strong>What that filter costs.</strong> The filter drops {pc(d['dropped_share'])} of in-scope pages, and the dropped pages were much smaller (median {n0(d['median_earlier_impressions_dropped'])} earlier impressions against {n0(d['median_earlier_impressions_kept'])} for kept pages). Pages that disappeared from search are therefore not in the evaluation, and every claim below carries that qualifier.</p>
<p><strong>What I excluded from the inputs, and why.</strong> The later-window columns (they define the label); the trend columns (the label source for another question); the 90-day totals and rates (they contain the later 30 days); the days since the last update (an update inside the later window happens after the prediction would have been made); and the page and client identifiers.</p>
<p><strong>Concentration.</strong> The largest client holds {pc(d['largest_client_share'])} of labelled pages. The base rate is {f2(BASE)} overall and {f2(d['base_rate_without_largest_client'])} without that client.</p></section>

<section id="methodology"><h2><span class="n">3</span>Methodology</h2>
<h3>The label (a proxy)</h3>
<p>Pages are grouped into five position bands (1-3, 4-6, 7-10, 11-15, 16-20). A band's typical CTR is total clicks divided by total impressions in that band. A page is <code>low_later</code> if its CTR in the later 30 days is under <strong>half</strong> of its band's typical CTR in that window. This is a proxy I defined. It is not the observed result of any fix, and the threshold of one half is a choice.</p>
<h3>The baseline, built first</h3>
<p>Compare a page's earlier-30-day CTR with its band's typical CTR. Flag it if it is under half of that norm <em>and</em> at least 10 clicks were expected; score it by the missing clicks (expected minus actual). I froze this rule in an earlier week and did not tune it afterwards.</p>
<h3>The models</h3>
<p>Logistic regression, which is the one I can explain; gradient boosting with depth-3 trees, as a check; and the average of the two probabilities, which is the review queue's score. All three use the same 17 features, each known before the label window: earlier-window impressions, clicks and sessions, earlier-window CTR, average position, page age, word count and keyword and intent fields (with flags for missing values instead of silent zero-filling), and three position-relative features built from the band norm.</p>
<h3>Validation design</h3>
<p>Five folds <strong>grouped by client</strong>: every page is scored by a model that never saw any page from its client, and band norms are recomputed from the training clients only. The question being answered is &ldquo;does it work on a client it has not seen?&rdquo;. The file holds only two 30-day windows, so a time-aware split is not possible; this paper tests new clients, not new months.</p>
<h3>Measure</h3>
<p>Precision@K is the share of the K highest-ranked pages that really were low a month later. It is always shown next to the base rate ({f2(BASE)}), because a high score can just reflect a high base rate. AUC is reported as the discrimination number.</p>
<h3>Leakage checks</h3>
<p>I audited every feature against a timeline (inputs strictly before the label window) and ran four tests, shown in Section 4. Two are harness checks: a deliberately leaky feature must push the score to 1.0, and shuffled labels must give about 0.5.</p></section>

<section id="results"><h2><span class="n">4</span>Results (model vs baseline)</h2>
<p>Same pages, same label, same client-grouped folds, same measure in every row. The queue is the average of the two models.</p>
{res_tbl}
{fig('fig_queue_vs_baselines', f"Bar chart of precision at 50: random picking {f2(main['random picking (average of 200)']['p50'])}, biggest pages first {f2(main['biggest pages first']['p50'])}, week-4 rule {f2(R['p50'])}, logistic regression {f2(LR['p50'])}, gradient boosting {f2(GB['p50'])}, the queue {f2(Q['p50'])}; base rate {f2(BASE)}.", f"Precision@50 by strategy, scored on held-out clients. Takeaway: the models beat my one-line rule, and &ldquo;biggest pages first&rdquo; is no better than chance. The gradient-boosting bar is the one most affected by client concentration (Section 5).", 1)}
<h3>What changed when I made the validation honest</h3>
<p>My earlier work used a random split and let the position-band norms include the pages being scored. I re-ran the same models and rule under three designs. Only the design changes between columns; each cell shows precision@50 / AUC.</p>
{des_tbl}
{fig('fig_validation_designs', f"Line chart of precision at 50 across three validation designs. Gradient boosting stays near 1.00, logistic regression stays at {f2(C['logreg']['p50'])}, and the week-4 rule falls from {f2(A['rule']['p50'])} to {f2(C['rule']['p50'])}.", "Precision@50 across three validation designs. Takeaway: the rule lost the most when I stopped letting pages help set their own benchmark; the models changed little, so I found little sign they were memorising clients.", 2)}
<p>I expected the grouped split to hurt the models more than it did, and I report the small gap as a result rather than a failure of the test. The rule's drop from {f2(A['rule']['p50'])} to {f2(C['rule']['p50'])} comes only from the norms, since the rule has no training step.</p>
<h3>Held-out folds</h3>
{pf_tbl}
{fig('fig_per_fold', "Grouped bars of precision at 50 per held-out fold, queue in blue and the week-4 rule in orange, with each fold's base rate marked.", f"Precision@50 in each of the five folds. Takeaway: the queue leads in every fold, but the range is wide ({f2(pf_min)} to {f2(pf_max)}) and fold 0 is a single large client. The rule's own range reaches down to {f2(rule_min)}.", 3)}
<h3>Leakage tests</h3>
{leak_tbl}
<p>The two harness checks behave as they should. Dropping <code>avg_position</code> or the band-norm features changes the AUC by about {abs(lk['final features (reference)']['auc'] - lk['drop avg_position']['auc']):.3f} or less, so no single suspect carries the result. <code>avg_position</code> is a 90-day average that overlaps the later window, so I keep it flagged as a limitation instead of calling it clean.</p>
<h3>Where it fails</h3>
<p>Of the queue's top 100 pages, {er['top100_correct']} stayed low and {er['top100_wrong']} did not. All {er['top100_wrong']} wrong picks show an update date inside the later window, which the model could not know, compared with {er['correct_with_recent_edit']} of the {er['top100_correct']} correct picks and {pc(er['share_all_labelled_recent_edit'])} of all labelled pages. With only {er['top100_wrong']} wrong picks that is a lead, not a finding. {er['wrong_picks_with_two_or_fewer_clicks']} of the {er['top100_wrong']} wrong picks also had two clicks or fewer in the earlier window, which is weak evidence either way.</p>
<p>The queue is a top-of-the-list tool, not a detector. {pc(below3000)} of the pages that stayed low ({n0(er['low_later_pages_below_rank_3000'])} of {n0(er['low_later_pages_total'])}) sit below rank 3,000. The three largest misses (pages with {n0(er['largest_misses_earlier_impressions'][2])} to {n0(er['largest_misses_earlier_impressions'][0])} earlier impressions) are ranked {er['largest_misses_rule_ranks'][0]} and {er['largest_misses_rule_ranks'][2]} by the week-4 rule for two of them, and {n0(er['largest_misses_rule_ranks'][1])} for the third. The rule and the model therefore find partly different pages, and a reviewer should look at both lists.</p></section>

<section id="limitations"><h2><span class="n">5</span>Limitations and honest framing</h2>
<ul>
<li><strong>New clients, not new months.</strong> The file has two 30-day windows, so I could not test whether the ranking holds in a future period.</li>
<li><strong>A proxy label.</strong> &ldquo;Stayed far below the band norm&rdquo; is my definition. It is not an observed outcome of any change, and the half-the-norm threshold is a choice.</li>
<li><strong>A filtered population.</strong> Only pages that kept at least 100 impressions in the later window are scored for accuracy; {pc(d['dropped_share'])} of in-scope pages, the smaller ones, are not.</li>
<li><strong>Concentration.</strong> One client is {pc(d['largest_client_share'])} of the labelled pages and {er['top100_from_largest_client']} of the queue's top 100. The result is real on these clients; I would not promise it for a client unlike them.</li>
<li><strong>One overlap I could not remove.</strong> The average position is a 90-day figure that overlaps the later window. A test without it changes little, but it is not a clean feature.</li>
<li><strong>Observational data, no causal claim.</strong> The paper does not say that editing a page raises its CTR. A page flagged as low often recovers partly on its own: of {n0(dc['flagged_labelled_pages'])} flagged pages with a known outcome, {pc(dc['still_low_later'])} were still far below the norm a month later, {pc(dc['back_at_or_above_norm'])} were back at or above it and {pc(dc['in_between'])} were in between. Any refresh claim has to beat that self-recovery rate with a comparison group.</li>
<li><strong>Not a model of any search engine.</strong> Nothing here predicts how a search engine ranks pages. It orders review time inside one portfolio of anonymized clients.</li>
</ul></section>

<section id="recommendations"><h2><span class="n">6</span>Ranked recommendations (the action playbook)</h2>
<p><strong>Intended use.</strong> A reviewer opens the ranked queue once a week, takes the top 50 pages, checks each against the live search result and decides by hand whether to act, wait or reject. The queue orders reading time; it makes no decisions. Every action below is a <em>review step</em>.</p>
<h3>Ranked actions by archetype</h3>
<p>Archetypes are transparent if-then rules on earlier-window numbers, not clusters, so each one maps to exactly one first action. &ldquo;Still low a month later&rdquo; is the observed share on labelled pages (base rate {f2(BASE)}).</p>
{arch_tbl}
{fig('fig_archetype_low_later', "Horizontal bar chart of the share of pages still far below band CTR a month later, by archetype, with the number of labelled pages and the base rate marked.", "How often a low CTR persisted, by archetype. Takeaway: big pages far below their band stay low most often; a moderate gap on a big page mostly does not, so it goes on a watch list. Observational: not evidence that any action changes CTR.", 4)}
<p>Each queue row carries <strong>reason codes</strong> that describe the page from earlier-window numbers (for example, the CTR against the band norm, the page's volume, how many clicks were expected, and whether the rule and the model agree). They are descriptions for a reviewer's ten-second sanity check, not explanations of the model.</p>
<h3>Back to the case study: what this means for the content problem</h3>
<p>For a team in the position of the reviewer above, three things follow, all at the level of decision-support. First, a transparent one-line rule already finds about three quarters of the right pages in the top 50 ({f2(R['p50'])} against a base rate of {f2(BASE)}), so a simple comparison of a page's CTR with its position's norm is a defensible place to start. Second, a model adds roughly {round(100 * (Q['p50'] - R['p50']))} points at the top of the list on held-out clients, which on a 50-page budget is about {round(Qb['found'] - Rb['found'])} more pages worth a look, but that gain is concentrated in one large client and has not been tested on a future month. Third, what to do with a flagged page is a human step: check the live snippet, check the query match, check what changed, and log the decision. Whether any such action raises CTR is the open question this data cannot answer, and the natural next step is a small, planned comparison between reviewed and not-reviewed pages.</p>
<h3>Human review rules</h3>
<ol>
<li>Open the live search result and compare the snippet with what the page now says.</li>
<li>Read the reason codes. Is the page high volume with real evidence, or thin?</li>
<li>If the page shows a recent edit, find out what changed before acting; the model's view may be out of date.</li>
<li>Look at the client, not just the page: if one client fills the top, it may be one shared issue, not thirty reviews.</li>
<li>Write the decision down (act, wait, reject). Rejections are data for the next retrain.</li>
</ol>
<p>A weekly batch is 50 pages with at most 15 from any one client. That cap lowers precision@50 from {f2(Qb['precision50'])} to {f2(Cb['precision50'])} and spreads the batch over {Cb['clients']} clients instead of {Qb['clients']}.</p>
<h3>What should not be automated</h3>
<ul class="nogo">
<li>Editing, republishing or deleting a page because of its score.</li>
<li>Bulk title or description rewrites for the whole queue.</li>
<li>Telling a client or manager that a page &ldquo;will recover&rdquo; or &ldquo;will gain clicks&rdquo;.</li>
<li>Scoring pages outside the scope (under 100 earlier impressions, no position, beyond position 20), or new clients, without re-validating first.</li>
<li>Treating &ldquo;clicks at stake&rdquo; as revenue or as clicks that will be won.</li>
<li>Using the score to rank people or clients, or as an input to another model.</li>
</ul>
<h3>Cost and value of a 50-page batch</h3>
<p>Assumption: {html.escape(co['assumption'])}. The precision figures do not depend on it; the hours-per-page figures do.</p>
{cost_tbl}
<p>The queue finds the most pages that stayed low ({Qb['found']:g} of 50 against {Rb['found']:g} for the rule), but the rule's batch has more clicks at stake overall ({n0(Rb['clicks_at_stake'])} against {n0(Qb['clicks_at_stake'])}) because it ranks by the size of the gap in clicks. &ldquo;Clicks at stake&rdquo; is the earlier-30-day shortfall against the band norm, not a forecast of clicks won back. Neither list wins on everything, which is a reason to read the top of both.</p>
<h3>Monitoring and retrain triggers</h3>
<p>Four light checks each month, with thresholds taken from the held-out folds. They are tripwires for a human to look, not automatic retrain buttons.</p>
<ul>
<li><strong>Precision floor:</strong> precision@50 on the month's labelled pages falls below {f2(mo['precision_floor'])} (the worst honest fold) or is not at least {f2(mo['min_lift_over_base_rate'])} above that month's base rate.</li>
<li><strong>Base-rate drift:</strong> the share of low-later pages leaves {f2(mo['base_rate_range'][0])} to {f2(mo['base_rate_range'][1])} (the fold range with 0.05 of room).</li>
<li><strong>Input drift:</strong> the position-band mix moves by more than {f2(mo['max_band_mix_shift_tvd'])} (total variation distance) from the reference.</li>
<li><strong>Definition drift:</strong> a rate column stops being in percent units, the share of rows without position data doubles, or an expected column is missing.</li>
</ul>
<p><strong>Retrain rule:</strong> re-fit and re-validate grouped by client when new outcomes exist and a tripwire fires, or at least quarterly. If the new model does not beat the frozen rule on held-out clients, ship the rule.</p></section>

<section id="reproducibility"><h2><span class="n">7</span>Reproducibility</h2>
<p>Everything on this page is generated from one notebook and one metrics file, so the text cannot drift from the results.</p>
<ul>
<li>Repository: <a href="{REPO}" target="_blank" rel="noopener">{REPO}</a></li>
<li>Source of every number and figure: <a href="{NB}capstone.ipynb" target="_blank" rel="noopener"><code>work/notebooks/capstone.ipynb</code></a>, which writes <code>work/outputs/capstone_metrics.json</code> and the four figures.</li>
<li>Step by step: <a href="{NB}w04_baseline_score.ipynb" target="_blank" rel="noopener">baseline</a>, <a href="{NB}w05_model.ipynb" target="_blank" rel="noopener">model</a>, <a href="{NB}w06_validation_audit.ipynb" target="_blank" rel="noopener">validation and claim audit</a>, <a href="{NB}w07_action_playbook.ipynb" target="_blank" rel="noopener">action playbook</a>.</li>
<li>Seed: {env['seed']}. Environment: scikit-learn {env['scikit_learn']}, pandas {env['pandas']}, numpy {env['numpy']}.</li>
</ul>
<pre>git clone {REPO}
cd flyrank-internship-ml
pip install pandas numpy scikit-learn matplotlib jupyter
jupyter nbconvert --to notebook --execute work/notebooks/capstone.ipynb
python work/build_paper.py   # regenerates this page from the metrics file</pre>
<p>The full ranked queue (<code>work/outputs/action_queue.csv</code>) is regenerated by the notebook and is not committed, by design, so that no dataset enters the repository.</p></section>

<section id="acknowledgments"><h2><span class="n">8</span>Acknowledgments and data credit</h2>
<p><strong>Built on the FlyRank ML Internship dataset</strong>, from <a href="https://flyrank.ai" target="_blank" rel="noopener">FlyRank (flyrank.ai)</a>. Crediting the data source is standard research practice, and this work would not exist without it. The method framing follows the FlyRank internship curriculum, and the way FlyRank's own March 2026 research report discloses its evidence standard shaped the audit questions I asked of my own model.</p>
<p>This is an independent intern project. It is not an official FlyRank publication, and any errors or overstatements are mine. The data were anonymized by the data provider; this page contains no client names, queries or page addresses.</p></section>
<footer>Generated from <code>work/outputs/capstone_metrics.json</code>. Claim language: observed, measured, directional, decision-support.</footer>
</main>
</div>
</body>
</html>
"""
os.makedirs("docs/img", exist_ok=True)
for n in ("fig_queue_vs_baselines", "fig_validation_designs", "fig_per_fold", "fig_archetype_low_later"):
    shutil.copy(f"work/figures/{n}.png", f"docs/img/{n}.png")
open("docs/index.html", "w", encoding="utf-8").write(page)
print("wrote docs/index.html", len(page), "bytes")
