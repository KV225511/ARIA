"""Build source-derived appendix and provenance records for the October revision."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
paper = ROOT / 'paper'
texpath = paper / 'ARIA_revised_journal_draft.tex'
text = texpath.read_text(encoding='utf-8')
report = (ROOT / 'ARIA_PROJECT_CONTEXT_AND_RESULTS.md').read_text(encoding='utf-8')
rows = []
for line in report.splitlines():
    if line.startswith('|') and re.search(r'\|\s*(?:\*\*)?0\.\d+', line):
        cols = [s.strip().replace('**', '').replace('`', '') for s in line.strip('|').split('|')]
        if len(cols) == 7 and cols[-1] == 'Success':
            rows.append(cols)
assert len(rows) == 25, len(rows)
table = [r'\small', r'\begin{longtable}{p{26mm}p{67mm}rrr}',
         r'\caption{Repository-reported rapid-run benchmark matrix. Unequal budgets and unrecovered split details prevent a controlled SOTA ranking.}\label{tab:rapid}\\',
         r'\toprule Domain & Architecture & Accuracy & Macro-F1 & Time (s) \\ \midrule', r'\endfirsthead',
         r'\toprule Domain & Architecture & Accuracy & Macro-F1 & Time (s) \\ \midrule',r'\endhead',
         r'\bottomrule\endfoot']
domains = {'Model 1: Video Emotion':'Video', 'Model 2: Audio CogLoad':'Audio',
           'Model 3: Text Semantic':'Text', 'Model 4: Multimodal Fusion':'Fusion',
           'Model 5: Cross-Modal Deception':'Annotated deception'}
for domain, model, script, acc, f1, elapsed, status in rows:
    table.append(f'{domains[domain]} & {model} & {acc} & {f1} & {elapsed} '+r'\\')
table += [r'\end{longtable}',r'\normalsize']
assert '% RAPID_TABLE_INSERT' in text
text = text.replace('% RAPID_TABLE_INSERT', '\n'.join(table))
texpath.write_text(text, encoding='utf-8')

release = ROOT / 'data/synthetic/v3/production-grounding-v10/derived-calibration-v7'
d = json.loads((release/'release/locked_test_evaluation_v1.json').read_text())
metrics = d['metrics']
base = d['v6_v7_paired_comparison']['baseline_metrics']
assert sum(sum(row) for row in metrics['confusion_matrix']) == 91
assert sum(metrics['confusion_matrix'][i][i] for i in range(3)) == 83
assert sum(base['confusion_matrix'][i][i] for i in range(3)) == 72
assert d['evaluates_learned_policy'] is False
assert f"{metrics['brier_score'] - base['brier_score']:.4f}" == '-0.1055'

paths = ['ARIA_updated.md','ARIA_PROJECT_CONTEXT_AND_RESULTS.md',
         'modules/module_07_rl/state_builder.py','modules/module_07_rl/rl_spec.py',
         'modules/module_07_rl/train.py','modules/module_07_rl/iql_policy.py',
         'modules/module_06_belief/belief_state.py','modules/module_07_rl/offline_policy_evaluation.py','app.py',
         str((release/'release/locked_test_evaluation_v1.json').relative_to(ROOT)),
         str((release/'audits/offline_policy_evaluation_v1.json').relative_to(ROOT))]
hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
(paper/'ARIA_revision_source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
state = json.loads((ROOT/'.paper/manuscript_state.json').read_text(encoding='utf-8'))
state['project']['active_release'] = '2026-10-02-journal-draft'
state['project']['lifecycle_stage'] = 'Baseline consolidation and journal manuscript rewrite; reference format pending'
state['project']['target_venue'] = 'Unspecified; user-supplied ScienceDirect style reference remains inaccessible'
for item in state['artifacts']:
    item['status'] = 'REFERENCE_ONLY'
    item['required_for_release'] = False
state['artifacts'].append({'id':'october-tex','path':'paper/ARIA_revised_journal_draft.tex','role':'main_manuscript','status':'ACTIVE','required_for_release':True})
state['authority_sources'].extend([
    {'id':'component-record','kind':'author_repository_report','path_or_reference':'ARIA_updated.md','scope':['historical baseline results'],'status':'PROBABLE','last_verified':'2026-10-02'},
    {'id':'rapid-record','kind':'author_repository_report','path_or_reference':'ARIA_PROJECT_CONTEXT_AND_RESULTS.md','scope':['rapid-run appendix'],'status':'PROBABLE','last_verified':'2026-10-02'}])
state['release'] = {'status':'DRAFT','candidate_artifacts':['october-tex'],'visual_check':'PENDING','checked_hashes':{}}
state['revision_notes'] = {
    'authority':'User explicitly requested current repository evidence and directed recovery of existing baseline results.',
    'source_status':'Current code and JSON inspected. Historical baseline metrics verified against reports, not raw predictions.',
    'style_reference':'ScienceDirect S2590005626003218 blocked by 403/CAPTCHA. Exact style and length match remains pending.',
    'open_issues':['Original component logs and combined CSV absent from inspected checkout','Conflicting historical model labels and metrics','No fresh current-IQL comparative rollouts','No human validation','Reference-paper PDF needed for exact format/length match'],
    'scientific_changes':['Restored component and historical simulation baselines','Distinguished historical reward heuristic from IQL','Corrected transition-level OPE interpretation','Corrected state groups and semantic-only competency likelihood'],
    'bibliography':'Focused verified set of nine sources replaces broad 70-source inventory for this draft; original preserved.'}
(ROOT/'.paper/manuscript_state.json').write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'appendix_rows':len(rows),'belief_n':91,'current_correct':83,'baseline_correct':72,'source_hashes':len(hashes)}))
