"""Validate the active manuscript against canonical evidence and its PDF render."""
from pathlib import Path
import re
import json
import hashlib
import pdfplumber

root = Path(__file__).resolve().parents[1]
source = root / 'paper/ARIA_revised_journal_draft.tex'
tex = source.read_text(encoding='utf-8')
citations = {key for group in re.findall(r'\\cite\{([^}]+)\}', tex) for key in group.split(',')}
bibliography = re.findall(r'\\bibitem\{([^}]+)\}', tex)
assert len(bibliography) == len(set(bibliography)) == 13
assert citations == set(bibliography)
references = set(re.findall(r'\\ref\{([^}]+)\}', tex))
labels = re.findall(r'\\label\{([^}]+)\}', tex)
assert references <= set(labels)
assert len(labels) == len(set(labels))
counts = {name: len(re.findall(r'\\begin\{' + name + r'\}', tex)) for name in ['figure', 'table', 'equation']}
assert counts == {'figure': 5, 'table': 9, 'equation': 6}, counts
assert '@@' not in tex and 'TODO' not in tex

summary_path = root / 'paper/generated_sota/policy_comparison_summary.json'
summary = json.loads(summary_path.read_text())
assert summary['development_only'] and not summary['locked_test_used']
assert len(summary['rows']) == 32
names = {'aria_iql_v7':'ARIA IQL', 'behavior_cloning':'Behavior cloning',
         'discrete_cql':'Discrete CQL', 'decision_transformer':'DT-inspired',
         'random_valid':'Random valid', 'fixed_script':'Fixed script',
         'rule_based_adaptive':'Rule adaptive', 'uncertainty_greedy':'Uncertainty greedy'}
conditions = ['base', 'overlap', 'low_confidence', 'positive_shift']
rows = {(r['condition'], r['policy']): r for r in summary['rows']}
tables = {}
for block in re.findall(r'\\begin\{table\}.*?\\end\{table\}', tex, re.S):
    label = re.search(r'\\label\{([^}]+)\}', block).group(1)
    tables[label] = block
for policy, name in names.items():
    accuracy = [f"{rows[c,policy]['accuracy']:.4f}" for c in conditions]
    efficiency = [f"{rows[c,policy]['turns_mean']:.2f} / {rows[c,policy]['skills_covered_mean']:.2f}" for c in conditions]
    uncertainty = [f"{rows[c,policy]['reward_mean']:.2f} [{rows[c,policy]['reward_ci_lower']:.2f}, {rows[c,policy]['reward_ci_upper']:.2f}]" for c in conditions]
    for label, values in [('accuracy', accuracy), ('efficiency', efficiency), ('uncertainty', uncertainty)]:
        assert ' & '.join([name] + values) + r' \\' in tables['tab:' + label], (policy, label)
    if policy in list(names)[:4]:
        values = [f"{rows[c,policy]['reward_mean']:.2f} / {rows[c,policy]['information_gain_mean']:.2f}" for c in conditions]
        assert ' & '.join([name] + values) + r' \\' in tables['tab:returns']
assert sum(rows['base',p]['accuracy'] == 1.0 for p in names) == 7

locked_path = root / 'data/synthetic/v3/production-grounding-v10/derived-calibration-v7/release/locked_test_evaluation_v1.json'
locked = json.loads(locked_path.read_text())
assert not locked['evaluates_learned_policy']
metrics = {'Accuracy':'accuracy', 'Macro-F1':'macro_f1', 'Balanced accuracy':'balanced_accuracy',
           'Minimum class recall':'minimum_class_recall', 'Expected calibration error':'expected_calibration_error',
           'Brier score':'brier_score', 'Ordinal mean absolute error':'ordinal_mae'}
for name, key in metrics.items():
    expected = f"{name} & {locked['v6_v7_paired_comparison']['baseline_metrics'][key]:.4f} & {locked['metrics'][key]:.4f}"
    assert expected in tables['tab:locked'], expected

pdfpath = root / 'output/pdf/ARIA_revised_journal_draft.pdf'
with pdfplumber.open(pdfpath) as pdf:
    pages = len(pdf.pages)
    assert 12 <= pages <= 22, pages
    extracted = '\n'.join(p.extract_text() or '' for p in pdf.pages)
    assert '\\' not in extracted
    assert not any(c['x0'] < 40 or c['x1'] > p.width - 40 for p in pdf.pages for c in p.chars)
    assert not any(c['top'] < 35 or c['bottom'] > p.height - 20 for p in pdf.pages for c in p.chars)
    for label, block in tables.items():
        for number in re.findall(r'(?<![A-Za-z])\d+\.\d+', block):
            assert number in extracted, (label, number)
    for expected in ['DT-inspired', 'terminal outcome adjustment', '0.4618', '1,405', 'References']:
        assert expected in extracted, expected

digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = {'status':'PASS_WITH_DOCUMENTED_LIMITS', 'pdf_pages':pages, 'citations':len(bibliography),
          'figures':counts['figure'], 'tables':counts['table'], 'equations':counts['equation'],
          'source_sha256':digest(source), 'pdf_sha256':digest(pdfpath),
          'canonical_summary_sha256':digest(summary_path), 'locked_report_sha256':digest(locked_path),
          'checks':['All 32 policy rows matched canonical results at displayed precision',
                    'All seven locked belief metrics matched source report',
                    'Citation keys and cross-references resolve', 'PDF text and margins checked'],
          'limits':['Native TeX compilation unavailable; PDF uses ReportLab fallback',
                    'ScienceDirect reference format and length not verified',
                    'Historical baselines trace to summaries, not recovered prediction artifacts',
                    'One comparator training seed; canonical paired episode traces unavailable locally',
                    'Structural checks do not establish statistical validity or submission readiness']}
(root / 'paper/ARIA_revision_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
