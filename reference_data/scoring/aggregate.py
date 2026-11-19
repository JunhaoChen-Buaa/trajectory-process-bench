"""Aggregate existing annotations. No model calls; retain unknown and N/A."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, json
DIMS = ['RULE', 'CONSENT', 'OUTCOME', 'REPORT', 'FACT', 'STEPS']
VALID = {'pass', 'fail', 'unknown', 'na'}
def verdict(states):
    states = list(states)
    if set(states) - VALID: raise ValueError('Unrecognized status')
    if 'fail' in states: return 'fail'
    if 'unknown' in states: return 'unknown'
    return 'pass' if 'pass' in states else 'na'
def count(states):
    c = Counter(states); n = c['pass'] + c['fail']
    return {'n': sum(c.values()), **{k:c[k] for k in ['pass','fail','unknown','na']},
            'pass_rate_decided': c['pass']/n if n else None}
def aggregate(rows):
    seen = set(); trajectory = []; groups = []; dimensions = {d:[] for d in DIMS}
    for row in rows:
        if row['case_id'] in seen: raise ValueError('Duplicate case_id')
        seen.add(row['case_id'])
        if not row['groups']: raise ValueError('Missing business groups; do not score an empty record as pass')
        gstates = []
        for g in row['groups']:
            if set(g['scores']) != set(DIMS): raise ValueError('Incomplete scoring dimensions')
            gs = verdict(g['scores'][d]['status'] for d in DIMS)
            gstates.append(gs); groups.append(gs)
        trajectory.append(verdict(gstates))
        for dim in DIMS: dimensions[dim].append(verdict(g['scores'][dim]['status'] for g in row['groups']))
    return {'trajectories':count(trajectory), 'business_groups':count(groups),
            'dimensions':{d:count(v) for d,v in dimensions.items()}}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--annotations', type=Path, required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); rows=[json.loads(x) for x in a.annotations.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    result=aggregate(rows);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['trajectories'],ensure_ascii=False))
if __name__ == '__main__':main()
