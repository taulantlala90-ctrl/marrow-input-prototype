#!/usr/bin/env python3
"""Offline synthetic behavioral-authentication research prototype, not a security product."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

SCHEMA = 1
# Selected on development identities 0–5, 1000–1005 and 2000–2005, then
# fixed before revised final identities 3000–3005 were generated/evaluated.
THRESHOLD = 1.1
MIN_FEATURES = 4
FLOORS = {
    'speed': 0.035, 'straightness': 0.06, 'turn': 0.11,
    'click_gap': 0.10, 'text_gap': 0.10, 'text_size': 0.08,
    'correction': 0.09, 'correction_gap': 0.12,
}


def med(values):
    return statistics.median(values) if values else None


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')


def features(session):
    """Extract only from the documented event fields, never identity or labels."""
    if session.get('version') != SCHEMA or not isinstance(session.get('events'), list):
        raise ValueError('expected version 1 and an events array')
    events = session['events']
    if len(events) > 100000:
        raise ValueError('session exceeds 100000 events')
    ordered = []
    for e in events:
        if not isinstance(e, dict) or not isinstance(e.get('t_ms'), (int, float)):
            raise ValueError('events need numeric t_ms')
        t = e['t_ms']
        if isinstance(t, bool) or not math.isfinite(t) or t < 0:
            raise ValueError('timestamps must be finite and nonnegative')
        kind = e.get('type')
        if kind not in ('pointer', 'click', 'text'):
            raise ValueError('unknown event type')
        if kind == 'pointer':
            for coord in ('x', 'y'):
                value = e.get(coord)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('pointer coordinates must be finite fractions in [0,1]')
        if kind == 'text':
            if e.get('source') not in ('typed', 'onscreen', 'inserted') or e.get('action') not in ('add', 'delete'):
                raise ValueError('text events need source and add/delete action')
            if isinstance(e.get('count'), bool) or not isinstance(e.get('count'), int) or not 1 <= e['count'] <= 10000:
                raise ValueError('text event count must be a positive integer')
        ordered.append(e)
    ordered.sort(key=lambda e: e['t_ms'])
    pts = [e for e in ordered if e['type'] == 'pointer']
    clicks = [e['t_ms'] for e in ordered if e['type'] == 'click']
    speed, angle, vectors = [], [], []
    for a, b in zip(pts, pts[1:]):
        dt = (b['t_ms'] - a['t_ms']) / 1000
        dx, dy = b['x'] - a['x'], b['y'] - a['y']
        distance = math.hypot(dx, dy)
        if 0.005 <= dt <= 10 and distance > 0:
            speed.append(distance / dt)
            vectors.append((dx, dy, distance))
    for a, b in zip(vectors, vectors[1:]):
        dot = max(-1.0, min(1.0, (a[0]*b[0] + a[1]*b[1]) / (a[2]*b[2])))
        angle.append(math.acos(dot))
    path = sum(v[2] for v in vectors)
    straight = (math.hypot(pts[-1]['x'] - pts[0]['x'], pts[-1]['y'] - pts[0]['y']) / path) if len(pts) > 1 and path else None
    out = {}
    for name, value in [('speed', med(speed)), ('turn', med(angle)), ('straightness', straight),
                        ('click_gap', med([(b-a)/1000 for a,b in zip(clicks, clicks[1:]) if b > a]))]:
        if value is not None:
            out[name] = value
    # Model each text source separately; absent channels do not imply a zero value.
    for source in ('typed', 'onscreen', 'inserted'):
        text = [e for e in ordered if e['type'] == 'text' and e['source'] == source]
        adds = [e for e in text if e['action'] == 'add']
        deletes = [e for e in text if e['action'] == 'delete']
        if len(adds) >= 2:
            gaps = [(b['t_ms']-a['t_ms'])/1000 for a,b in zip(adds, adds[1:]) if b['t_ms'] > a['t_ms']]
            if gaps:
                out[source + ':text_gap'] = med(gaps)
            out[source + ':text_size'] = med([e['count'] for e in adds])
            out[source + ':correction'] = len(deletes) / len(adds)
        cgaps = [(d['t_ms']-a['t_ms'])/1000 for a,d in zip(text, text[1:]) if a['action'] == 'add' and d['action'] == 'delete' and d['t_ms'] >= a['t_ms']]
        if cgaps:
            out[source + ':correction_gap'] = med(cgaps)
    return out


def train(sessions):
    if not sessions:
        raise ValueError('enrollment is empty')
    samples = [features(s) for s in sessions]
    keys = sorted(set().union(*(s.keys() for s in samples)))
    result = {}
    for key in keys:
        values = [s[key] for s in samples if key in s]
        if len(values) < 2:
            continue
        center = med(values)
        mad = med([abs(v-center) for v in values])
        floor = FLOORS[key.split(':')[-1]]
        scale = max(1.4826 * mad, floor, abs(center) * 0.05)
        result[key] = {'median': center, 'scale': scale, 'samples': len(values)}
    if len(result) < MIN_FEATURES:
        raise ValueError('at least four usable features across two enrollment sessions are required')
    return {'version': 1, 'threshold': THRESHOLD, 'model': result}


def score(profile, session):
    obs = features(session)
    # A coherent global speed change can reflect fatigue or deliberate pacing.
    # Normalize only with independent click AND text cadence observations, and
    # only within a bounded factor. This can also help an imitator: report FAR.
    tempo_pairs = []
    for key in ('click_gap', 'typed:text_gap', 'onscreen:text_gap', 'inserted:text_gap'):
        if key in obs and key in profile['model'] and profile['model'][key]['median'] > 0:
            tempo_pairs.append((key, obs[key] / profile['model'][key]['median']))
    if len(tempo_pairs) >= 2 and any(key == 'click_gap' for key, _ in tempo_pairs):
        tempo = max(.7, min(1.4, med([ratio for _, ratio in tempo_pairs])))
        for key, _ in tempo_pairs:
            obs[key] /= tempo
    diffs = []
    for key, p in profile['model'].items():
        if key in obs:
            diffs.append(min(10.0, abs(obs[key] - p['median']) / p['scale']))
    if len(diffs) < MIN_FEATURES:
        return {'decision': 'insufficient_evidence', 'score': None, 'shared_features': len(diffs)}
    value = sum(sorted(diffs, reverse=True)[:max(1, math.ceil(len(diffs)*0.6))]) / max(1, math.ceil(len(diffs)*0.6))
    return {'decision': 'admit' if value <= profile['threshold'] else 'reject', 'score': round(value, 6), 'shared_features': len(diffs)}


def parameters(style, identity):
    r = random.Random(f'person:{style}:{identity}')
    return {'speed': r.uniform(.035, .16), 'turn': r.uniform(.15, 1.1),
            'gap': r.uniform(.25, 1.8), 'click': r.uniform(1.0, 3.5),
            'length': r.uniform(1.5, 8) if style == 'inserted' else r.uniform(1, 2.5),
            'correction': r.uniform(.02, .4), 'correction_delay': r.uniform(.1, 1.3)}


def synthetic_session(style, identity, seed, role='owner', mimic_of=None):
    """Generate events with deliberate within-person variance; roles never reach scoring."""
    r = random.Random(f'session:{style}:{identity}:{seed}:{role}')
    own = parameters(style, identity)
    if mimic_of is not None:
        target = parameters(style, mimic_of)
        # A mimic copies timing and text chunk means only imperfectly; other dynamics differ.
        own['gap'] = .6*target['gap'] + .4*own['gap']
        own['length'] = .6*target['length'] + .4*own['length']
    fatigue = 1.12 if role == 'tired' else 1.0
    events = []
    x, y, heading, t = .48, .48, r.uniform(0, 2*math.pi), 0
    for i in range(160):
        dt = r.uniform(80, 180)*fatigue
        t += dt
        heading += r.gauss(0, own['turn']*.65)
        stride = own['speed'] * dt/1000 * r.uniform(.8, 1.2)
        x = min(.96, max(.04, x + math.cos(heading)*stride))
        y = min(.96, max(.04, y + math.sin(heading)*stride))
        events.append({'type':'pointer', 't_ms':round(t), 'x':round(x,5), 'y':round(y,5)})
    t = 0
    for i in range(18):
        t += 1000 * own['click'] * fatigue * r.uniform(.8,1.2)
        events.append({'type':'click', 't_ms':round(t), 'button':0})
    t = 0
    for i in range(55):
        t += 1000 * own['gap'] * fatigue * r.uniform(.8,1.2)
        events.append({'type':'text', 't_ms':round(t), 'source':style,
                       'action':'add', 'count':max(1, round(r.gauss(own['length'], .6)))})
        if r.random() < own['correction']:
            events.append({'type':'text', 't_ms':round(t + own['correction_delay']*1000),
                           'source':style, 'action':'delete', 'count':1})
    events.sort(key=lambda e: e['t_ms'])
    return {'version':1, 'session_id':f'{style}-{identity}-{seed}-{role}', 'events':events}


def fixture(style, subject, seed):
    enroll = [synthetic_session(style, subject, f'{seed}-enroll-{n}') for n in range(8)]
    tests = [
        ('owner', synthetic_session(style, subject, f'{seed}-owner-{n}')) for n in range(5)
    ] + [
        ('tired', synthetic_session(style, subject, f'{seed}-tired-{n}', role='tired')) for n in range(5)
    ] + [
        ('stranger', synthetic_session(style, subject+101+n, f'{seed}-stranger-{n}')) for n in range(5)
    ] + [
        ('mimic', synthetic_session(style, subject+201+n, f'{seed}-mimic-{n}', role='mimic', mimic_of=subject)) for n in range(5)
    ]
    return enroll, tests


def report_demo(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    result = {'threshold':THRESHOLD, 'styles':{}, 'note':'All data synthetic; no real-world accuracy, fairness or security is established.'}
    swap_fail = swap_total = 0
    for style in ('typed','inserted','onscreen'):
        counts = {r:{'accepted':0,'total':0,'insufficient':0} for r in ('owner','tired','stranger','mimic')}
        for identity in range(3000, 3006):
            enroll, tests = fixture(style, identity, 'final-v2')
            profile = train(enroll)
            case_dir = outdir / 'cases' / f'{style}-{identity}'
            case_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(case_dir/'enrollment.jsonl', enroll)
            write_jsonl(case_dir/'test.jsonl', [session for _, session in tests])
            (case_dir/'labels.json').write_text(json.dumps([{'session_id':s['session_id'], 'role':role} for role,s in tests], indent=2)+'\n',encoding='utf-8')
            for role, session in tests:
                result_row = score(profile, session)
                counts[role]['total'] += 1
                counts[role]['accepted'] += result_row['decision'] == 'admit'
                counts[role]['insufficient'] += result_row['decision'] == 'insufficient_evidence'
            stranger_enroll, _ = fixture(style, identity+101, 'swap-profile-v1')
            wrong = train(stranger_enroll)
            for role, session in tests:
                if role == 'owner':
                    swap_total += 1
                    swap_fail += score(wrong, session)['decision'] == 'admit'
        result['styles'][style] = counts
    result['swap_test'] = {'owner_admitted_by_other_profile':swap_fail,'trials':swap_total,'passed':swap_fail == 0}
    all_roles = {role: {'accepted': sum(v[role]['accepted'] for v in result['styles'].values()),
                        'total': sum(v[role]['total'] for v in result['styles'].values())}
                 for role in ('owner','tired','stranger','mimic')}
    result['summary'] = {
        'false_accept': {'accepted':all_roles['stranger']['accepted']+all_roles['mimic']['accepted'],
                         'trials':all_roles['stranger']['total']+all_roles['mimic']['total']},
        'false_reject': {'rejected':all_roles['owner']['total']+all_roles['tired']['total']-all_roles['owner']['accepted']-all_roles['tired']['accepted'],
                         'trials':all_roles['owner']['total']+all_roles['tired']['total']},
    }
    (outdir/'report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if swap_fail:
        raise SystemExit('swap test failed; see report')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    demo = sub.add_parser('demo'); demo.add_argument('--out',default='marrow-demo')
    training = sub.add_parser('train'); training.add_argument('--input',required=True); training.add_argument('--output',required=True)
    testing = sub.add_parser('score'); testing.add_argument('--profile',required=True); testing.add_argument('--input',required=True); testing.add_argument('--output',required=True)
    args = p.parse_args()
    if args.cmd == 'demo':
        report_demo(Path(args.out))
    elif args.cmd == 'train':
        profile = train(read_jsonl(args.input)); Path(args.output).write_text(json.dumps(profile, indent=2)+'\n',encoding='utf-8')
    else:
        profile = json.loads(Path(args.profile).read_text(encoding='utf-8'))
        rows = [{'session_id':s.get('session_id'), **score(profile,s)} for s in read_jsonl(args.input)]
        write_jsonl(args.output, rows)


if __name__ == '__main__':
    main()
