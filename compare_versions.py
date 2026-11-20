"""Post-hoc exploratory comparison: old and revised scorers on final-v2 fixtures."""
import math
import marrow


def baseline(profile, session):
    obs = marrow.features(session)
    differences = [min(10.0, abs(obs[k]-p['median']) / p['scale'])
                   for k, p in profile['model'].items() if k in obs]
    if len(differences) < marrow.MIN_FEATURES:
        return False
    n = math.ceil(len(differences)*.6)
    value = sum(sorted(differences, reverse=True)[:n]) / n
    return value <= 1.0  # v1 original threshold, no tempo normalization


counts = {version:{r:[0,0] for r in ('owner','tired','stranger','mimic','swap')}
          for version in ('v1', 'v2')}
for style in ('typed','inserted','onscreen'):
    for identity in range(3000,3006):
        enroll, tests = marrow.fixture(style, identity, 'final-v2')
        profile = marrow.train(enroll)
        wrong, _ = marrow.fixture(style, identity+101, 'swap-profile-v1')
        wrong_profile = marrow.train(wrong)
        for role, session in tests:
            results = {'v1':baseline(profile, session),
                       'v2':marrow.score(profile, session)['decision'] == 'admit'}
            for version, admitted in results.items():
                bucket = counts[version][role]; bucket[0] += int(admitted); bucket[1] += 1
            if role == 'owner':
                for version, admitted in {'v1':baseline(wrong_profile,session),
                    'v2':marrow.score(wrong_profile,session)['decision'] == 'admit'}.items():
                    bucket = counts[version]['swap']; bucket[0] += int(admitted); bucket[1] += 1
print(counts)
