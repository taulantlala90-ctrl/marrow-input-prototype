"""Development-only calibration; never called by the final evaluation."""
import marrow

thresholds = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]
data = []
ids = list(range(6)) + list(range(1000,1006)) + list(range(2000,2006))
for style in ('typed','inserted','onscreen'):
    for identity in ids:
        enroll, tests = marrow.fixture(style, identity, 'development-v2')
        profile = marrow.train(enroll)
        for role, session in tests:
            result = marrow.score(profile, session)
            data.append((role,result['score']))
        wrong, _ = marrow.fixture(style, identity+101, 'swap-profile-v1')
        wrong_profile = marrow.train(wrong)
        for role, session in tests:
            if role == 'owner':
                data.append(('swap',marrow.score(wrong_profile,session)['score']))
for threshold in thresholds:
    groups={}
    for role,value in data:
        ok = value is not None and value <= threshold
        bucket = groups.setdefault(role,[0,0]); bucket[0] += int(ok); bucket[1] += 1
    print(threshold, groups)
