import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import marrow


class AuthenticationPrototypeTests(unittest.TestCase):
    def setUp(self):
        self.enrollment, self.tests = marrow.fixture('inserted', 2000, 'final-v1')
        self.profile = marrow.train(self.enrollment)

    def test_fixture_is_repeatable(self):
        self.assertEqual(self.enrollment, marrow.fixture('inserted', 2000, 'final-v1')[0])

    def test_labels_and_unknown_fields_are_not_read(self):
        original = self.tests[0][1]
        altered = copy.deepcopy(original)
        altered.update(role='stranger', style='different', session_id='another')
        for event in altered['events']:
            event['unrelated'] = 'ignored'
        self.assertEqual(marrow.features(original), marrow.features(altered))
        self.assertEqual(marrow.score(self.profile, original), marrow.score(self.profile, altered))

    def test_invalid_recordings_rejected(self):
        for event in (
            {'type':'pointer', 't_ms':1, 'x':1.2, 'y':0.5},
            {'type':'pointer', 't_ms':float('nan'), 'x':0.2, 'y':0.5},
            {'type':'text', 't_ms':1, 'source':'inserted', 'action':'add', 'count':0},
            {'type':'mystery', 't_ms':1},
        ):
            with self.subTest(event=event), self.assertRaises(ValueError):
                marrow.features({'version':1, 'events':[event]})

    def test_short_recording_does_not_get_admitted(self):
        result = marrow.score(self.profile, {'version':1, 'events':[]})
        self.assertEqual(result['decision'], 'insufficient_evidence')
        self.assertIsNone(result['score'])

    def test_common_slowdown_can_be_normalized(self):
        profile = {'threshold': 1.1, 'model': {
            'click_gap': {'median': 1.0, 'scale': .1},
            'typed:text_gap': {'median': 1.0, 'scale': .1},
            'speed': {'median': .2, 'scale': .1},
            'turn': {'median': .5, 'scale': .1},
        }}
        with patch.object(marrow, 'features', return_value={
            'click_gap': 1.2, 'typed:text_gap': 1.2, 'speed': .2, 'turn': .5,
        }):
            self.assertEqual(marrow.score(profile, {})['decision'], 'admit')

    def test_unordered_events_have_same_features(self):
        a = self.tests[0][1]
        b = copy.deepcopy(a)
        b['events'].reverse()
        # Equal-timestamp ordering is preserved by Python's stable sort, so ensure
        # test session timestamps are distinct before checking order invariance.
        for n, event in enumerate(b['events']):
            event['t_ms'] += n / 1000000
        b['events'].sort(key=lambda event: event['t_ms'])
        expected = marrow.features(b)
        b['events'].reverse()
        self.assertEqual(expected, marrow.features(b))

    def test_offline_profile_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marrow.write_jsonl(root/'enroll.jsonl', self.enrollment)
            profile = marrow.train(marrow.read_jsonl(root/'enroll.jsonl'))
            encoded = json.dumps(profile)
            self.assertEqual(marrow.score(profile, self.tests[0][1]),
                             marrow.score(json.loads(encoded), self.tests[0][1]))

    def test_own_and_other_profile_receive_same_unlabeled_schema(self):
        own = self.tests[0][1]
        wrong_enroll, _ = marrow.fixture('inserted', 2101, 'swap-profile-v1')
        result = marrow.score(marrow.train(wrong_enroll), own)
        self.assertEqual(result['decision'], 'reject')


if __name__ == '__main__':
    unittest.main()
