# Portable event recording format (version 1)

UTF-8 JSON Lines: each line is one JSON object containing `version: 1`, an optional `session_id` for the report, and an `events` array. Timestamps are nonnegative finite milliseconds relative to the start of the session. Events may be unsorted; the scorer sorts by timestamp. Pointer coordinates are finite fractions of the screen, in `[0,1]`. Device names are not used.

```json
{"version":1,"session_id":"example","events":[
 {"t_ms":100,"type":"pointer","x":0.25,"y":0.5},
 {"t_ms":220,"type":"pointer","x":0.27,"y":0.52},
 {"t_ms":300,"type":"click","button":0},
 {"t_ms":700,"type":"text","source":"inserted","action":"add","count":8},
 {"t_ms":880,"type":"text","source":"inserted","action":"delete","count":2}
]}
```

The example above is indented for readability; in JSONL serialize each whole session on one line. Recognized event fields:

- All events: `t_ms`, `type`. Types: `pointer`, `click`, `text`.
- `pointer`: `x` and `y`, finite numbers from 0 to 1 inclusive.
- `click`: optional `button` metadata; the prototype currently uses click timing only.
- `text`: `source` is `typed`, `onscreen`, or `inserted`; `action` is `add` or `delete`; `count` is a positive integer (up to 10,000). This reports length only, never the words or key identities.

Unknown fields are ignored; do not place real text, passwords, or private content in recordings. Session IDs and any additional labels have no effect on feature extraction or scoring. Timestamps at an identical value retain file order; an unknown event type or an invalid coordinate, time, text source/action/count causes a validation error. A session is capped at 100,000 events.

Text source is the capture tool's best description of how input arrived, not proof of a device or disability. The scorer uses the same event schema for browser, desktop, assistive and synthetic sources; it does not itself capture events or verify capture provenance.
