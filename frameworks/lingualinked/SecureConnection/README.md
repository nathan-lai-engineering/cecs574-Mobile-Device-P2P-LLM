# SecureConnection

Python-side communication layer for LinguaLinked. Contains the coordinator server, monitor, and client protocol implementations.

Key files in this directory:

| File | Role |
|---|---|
| `root_server.py` | Thread handler for the coordinator's device lifecycle (Ready → Open → Prepare → Initialized → Start → Running → Finish → Close) |
| `server.py` | ZMQ ROUTER socket helpers for the coordinator |
| `monitor.py` | Hardware metrics collection from connected devices before sharding |

To run the coordinator, do **not** run files in this directory directly. Run `root.py` from the `frameworks/lingualinked/` directory:

```bash
# From frameworks/lingualinked/
python root.py
```

See `frameworks/lingualinked/README.md` for the full startup sequence.
