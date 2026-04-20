# session_logs/ — Development Session Handoff Reports

Point-in-time snapshots of research / engineering progress, intended as handoff
notes between sessions (or between contributors). Files are timestamped
`SESSION_HANDOFF_YYYYMMDD.md` or `SESSION_HANDOFF_YYYYMMDD_topic.md`.

## Files

| File | Topic | State at time of writing |
|------|-------|--------------------------|
| `SESSION_HANDOFF_20260415.md` | PLIC JAX parallel research status briefing | Redistribute spec validated, float32 only; was previously `docs/PLIC_JAX_PARALLEL_STATUS.md` |
| `SESSION_HANDOFF_20260416.md` | Conservative redistribute + float64 support + OpenFOAM Zalesak 3D comparison | Current |

## When to use

- **Onboarding** a new contributor: start with the latest handoff.
- **Understanding project history**: chronological narrative of what was tried,
  what failed, and what was decided.
- **Debugging a regression**: find the session where the relevant change landed.

## Authority

Session logs are **historical snapshots**. They may reference numbers,
decisions, or code paths that no longer apply. When a session log conflicts
with `specs/` or `benchmark/`, **the latter wins** — session logs are not
maintained after they are written.

Do not update a session log retroactively. Write a new one instead.
