---
name: feature-addition-with-ui-and-tests
description: Workflow command scaffold for feature-addition-with-ui-and-tests in pw-plan-cleaner-01.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-addition-with-ui-and-tests

Use this workflow when working on **feature-addition-with-ui-and-tests** in `pw-plan-cleaner-01`.

## Goal

Implements a new feature by adding backend logic, updating the UI, and creating corresponding pipeline tests.

## Common Files

- `cleaner/*.py`
- `app.py`
- `tests/test_pipeline_stages.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create new processing logic in a dedicated module under cleaner/
- Update app.py to add UI controls and integrate the new feature into the interface
- Add or update tests in tests/test_pipeline_stages.py to cover the new feature

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.