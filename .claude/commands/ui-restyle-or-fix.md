---
name: ui-restyle-or-fix
description: Workflow command scaffold for ui-restyle-or-fix in pw-plan-cleaner-01.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /ui-restyle-or-fix

Use this workflow when working on **ui-restyle-or-fix** in `pw-plan-cleaner-01`.

## Goal

Makes visual or usability changes to the application's UI, including bug fixes and design updates.

## Common Files

- `app.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit app.py to adjust styles, layout, or component behavior
- Test visually to confirm changes
- Commit with a descriptive message

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.