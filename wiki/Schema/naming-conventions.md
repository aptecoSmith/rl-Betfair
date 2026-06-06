# Naming conventions

- **Files:** `kebab-case.md` — e.g. `ppo-lstm-policy.md`, not `PPO LSTM Policy.md`.
- **Folders:** notes live under their cloud + type:
  `shared/concepts/`, `shared/entities/`, `shared/topics/`, `shared/projects/`,
  `shared/syntheses/`, `shared/logs/`, `shared/queries/` (and the same under `personal/`).
- **Hubs:** each cloud has `index.md` (root hub, generated) and may have topic hub notes.
- **Source IDs:** `src-` + 6-char hash, stable once assigned.
- **Links:** Obsidian `[[note-name]]` in the body; cross-cloud links from `personal` → `shared` use
  `[[shared/concepts/foo]]` form. Body links are the graph the connectivity check reads.
- **Tags:** lowercase, singular, kebab-case; must exist in `tag-vocabulary.md`.
- **Dates:** ISO-8601 (`YYYY-MM-DD`).
