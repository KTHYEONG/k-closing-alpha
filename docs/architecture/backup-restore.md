# Backup and Restore Runbook

Tiers:

- Sealed capture segments (`capture_sealed/<tier>/<YYYY-MM>/...`): append-only
  tar.zst segments uploaded with `rclone copyto --immutable`. An existing
  remote object with different content fails the upload instead of being
  overwritten.
- Loose copy (`data`, `artifacts`): nightly `rclone copy --backup-dir
  .../_deleted/<subtree>/<YYYY-MM-DD>`. Overwritten files move to `_deleted`.
- `_deleted/<subtree>/<YYYY-MM-DD>/`: 30-day age-based retention enforced by
  `backup_prune`. Purges abort when one subtree has more than 7 expired
  directories (clock/parse anomaly signal) and support `--dry-run`.
- Weekly/monthly snapshots (`snapshots/<YYYY-MM-DD>/` with `manifest.json`):
  immutable copies of rebuild-irreplaceable core panels (price history,
  decision archive, paper ledgers, top-k decisions, rank pool predictions, live
  bundle plus retrain registry), written with `rclone copy --immutable` every
  Sunday 10:00 KST. Retention keeps the newest 8 snapshots plus the first
  snapshot of each of the newest 12 months; nothing is pruned while fewer than
  8 snapshots exist. Never age-pruned.

How to list snapshots:

- List dated snapshots with `rclone lsf --dirs-only
  gdrive:quant-lake/live/k-closing-alpha/snapshots`.
- Read a manifest with `rclone cat
  gdrive:quant-lake/live/k-closing-alpha/snapshots/<YYYY-MM-DD>/manifest.json`.

Restore a core panel from `snapshots/<date>/` or `_deleted/<date>/`:

- Stop the kca timers before restoring so scheduled jobs do not overwrite the
  restored files.
- Download with `rclone copyto
  gdrive:quant-lake/live/k-closing-alpha/snapshots/<YYYY-MM-DD>/<relpath>
  <project-root>/<relpath>` (or the matching `_deleted/<subtree>/<date>/`
  path for the loose tier).
- Verify with the `manifest.json` sha256 entry for the restored relpath before
  restarting services.

Restore sealed capture with `restore_date`:

- Use the capture ledger (`<capture-root>/offsite/ledger`) and
  `restore_date` to materialize every committed segment of one trading date
  into a destination root. The restore verifies archive MD5 and per-member
  sha256 and refuses to overwrite a conflicting existing file.

Rule: stop kca timers for any restore (read-only procedure listed, no
commands executed by this runbook).
