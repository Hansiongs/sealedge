# Privacy Policy

The sealedge package does not collect, transmit, or store user data on any
remote server.  This page documents what the package touches at runtime so
that JSS reviewers and end users can audit the data surface.

## Runtime data flow

- **Local-only data.**  All OHLCV, funding, and feature inputs are read from
  a user-specified `data_cache/` directory on the local filesystem.  The
  package never opens a network connection for OHLCV data; pre-caching is
  the user's responsibility and the SPA reproduction script fails pre-flight
  when required cache files are missing.
- **Public data fetchers are opt-in.**  `quant_lib.tools.data.fetch_klines`
  and `fetch_funding` hit Binance Vision monthly archives only when the user
  explicitly calls them.  No background telemetry, no automatic update checks,
  no error reporting over the network.
- **Holdout seals.**  HMAC seals are written to disk only under the
  user-chosen `holdout_seals/` directory (or the configured equivalent).  The
  HMAC secret is read from the environment variable `QUANT_LIB_HMAC_SECRET`;
  the package never logs the secret value.
- **Local logs.**  Optional CLI logs are written to a local file under
  `data_cache/logs/` (or the configured equivalent) when verbose mode is
  enabled.  These logs contain per-strategy metric numbers, file paths, and
  timing information; they do not contain user-identifiable information.

## Telemetry

- **No telemetry.**  The package does not emit, report, or aggregate
  usage statistics.
- **No third-party analytics.**  No cookies, no trackers, no external
  monitoring SDKs.

## Retention

- **User-controlled.**  All data and artifacts written by the package
  remain on the user's filesystem under their chosen paths.  Deleting the
  repository and the user-specified `data_cache/` directory is sufficient to
  remove all sealedge-managed data.

## User rights

- **Inspect.**  The committed `replication/output_paper_grade/` and
  `output_seal_demo/` directories contain only deterministic metrics, git
  commit hashes, and dependency versions.  Reviewers can inspect them without
  any private data exposure.
- **Erase.**  Removing `data_cache/`, `holdout_seals/`, and
  `data_cache/logs/` removes all package-managed state.
- **No profiling.**  Because the package collects no telemetry, no
  user-profile data exists to inspect, export, or erase.

## Contact

For privacy questions about the software itself, open a GitHub issue at
\url{https://github.com/Hansiongs/sealedge/issues}.  For questions about the
JSS manuscript, contact the corresponding author through the journal.

## Source of truth

This statement is consistent with the public source tree at
\url{https://github.com/Hansiongs/sealedge} (tag `v0.5.1`,
Zenodo DOI \url{https://doi.org/10.5281/zenodo.21329428}).
