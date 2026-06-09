# Step 1: Credential Security and Ignore Rules

Date: 2026-06-10

## Scope

This step addresses only the first production-readiness item:

1. Remove Spotify credentials and cached access tokens from the working tree.
2. Read Spotify credentials from environment variables.
3. Correct Git ignore rules for local secrets and generated files.

Model preprocessing, application architecture, tests, and deployment are intentionally
left for later steps.

## Before

### Spotify credentials

The Spotify client ID and client secret were written directly in:

- `real_time_emotion_detection.py`
- `spotify_auth.py`

The same credential pair was also committed in Git commit `da0d69a`.

Risk:

- Anyone with repository access could use the Spotify application credentials.
- Removing the values from the latest files would not remove them from Git history.
- Deployments required editing source code to change credentials.

### Authentication cache

Spotipy created a `.cache` file containing a bearer access token. The file was
untracked, but the `.gitignore` entry was `.cache/`, which only matched a directory.
It did not ignore the actual `.cache` file.

### Other ignore rules

The Python cache rule was `_pycache_/` instead of `__pycache__/`. The existing
`.dist` directory was also not covered by the `dist/` rule.

## After

### Environment-based configuration

`spotify_config.py` now reads:

- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_REDIRECT_URI` (optional, defaults to `http://localhost:8888/callback`)

Both Spotify entry points use this shared configuration. Missing required credentials
produce a clear error instead of silently using an embedded secret.

`.env.example` documents the expected names. As a follow-up verified on 2026-06-10,
the project now loads the ignored local `.env` file with `python-dotenv`; production
platform environment variables still take precedence.

### Local sensitive files

The cached token file was removed. `.gitignore` now excludes `.cache*`,
`.spotify-cache*`, `.env`, and environment-specific `.env.*` files while retaining
`.env.example`.

Python cache, virtual-environment, generated model, dataset, editor, and build-output
rules were also corrected and grouped by purpose.

## Why This Changed

Secrets must be supplied by the runtime environment so they can be changed without
editing code, kept out of commits, and managed separately in development and
production. Token caches are local authentication state and must never be committed.

## Required Manual Action

Code changes cannot revoke a secret that has already been exposed. Before using the
Spotify integration again:

1. Open the Spotify Developer Dashboard.
2. Rotate or regenerate the client secret for this application.
3. Treat the previous client ID/secret pair as compromised.
4. Configure the new values in the shell or deployment platform.

PowerShell example for the current terminal session:

```powershell
$env:SPOTIFY_CLIENT_ID = "your-client-id"
$env:SPOTIFY_CLIENT_SECRET = "your-new-client-secret"
$env:SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"
```

Do not place real values in `.env.example`.

## Git History Note

The old credential remains visible in existing Git history until that history is
rewritten. History rewriting affects every clone and collaborator, so it was not
performed automatically in this step. Rotating the secret is the immediate security
requirement; history cleanup can be planned separately if the repository is public.
