#!/usr/bin/env bash
#
# Optional empty commit + push for a dedicated "contributions" repo.
# GitHub counts contributions when the commit author email matches your account
# and the commit lands on the default branch.
#
# Setup:
#   1. Create a NEW empty repo on GitHub (e.g. naterosenfeld08/daily-activity), clone it.
#   2. Set REPO_DIR below to that clone path (NOT your main project unless you want noise).
#   3. In that repo: git config user.email to match GitHub (e.g. naterosenfeld08@users.noreply.github.com)
#   4. chmod +x scripts/auto_contribution_commit.sh
#   5. Run once manually: ./scripts/auto_contribution_commit.sh
#   6. Install LaunchAgent (see docs/AUTO_CONTRIBUTION_SCHEDULE.md)
#
set -euo pipefail

# ---- EDIT THIS: path to the repo that should receive empty commits ----
REPO_DIR="${AUTO_COMMIT_REPO_DIR:-$HOME/Developer/daily-activity}"

# Random skip: ~2/3 of days no commit (less uniform grid). Set to 0 to always commit.
RANDOM_SKIP="${AUTO_COMMIT_RANDOM_SKIP:-1}"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "ERROR: REPO_DIR is not a git repo: $REPO_DIR" >&2
  echo "Set AUTO_COMMIT_REPO_DIR or edit REPO_DIR in this script." >&2
  exit 1
fi

# Homebrew git/ssh (cron/LaunchAgent have minimal PATH)
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

cd "$REPO_DIR"

_ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }

if [[ "$RANDOM_SKIP" == "1" ]] && [[ $((RANDOM % 3)) -ne 0 ]]; then
  echo "$(_ts) skip (random)"
  exit 0
fi

git commit --allow-empty -m "chore: activity $(date '+%Y-%m-%d %H:%M:%S %z')"
git push

echo "$(_ts) pushed empty commit OK"
