# Scheduled empty commits (macOS) — optional

This is for a **separate, small GitHub repo** you use only for daily empty commits, so your main project (e.g. ProteinPredictor) does not fill with noise.

**Note:** The contribution graph is meant to reflect activity you care about; empty commits are allowed by Git but are purely cosmetic.

## 1. Create a repo on GitHub

1. New repository → e.g. `daily-activity` → **private** is fine → **no** README (empty).
2. Clone with SSH (same setup as ProteinPredictor):

   ```bash
   mkdir -p ~/Developer
   cd ~/Developer
   git clone git@github.com:naterosenfeld08/daily-activity.git
   cd daily-activity
   ```

3. Set **author email** GitHub will attribute (must match your account):

   ```bash
   git config user.name "Nathaniel Rosenfeld"
   git config user.email "naterosenfeld08@users.noreply.github.com"
   ```

## 2. Point the script at that repo

Either export before running:

```bash
export AUTO_COMMIT_REPO_DIR="$HOME/Developer/daily-activity"
```

Or edit `REPO_DIR` in `scripts/auto_contribution_commit.sh`.

## 3. Make executable and test once

```bash
cd /path/to/ProteinPredictor
chmod +x scripts/auto_contribution_commit.sh
AUTO_COMMIT_REPO_DIR="$HOME/Developer/daily-activity" ./scripts/auto_contribution_commit.sh
```

Check GitHub: a new commit on `main`.

To **always** commit (no random skip):

```bash
AUTO_COMMIT_RANDOM_SKIP=0 AUTO_COMMIT_REPO_DIR="$HOME/Developer/daily-activity" ./scripts/auto_contribution_commit.sh
```

## 4. Schedule on macOS — LaunchAgent (recommended)

`cron` on Mac often lacks your login **SSH agent** / Keychain context, so `git push` fails silently. **LaunchAgent** runs in your user session and usually works with existing SSH keys.

1. Create log directory:

   ```bash
   mkdir -p ~/.local/share
   ```

2. Copy the template and **fix paths** inside the plist (username, script path):

   ```bash
   cp docs/com.github.naterosenfeld08.autocommit.plist.template ~/Library/LaunchAgents/com.github.naterosenfeld08.autocommit.plist
   ```

3. Edit the plist: replace `YOURUSERNAME` and paths if needed.

4. Load and start:

   ```bash
   launchctl load ~/Library/LaunchAgents/com.github.naterosenfeld08.autocommit.plist
   ```

5. Verify loaded:

   ```bash
   launchctl list | grep autocommit
   ```

Logs: `~/.local/share/auto_commit.log` and `auto_commit.err`.

To unload:

```bash
launchctl unload ~/Library/LaunchAgents/com.github.naterosenfeld08.autocommit.plist
```

## 5. SSH must work non-interactively

If push asks for a passphrase, background jobs will hang or fail.

- Prefer **SSH key in ssh-agent** (unlocked after login), or
- A **dedicated key without passphrase** only for GitHub (narrow risk), configured in `~/.ssh/config` with `Host github.com` and `IdentityFile`.

Test:

```bash
ssh -T git@github.com
```

## 6. Cron alternative (less reliable on Mac)

```bash
crontab -e
```

Add (adjust paths):

```cron
0 12 * * * AUTO_COMMIT_REPO_DIR=$HOME/Developer/daily-activity /bin/bash /Users/YOU/ProteinPredictor/scripts/auto_contribution_commit.sh >> $HOME/.local/share/auto_commit.cron.log 2>&1
```

If push fails, switch to LaunchAgent or run from a small VPS with SSH deploy key.
