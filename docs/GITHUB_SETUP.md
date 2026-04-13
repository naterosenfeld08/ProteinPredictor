# Upload this project to a new GitHub account

## 1. Install tools (new machine checklist)

| Tool | Why |
|------|-----|
| **Git** | Version control + push to GitHub |
| **Python 3.10+** (3.11/3.12 OK) | Runs the pipeline |
| **pip** | Installs dependencies |

Optional but convenient:

- **GitHub CLI** (`gh`): create repos and auth from the terminal — [install `gh`](https://cli.github.com/).

Verify:

```bash
git --version
python3 --version
pip --version
```

## 2. Python environment (recommended)

```bash
cd /path/to/petase-thermostability-benchmark
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Quick sanity check:

```bash
python -m py_compile predict.py train_mlp_rf_ensemble.py mlp_rf_ensemble.py protein_baseline.py
```

First run will download **Hugging Face** model weights (~GBs) into your user cache (ignored by git).

## 3. What gets pushed vs stays local

`.gitignore` is set so you **do not** commit:

- Large **CSV** datasets (`*.csv`, `fireprotdb*.csv`)
- **Trained models** (`*.pkl`), **embeddings cache**, **numpy** artifacts
- **`training_output*/`** (including `training_output (CRITICAL DIRECTORY DO NOT TOUCH)/`)
- **`uniprot_cache/`** (downloaded sequences / API cache)
- **`embeddings_cache/`**, Hugging Face **`.cache`**

Cloners get the **code** and **`requirements.txt`**; they obtain FireProtDB separately and retrain or supply their own `*.pkl` if you distribute it outside GitHub (e.g. release asset, cloud storage).

## 4. Commit your latest code

```bash
git status
git add README.md predict.py protein_baseline.py mlp_baseline.py validate_model.py uniprot_fetcher.py \
  mlp_rf_ensemble.py train_mlp_rf_ensemble.py .gitignore docs/GITHUB_SETUP.md
git commit -m "Ensemble training, inference fixes, GitHub setup docs"
```

Adjust `git add` if you want fewer/more files.

## 5. Point `origin` at the new GitHub repo

On GitHub: **New repository** → create **empty** repo (no README/license if you already have them here).

Then either replace the remote:

```bash
git remote remove origin   # optional: only if you want to drop the old remote
git remote add origin https://github.com/NEW_USERNAME/NEW_REPO.git
```

Or change URL in place:

```bash
git remote set-url origin https://github.com/NEW_USERNAME/NEW_REPO.git
git remote -v
```

Push:

```bash
git push -u origin main
```

If GitHub’s default branch is `master`, use `git push -u origin main:main` or rename branches to match.

### Using SSH instead of HTTPS

```bash
git remote set-url origin git@github.com:NEW_USERNAME/NEW_REPO.git
```

Ensure your SSH key is [added to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

## 6. Optional: GitHub CLI workflow

After installing `gh`:

```bash
gh auth login
cd /path/to/petase-thermostability-benchmark
gh repo create NEW_REPO --private --source=. --remote=origin --push
```

## 7. Optional: Git LFS

Only needed if you **intentionally** want large binaries in the repo. This project is configured to **exclude** models/data from git; LFS is usually **not** required.

---

**Security:** Never commit API tokens, `.env` with secrets, or personal `uniprot_cache` if it contains sensitive paths—keep them gitignored.
