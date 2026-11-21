# Overhead Traffic Detection and Analysis

## Overview


## Code Usage

### Setting up environment in windows
> Don't use windows it sucks if you are on a windows machine, install wsl and follow the linux instructions

### Setting up environment linux
> navigate to to the root directory of the project
> create venv: `python -m venv .venv`
> activate venv: `source .venv/bin/activate`
> install requirements: `pip install -r requirements.txt`

### Running the code
> after environment setup run code with `python ./code/main.py`

### Training
> Actual training happens on Colab where cloud GPUs are avalable
> 1. Push your changes to GitHub. `train_colab.ipynb` clones the repo on Colab, so the branch set in `REPO_BRANCH` is what gets trained (no uploading code to Drive)
> 2. Private repo only: add a GitHub token with read access as the Colab secret `GITHUB_TOKEN` (key icon in the left sidebar)
> 3. Open `train_colab.ipynb` in Colab, set `REPO_URL`, `REPO_BRANCH`, and `DRIVE_DIR` in the settings cell, then Runtime → Run all
> 4. Trained models are saved to `DRIVE_DIR/models/` on Google Drive. Each run folder has `code_version.txt` with the commit it was trained from



