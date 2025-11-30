## Environment Setup

### 1. Python Environment

**Using Conda**
```bash
conda env create -f env_setup/decVAE_conda.yml
conda activate DecVAE
```

**Using pip only**
```bash
pip install -r env_setup/decVAE_pip_requirements.txt
```

*Note: The `.yml` file creates a Conda environment with Python 3.11.9 and installs all packages from `.txt` via pip. If using Conda, you only need the `.yml` file.*

### 2. R Dependencies

Install R packages:
```bash
Rscript env_setup/setup.R
```