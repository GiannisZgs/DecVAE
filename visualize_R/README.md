# R Visualization Scripts

R scripts for generating figures from the paper. Each subdirectory contains scripts for specific figures.

## Prerequisites

Ensure R packages are installed:
```bash
Rscript ../env_setup/setup.R
```

### Method 1: Direct Execution 
From the command line, navigate to the script's directory and run:

```bash
cd visualize_R/fig_2
Rscript models_performance_scatter_vowels.R
```

### Method 2: Using R Console

Launch R console from the DecVAE root directory:

```bash
R
```

Then source the script:

```r
# From DecVAE root
source("visualize_R/fig_2/models_performance_scatter_vowels.R")
```

## Output

Figures are saved to `../figures/` and `../supplementary_figures` directories.

## Notes

- Most scripts load data from `../data/results_wandb_exports_for_figures/`
