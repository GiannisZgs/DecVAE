# ============================================================================
# DecVAE Visualizations R Environment Setup
# ============================================================================
# This script installs all required R packages for reproducing the 
# DecVAE paper visualization analysis.
#
# Usage:
#   cd DecVAE
#   source("env_setup/setup.R")
#
# Copyright 2025 Ioannis Ziogas <ziogioan@ieee.org>
# ============================================================================

cat("=========================================\n")
cat("DecVAE Visualizations R Environment Setup\n")
cat("=========================================\n\n")

required_r_version <- "4.5.1"
current_r_major <- R.version$major
current_r_minor <- R.version$minor
current_r_version <- paste(current_r_major, current_r_minor, sep = ".")

cat("Required R version:", required_r_version, "\n")
cat("Current R version: ", current_r_version, "\n\n")

if (current_r_version != required_r_version) {
  warning(sprintf(
    "R version mismatch detected!\n  Required: %s\n  Current:  %s\n",
    required_r_version, current_r_version
  ))
  cat("Continuing with current version...\n\n")
}

# Core packages used across visualization scripts
required_packages <- c(
  # Data manipulation
  "dplyr",           # Data manipulation
  "tidyr",           # Data tidying
  "readr",           # Reading CSV files
  "stringr",         # String manipulation
  "purrr",           # Functional programming tools
  
  # Visualization core
  "ggplot2",         # Graphics and plotting
  "scales",          # Scale functions for visualization
  "viridis",         # Color palettes
  
  # Plot composition and arrangement
  "patchwork",       # Combining plots
  "cowplot",         # Publication-ready plots
  "gridExtra",       # Grid arrangements
  "grid",            # Grid graphics
  
  # Specialized plotting
  "ggExtra",         # Marginal plots
  "ggrepel",         # Repelling text labels
  "ggbreak",         # Axis breaks
  "ggnewscale",      # Multiple color scales
  "plotly",          # Interactive 3D plots
  
  # Color palettes
  "RColorBrewer",    # Color schemes
  "MetBrewer",       # Art museum color palettes
  
  # Data import/export
  "jsonlite",        # JSON parsing
  "R.utils",         # Utility functions
  
  # Fonts and text
  "showtext"         # Custom fonts
)

cat("Required packages (", length(required_packages), " total):\n", sep = "")
cat(paste("-", required_packages), sep = "\n")
cat("\n")

cat("Checking installed packages...\n\n")

# Determine which packages need to be installed
installed <- installed.packages()[, "Package"]
missing_packages <- setdiff(required_packages, installed)

if (length(missing_packages) == 0) {
  cat("All required packages are already installed!\n\n")
} else {
  cat("Installing", length(missing_packages), "missing package(s):\n")
  cat(paste("-", missing_packages), sep = "\n")
  cat("\n")
  
  # Install missing packages with progress
  for (pkg in missing_packages) {
    cat("Installing", pkg, "... ")
    tryCatch({
      install.packages(pkg, dependencies = TRUE, quiet = TRUE)
      cat("OK\n")
    }, error = function(e) {
      cat("FAILED\n")
      warning(sprintf("Failed to install package '%s': %s", pkg, e$message))
    })
  }
  cat("\n")
}


cat("Verifying package installation...\n\n")

verification_failed <- character(0)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    verification_failed <- c(verification_failed, pkg)
  }
}

if (length(verification_failed) > 0) {
  cat("WARNING: The following packages could not be loaded:\n")
  cat(paste("-", verification_failed), sep = "\n")
  cat("\nPlease install these packages manually:\n")
  cat(sprintf('install.packages(c("%s"))\n', 
              paste(verification_failed, collapse = '", "')))
} else {
  cat("All packages verified successfully!\n")
}

cat("\n")
cat("=========================================\n")
cat("Installed Package Versions\n")
cat("=========================================\n")

for (pkg in required_packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    version <- as.character(packageVersion(pkg))
    cat(sprintf("%-20s %s\n", pkg, version))
  } else {
    cat(sprintf("%-20s NOT INSTALLED\n", pkg))
  }
}

cat("\n")
cat("=========================================\n")
cat("Setup Complete!\n")
cat("=========================================\n")
cat("\nYou can now run any visualization script in this project.\n")
cat("Example: source('fig_2/models_performance_scatter_vowels.R')\n\n")

#invisible(NULL)
