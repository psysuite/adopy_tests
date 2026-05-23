# ============================================================================== =
# effect_size_utils.R
# Utility functions for calculating effect sizes from ANOVA results
# ============================================================================== =

# Function to extract eta-squared from aovperm results
# Handles both simple ANOVA (no Error term) and mixed models (with Error term)
extract_eta_squared <- function(anova_result) {
  
  # Get the summary
  res <- summary(anova_result)
  
  # Check if this is a simple ANOVA (no Error term) or mixed model
  if (is.data.frame(res) && "SS" %in% colnames(res)) {
    # Simple ANOVA: res is a data frame with SS column
    SS_total <- sum(res$SS, na.rm = TRUE)
    eta_sq <- res$SS / SS_total
    
    effect_sizes <- data.frame(
      term = row.names(res),
      SS = res$SS,
      eta_squared = eta_sq,
      row.names = NULL,
      stringsAsFactors = FALSE
    )
  } else if (is.data.frame(res) && "SSn" %in% colnames(res) && "SSd" %in% colnames(res)) {
    # Mixed model with Error term: res is a data frame with SSn and SSd columns
    SSn <- res$SSn
    SSd <- res$SSd
    
    # Calculate eta-squared: η² = SSn / (SSn + SSd)
    eta_sq <- SSn / (SSn + SSd)
    
    effect_sizes <- data.frame(
      term = row.names(res),
      SSn = SSn,
      SSd = SSd,
      eta_squared = eta_sq,
      row.names = NULL,
      stringsAsFactors = FALSE
    )
  } else {
    stop("Could not extract effect sizes from ANOVA result. Expected SS or (SSn, SSd) columns.")
  }
  
  return(effect_sizes)
}

# Function to print effect sizes nicely
print_effect_sizes <- function(effect_sizes, title = "Effect Sizes (η²)") {
  cat("\n", strrep("=", 70), "\n", sep = "")
  cat(title, "\n")
  cat(strrep("=", 70), "\n", sep = "")
  
  for (i in 1:nrow(effect_sizes)) {
    term <- effect_sizes$term[i]
    eta_sq <- effect_sizes$eta_squared[i]
    
    # Interpret effect size
    if (eta_sq < 0.01) {
      interpretation <- "negligible"
    } else if (eta_sq < 0.06) {
      interpretation <- "small"
    } else if (eta_sq < 0.14) {
      interpretation <- "medium"
    } else {
      interpretation <- "large"
    }
    
    cat(sprintf("%-30s η² = %.4f (%s)\n", term, eta_sq, interpretation))
  }
  cat(strrep("=", 70), "\n\n", sep = "")
}

# Function to add effect sizes to a dataframe
add_effect_sizes_to_df <- function(anova_result, df_name = "anova_results") {
  
  effect_sizes <- extract_eta_squared(anova_result)
  effect_sizes$anova_model <- df_name
  
  return(effect_sizes)
}

# Interpretation guide
interpret_eta_squared <- function(eta_sq) {
  if (eta_sq < 0.01) {
    return("negligible")
  } else if (eta_sq < 0.06) {
    return("small")
  } else if (eta_sq < 0.14) {
    return("medium")
  } else {
    return("large")
  }
}
