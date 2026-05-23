library(stats)
library(rcompanion)
library(tidyr)
library(dplyr)


meansd_aggfunc    <- function(x) c(mean = mean(x), sd = sd(x))


# ============================================================================================== = 
# run pairwise comparisons
# ============================================================================================== = 
do_phpw <- function(df, splitfactor, resp_var, factor_var, corr="fdr") {
  
  df_split <- split(df, df[[splitfactor]])
  run_lapply_npar_ph_pairwise(df_split, resp_var, factor_var, corr)
  
  summary_df <- aggregate(as.formula(paste0(resp_var, "~", splitfactor, " + ", factor_var)), data = df, FUN = meansd_aggfunc)
  return(summary_df)
}

run_lapply_npar_ph_pairwise <- function(datasets, response_var, group_var, corr="fdr") {
  print(lapply(datasets, function(subset_data) {
    npar_ph_pairwise(subset_data, response_var, group_var, corr)
  }))
}

npar_ph_pairwise <- function(df, response_var, group_var, corr="fdr") {
  pairwisePermutationTest(as.formula(paste0(response_var, " ~ ", group_var)), data = df, method = corr)
}

# ============================================================================================== = 
# run non parametric ANOVA and its posthoc
# ============================================================================================== = 
do_npar_anova_phpw <- function(df, splitfactor, resp_var, factor_var, corr="fdr") {
  
  df_split  <- split(df, df[[splitfactor]])
  anovas    <- run_lapply_npar_ph_anova(df_split, resp_var, factor_var)
  
  pvalues <- c()
  for(v in anovas){
    pvalues <- c(pvalues, v$p.value)
  }
  corrpvalues <- p.adjust(pvalues, corr)  
  
  print(paste0(resp_var, " x ", factor_var, " splitted by ", splitfactor))
  sign_res <- list()
  id <- 0
  for(v in corrpvalues){
    id <- id + 1
    if(v < 0.05){
      split_level  <- names(anovas)[id]
      df_filtered <- df[df[[splitfactor]] == split_level,]
      print(paste0("in ", split_level, " (H=", anovas[[id]]$statistic, ", p=", corrpvalues[id], ")"))
      sign_res[[split_level]] <- list("anova_pvalues"=corrpvalues[id], "ph"=npar_ph_pairwise(df_filtered, resp_var, factor_var, corr))
      pvalues <- sign_res[[split_level]]$ph$p.adjust
      iid <- 0
      for(pv in pvalues){
        iid <- iid + 1
        if(pv < 0.05){
          print(paste0(sign_res[[split_level]]$ph$Comparison[iid], " (W=", sign_res[[split_level]]$ph$Stat[iid], ", p=", pv, ")"))
        }
        else{
          print(paste0("NOT SIGNIFICANT in ", sign_res[[split_level]]$ph$Comparison[iid], " (W=", sign_res[[split_level]]$ph$Stat[iid], ", p=", pv, ")"))
          
        }
      } 
    }
    else{
      print(paste0("NOT SIGNIFICANT in ", names(anovas)[id], " (H=", anovas[[id]]$statistic, ", p=", corrpvalues[id], ")"))
    }
  }
  summary_df <- aggregate(as.formula(paste0(resp_var, "~", splitfactor, " + ", factor_var)), data = df, FUN = meansd_aggfunc)
  return(list("sign_levels"=sign_res, "summary"=summary_df))
}


run_lapply_npar_ph_anova <- function(datasets, response_var, group_var) {
  lapply(datasets, function(subset_data) {
    npar_ph_anova(subset_data, response_var, group_var)
  })
}

npar_ph_anova <- function(df, response_var, factor_var) kruskal.test(as.formula(paste0(response_var, " ~ ", factor_var)), data = df)


# ============================================================================================== = 
# Main effects: ANOVA non-parametrica senza split
# ============================================================================================== = 

do_npar_anova_main <- function(df, resp_var, factor_var, corr="fdr") {
  # Test main effect di un fattore singolo usando Kruskal-Wallis
  # Utile per testare effetti principali senza interazioni
  
  anova_result <- npar_ph_anova(df, resp_var, factor_var)
  
  print(paste0("Main effect: ", resp_var, " ~ ", factor_var))
  print(paste0("H = ", round(anova_result$statistic, 4), ", p = ", round(anova_result$p.value, 4)))
  
  if(anova_result$p.value < 0.05) {
    print("SIGNIFICANT - Running pairwise comparisons...")
    ph_result <- npar_ph_pairwise(df, resp_var, factor_var, corr)
    print(ph_result)
    return(list("anova" = anova_result, "posthoc" = ph_result))
  } else {
    print("NOT SIGNIFICANT")
    return(list("anova" = anova_result, "posthoc" = NULL))
  }
}

do_npar_anova_main_with_summary <- function(df, resp_var, factor_var, corr="fdr") {
  # Main effect + summary statistics
  
  result <- do_npar_anova_main(df, resp_var, factor_var, corr)
  summary_df <- aggregate(as.formula(paste0(resp_var, "~", factor_var)), data = df, FUN = meansd_aggfunc)
  
  print("\nSummary statistics:")
  print(summary_df)
  
  return(list("anova" = result$anova, "posthoc" = result$posthoc, "summary" = summary_df))
}


# ============================================================================================== = 
# Main effects con controllo per soggetti (repeated measures)
# ============================================================================================== = 

do_npar_anova_main_repeated <- function(df, resp_var, factor_var, subj_var = "subj", corr="fdr") {
  # Test main effect controllando per soggetti (repeated measures)
  # Usa Friedman test per dati appaiati
  # NOTA: Friedman test richiede dati bilanciati (ogni soggetto in ogni condizione)
  
  friedman_result <- friedman.test(as.formula(paste0(resp_var, " ~ ", factor_var, " | ", subj_var)), data = df)
  
  print(paste0("Main effect (repeated measures): ", resp_var, " ~ ", factor_var))
  print(paste0("chi-sq = ", round(friedman_result$statistic, 4), ", p = ", round(friedman_result$p.value, 4)))
  
  if(friedman_result$p.value < 0.05) {
    print("SIGNIFICANT - Running pairwise comparisons...")
    ph_result <- npar_ph_pairwise(df, resp_var, factor_var, corr)
    print(ph_result)
    return(list("anova" = friedman_result, "posthoc" = ph_result))
  } else {
    print("NOT SIGNIFICANT")
    return(list("anova" = friedman_result, "posthoc" = NULL))
  }
}


# ============================================================================================== = 
# Repeated measures ANOVA with posthoc (within-subject factor split by between-subject factor)
# ============================================================================================== = 

do_npar_anova_repeated_phpw <- function(df, splitfactor, resp_var, factor_var, subj_var = "subj", corr="fdr") {
  # Test within-subject factor (repeated measures) split by between-subject factor
  # Usa Friedman test per ogni livello del fattore di split
  
  df_split <- split(df, df[[splitfactor]])
  friedmans <- run_lapply_npar_repeated_anova(df_split, resp_var, factor_var, subj_var)
  
  pvalues <- c()
  for(v in friedmans){
    pvalues <- c(pvalues, v$p.value)
  }
  corrpvalues <- p.adjust(pvalues, corr)  
  
  print(paste0(resp_var, " x ", factor_var, " (within) splitted by ", splitfactor, " (between)"))
  sign_res <- list()
  id <- 0
  for(v in corrpvalues){
    id <- id + 1
    if(v < 0.05){
      split_level  <- names(friedmans)[id]
      df_filtered <- df[df[[splitfactor]] == split_level,]
      print(paste0("in ", split_level, " (chi-sq=", round(friedmans[[id]]$statistic, 4), ", p=", round(corrpvalues[id], 4), ")"))
      
      ph_result <- npar_ph_pairwise(df_filtered, resp_var, factor_var, corr)
      sign_res[[split_level]] <- list("anova_pvalues"=corrpvalues[id], "ph"=ph_result)
    }
    else{
      print(paste0("NOT SIGNIFICANT in ", names(friedmans)[id], " (chi-sq=", round(friedmans[[id]]$statistic, 4), ", p=", round(corrpvalues[id], 4), ")"))
    }
  }
  summary_df <- aggregate(as.formula(paste0(resp_var, "~", splitfactor, " + ", factor_var)), data = df, FUN = meansd_aggfunc)
  return(list("sign_levels"=sign_res, "summary"=summary_df))
}

run_lapply_npar_repeated_anova <- function(datasets, response_var, factor_var, subj_var) {
  lapply(datasets, function(subset_data) {
    npar_repeated_anova(subset_data, response_var, factor_var, subj_var)
  })
}

npar_repeated_anova <- function(df, response_var, factor_var, subj_var) {
  # Converte a formato wide per Friedman test
  # (Friedman richiede: una riga per soggetto, una colonna per ogni livello del fattore)
  
  # Aggregazione: se più osservazioni per soggetto-fattore, prendi la media
  df_agg <- df %>%
    select(all_of(c(subj_var, factor_var, response_var))) %>%
    group_by(across(all_of(c(subj_var, factor_var)))) %>%
    summarise(
      value = mean(get(response_var), na.rm = TRUE),
      .groups = "drop"
    ) %>%
    rename(!!response_var := value)
  
  df_wide <- df_agg %>%
    pivot_wider(
      names_from = factor_var,
      values_from = response_var
    )
  
  # Verifica che non ci siano NAs (dati non bilanciati)
  if(any(is.na(df_wide[, -1]))) {
    warning("Data not perfectly balanced - some subject-factor combinations missing")
  }
  
  # Estrai i dati numerici (escludendo la colonna soggetto)
  data_matrix <- as.matrix(df_wide[, -1])
  rownames(data_matrix) <- df_wide[[subj_var]]
  
  # Friedman test
  friedman.test(data_matrix)
}
