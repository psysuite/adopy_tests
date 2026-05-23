library(tidyverse)

# Leggi il CSV
data <- read_csv("/data/Dropbox/RDATA/R_bis_ad_fx/results_simulations/tables/stimulus_metrics_by_group.csv", show_col_types = FALSE)

# Crea tabella per ogni metrica
# Layout: 3 righe (PSE group) × 9 colonne
# Riga 1: G1_ABS1, G1_REL1, G1_REL2, G2_ABS1, G2_REL1, G2_REL2, G3_ABS1, G3_REL1, G3_REL2
# Riga 2: G4_ABS1, G4_REL1, G4_REL2, G5_ABS1, G5_REL1, G5_REL2, G6_ABS1, G6_REL1, G6_REL2
# Riga 3: G7_ABS1, G7_REL1, G7_REL2, G8_ABS1, G8_REL1, G8_REL2, G9_ABS1, G9_REL1, G9_REL2

create_table <- function(data, metric_mean, metric_sd, decimals = 2) {
  # Crea la tabella 3x9
  result <- matrix(NA, nrow = 3, ncol = 9)
  
  # Mapping: riga -> gruppi
  groups_by_row <- list(
    c("G1", "G2", "G3"),
    c("G4", "G5", "G6"),
    c("G7", "G8", "G9")
  )
  
  for (i in 1:3) {
    groups_for_row <- groups_by_row[[i]]
    col_idx <- 1
    
    for (j in 1:3) {
      group_val <- groups_for_row[j]
      
      for (k in 1:3) {
        model_val <- c("ABS1", "REL1", "REL2")[k]
        
        # Trova il valore nel dataframe usando group e model
        row_data <- data %>%
          filter(group == group_val, model == model_val)
        
        if (nrow(row_data) > 0) {
          mean_val <- row_data %>% pull(!!sym(metric_mean)) %>% first()
          sd_val <- row_data %>% pull(!!sym(metric_sd)) %>% first()
          fmt_str <- paste0("%.", decimals, "f")
          result[i, col_idx] <- paste0(sprintf(fmt_str, mean_val), " ± ", sprintf(fmt_str, sd_val))
        }
        
        col_idx <- col_idx + 1
      }
    }
  }
  
  return(as.data.frame(result, stringsAsFactors = FALSE))
}

# Crea le tre tabelle
table_center <- create_table(data, "stimulus_center_mean", "stimulus_center_sd", decimals = 1)
table_spread <- create_table(data, "stimulus_spread_mean", "stimulus_spread_sd", decimals = 1)

# Salva senza header
write_csv(table_center, 
          "/data/Dropbox/RDATA/R_bis_ad_fx/results_simulations/tables/stimulus_center_table.csv",
          col_names = FALSE)
write_csv(table_spread, 
          "/data/Dropbox/RDATA/R_bis_ad_fx/results_simulations/tables/stimulus_spread_table.csv",
          col_names = FALSE)


cat("✓ Tabelle create (3 righe × 9 colonne):\n")
cat("  SC e SS: 1 decimale\n")
cat("  BI: 2 decimali\n")
cat("  - stimulus_center_table.csv\n")
cat("  - stimulus_spread_table.csv\n")

