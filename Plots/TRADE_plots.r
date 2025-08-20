# Load required packages
library(tidyverse)
library(reshape2)
library(gridExtra)
library(patchwork)
library(viridis)
library(ggrepel)

# Custom color mapping (fixed syntax error, using c() instead of {})
COLOR_MAP <- c(
  "NoPerturb" = "#BA5B59",
  "Baseline_1" = "#866AA3",  # Changed to display name Baseline_1
  "CPA" = "#708090",
  "GEARS" = "#F6DEA4",
  "scELMO" = "#E58760",
  "PerturbNet" = "#759971",
  "scGPT" = "#6CB3DA",
  "scFoundation" = "#92A8D3",
  "Biolord" = "#869FA1"
)

# Model name mapping - map original names to display names
MODEL_DISPLAY_MAPPING <- c(
  "Baseline" = "Baseline_1",
  "scGPT_epoch2" = "scGPT"
)

# Define file paths
base_path <- "/data2/lanxiang/perturb_benchmark_v2/Table/Task1_metrics"
trade_base_path <- "/data2/lanxiang/perturb_benchmark_v2/model/Trade"
output_base_path <- "/data2/lanxiang/perturb_benchmark_v2/Fig/Fig2/Trade_plot"

# Create output directory
dir.create(output_base_path, showWarnings = FALSE, recursive = TRUE)

# Define dataset list
datasets <- c(
  "DatlingerBock2017",
  "DixitRegev2016", 
  "NormanWeissman2019_filtered",
  "PapalexiSatija2021_eccite_RNA",
  "ReplogleWeissman2022_rpe1",
  "Sunshine2023_CRISPRi_sarscov2",
  "TianKampmann2021_CRISPRa",
  "TianKampmann2021_CRISPRi"
)

# Define categories
categories <- c("Allgenes", "Top100", "Top50", "Top20")

# Metric columns
metrics <- c('R_squared', 'Pearson_Correlation', "Pearson_Correlation_delta", 'Cosine_Similarity',
             'MSE', 'RMSE', 'MAE', 'L2', 'MMD', 'Wasserstein')

# Function to create line plots for individual metrics
plot_single_metric_trend <- function(data, metric_name, category_name, dataset_name) {
  # Filter data
  data_filtered <- data %>% 
    filter(category == category_name, Metric == metric_name) %>%
    mutate(condition_label = as.character(condition))
  
  # Create base plot
  p <- ggplot(data_filtered, aes(x = condition, y = Value, color = model_display, group = model_display)) +
    # Add connecting lines
    geom_line(size = 1.2, alpha = 0.8) +
    # Add data points
    geom_point(size = 3, alpha = 0.9) +
    # Use custom colors
    scale_color_manual(values = COLOR_MAP, name = "Model") +
    # Beautify theme
    theme_minimal() +
    theme(
      # Background settings
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      # Axis settings
      axis.line = element_line(color = "black", size = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10, color = "black"),
      axis.text.y = element_text(size = 10, color = "black"),
      axis.title.x = element_text(size = 12, color = "black", margin = margin(t = 10)),
      axis.title.y = element_text(size = 12, color = "black", margin = margin(r = 10)),
      # Title settings
      plot.title = element_text(size = 14, hjust = 0.5, color = "black", face = "bold"),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "grey40"),
      # Legend settings
      legend.position = "bottom",
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 10),
      legend.box.background = element_blank(),
      legend.box.margin = margin(t = 10),
      # Plot margins
      plot.margin = margin(20, 20, 20, 20),
      # Remove grid lines
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    ) +
    # Label settings
    labs(
      title = paste(metric_name, "Performance Across Perturbation Genes -", dataset_name),
      subtitle = paste("Category:", category_name, "| Genes ordered by decreasing perturbation strength"),
      x = "Perturbation Gene (High → Low Impact)",
      y = metric_name,
      color = "Model"
    )
  
  return(p)
}

# Function to create Train/Test distribution plot
plot_train_test_distribution <- function(impact_df, test_genes, dataset_name) {
  # Extract single genes (excluding those containing "+")
  impact_df_single <- impact_df %>%
    filter(!str_detect(perturb_gene, "\\+"))
  
  # Add column to mark whether it's train or test
  impact_df_single <- impact_df_single %>%
    mutate(split = if_else(perturb_gene %in% test_genes, "test", "train"))
  
  # Create plot
  p <- ggplot(impact_df_single, aes(x = split, y = transcriptome_wide_impact, fill = split)) +
    geom_violin(trim = FALSE, alpha = 0.4) +
    geom_jitter(width = 0.1, size = 2, aes(color = split)) +
    scale_fill_manual(values = c("train" = "#A8BCCC", "test" = "#E58760")) +
    scale_color_manual(values = c("train" = "#A8BCCC", "test" = "#E58760")) +
    labs(
      title = dataset_name,
      x = "", y = "Transcriptome-wide Impact"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black"),
      axis.text = element_text(size = 12, face = "bold", color = "black"),
      axis.title = element_text(size = 12, face = "bold", color = "black"),
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      legend.position = "none"  # Hide legend
    )
  
  return(p)
}

# Main loop: process each dataset
for(dataset in datasets) {
  cat("Processing dataset:", dataset, "\n")
  
  # Create dataset-specific output directory
  dataset_output_path <- file.path(output_base_path, dataset)
  dir.create(dataset_output_path, showWarnings = FALSE, recursive = TRUE)
  
  # Load transcriptome impact data
  impact_file <- file.path(trade_base_path, dataset, "transcriptome_wide_impact_summary.csv")
  
  # Check if impact file exists
  if(!file.exists(impact_file)) {
    cat("  Warning: Impact file does not exist, skipping dataset", dataset, "\n")
    next
  }
  
  impact_df <- read_csv(impact_file)
  
  # Sort data
  impact_df_sorted <- impact_df %>%
    mutate(
      perturb_gene = str_replace(perturb_gene, "\\+ctrl", ""),
      transcriptome_wide_impact = as.numeric(transcriptome_wide_impact)
    ) %>%
    arrange(desc(transcriptome_wide_impact))
  
  # Load data for each category
  data_list <- list()
  
  for(category in categories) {
    file_name <- paste0("Task1_", category, "_common_data.csv")
    file_path <- file.path(base_path, file_name)
    
    if(file.exists(file_path)) {
      data_list[[category]] <- read_csv(file_path) %>%
        mutate(category = category)
    } else {
      cat("  Warning: File does not exist:", file_path, "\n")
    }
  }
  
  # If no data found, skip
  if(length(data_list) == 0) {
    cat("  Warning: No data files found, skipping dataset", dataset, "\n")
    next
  }
  
  # Combine all data
  all_data <- bind_rows(data_list)
  
  # Filter current dataset, fix dataset name matching logic
  dataset_data <- all_data %>% 
    filter(tolower(dataset) == tolower(!!dataset)) %>%  # Fixed: use !!dataset instead of dataset
    # Filter out scGPT_epoch5 and rename scGPT_epoch2 to scGPT
    filter(model != "scGPT_epoch5") %>%
    mutate(model = case_when(
      model == "scGPT_epoch2" ~ "scGPT",
      TRUE ~ model
    )) %>%
    # Add display name column
    mutate(model_display = case_when(
      model %in% names(MODEL_DISPLAY_MAPPING) ~ MODEL_DISPLAY_MAPPING[model],
      TRUE ~ model
    ))
  
  # If no matching data, skip
  if(nrow(dataset_data) == 0) {
    cat("  Warning: No matching data found, skipping dataset", dataset, "\n")
    next
  }
  
  # Set condition order
  dataset_data <- dataset_data %>%
    mutate(condition = factor(condition, levels = impact_df_sorted$perturb_gene))
  
  # Convert to long format
  dataset_long <- dataset_data %>%
    pivot_longer(cols = all_of(metrics),
                 names_to = "Metric",
                 values_to = "Value")
  
  # Set metric order
  dataset_long$Metric <- factor(dataset_long$Metric, levels = metrics)
  
  # Generate line plots for each category and each metric
  for(category in categories) {
    # Check if this category has data
    if(!category %in% dataset_long$category) next
    
    cat("  Processing category:", category, "\n")
    
    for(metric in metrics) {
      # Check if this metric has data
      metric_data <- dataset_long %>% 
        filter(category == category, Metric == metric)
      
      if(nrow(metric_data) == 0) next
      
      cat("    Generating metric:", metric, "\n")
      
      # Generate plot
      tryCatch({
        p <- plot_single_metric_trend(dataset_long, metric, category, dataset)
        
        # Save plot
        filename <- paste0("trend_", category, "_", metric, ".pdf")
        ggsave(file.path(dataset_output_path, filename), 
               p, width = 10, height = 8, dpi = 300)
      }, error = function(e) {
        cat("      Error: Unable to generate", metric, "plot:", e$message, "\n")
      })
    }
  }
  
  # Generate Train/Test distribution plot
  cat("  Generating Train/Test distribution plot\n")
  tryCatch({
    # Get test gene list
    test_genes <- dataset_data$condition %>% unique() %>% as.character()
    
    # Generate distribution plot
    train_test_plot <- plot_train_test_distribution(impact_df, test_genes, dataset)
    
    # Save distribution plot
    ggsave(file.path(dataset_output_path, "TRADE_train_test_plot.pdf"), 
           train_test_plot, width = 6, height = 4, dpi = 300)
    ggsave(file.path(dataset_output_path, "TRADE_train_test_plot.png"), 
           train_test_plot, width = 6, height = 4, dpi = 300)
    
  }, error = function(e) {
    cat("    Error: Unable to generate Train/Test distribution plot:", e$message, "\n")
  })
  
  cat("  Completed dataset:", dataset, "\n\n")
}

cat("All datasets processing completed!\n")
cat("Plots saved in:", output_base_path, "\n")

# Generate processing summary
cat("\nProcessing Summary:\n")
for(dataset in datasets) {
  dataset_output_path <- file.path(output_base_path, dataset)
  if(dir.exists(dataset_output_path)) {
    files <- list.files(dataset_output_path, pattern = "\\.pdf$")
    cat("Dataset", dataset, ":", length(files), "files\n")
  } else {
    cat("Dataset", dataset, ": Not processed\n")
  }
}