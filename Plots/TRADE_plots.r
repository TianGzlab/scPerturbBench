library(tidyverse)
library(reshape2)
library(gridExtra)
library(patchwork)
library(viridis)
library(ggrepel)


COLOR_MAP <- c(
  "NoPerturb" =  "#BA5B59",
  "Baseline1_Linear" = "#A09898",
  "Baseline1_ContextMean" = "#B6818E",
  "Baseline1_PerturbMean" = "#769A9B",
  "Baseline1_AdditiveMean" = "#829091",
  "Baseline1_MLP" =  "#866AA3",
  "CPA" = "#708090",
  'GEARS' =  '#F6DEA4',
  'scELMO' =  '#E58760',
  "PerturbNet" = "#759971",
  "scGPT" =  "#6CB3DA",
  "scGPT_epoch5" =  "#C9DEE5",
  "scFoundation" =  "#92A8D3",
  "Biolord" =  "#869FA1"
)

MODEL_DISPLAY_MAPPING <- c(
  'NoPerturb' =  'NoPerturb',
  'Baseline1_Linear' =  'Baseline1_Linear',
  'Baseline1_ContextMean' =  'Baseline1_ContextMean',
  'Baseline1_PerturbMean' =  'Baseline1_PerturbMean',
  'Baseline1_AdditiveMean' =  'Baseline1_AdditiveMean',
  'Baseline1_MLP' =  'Baseline1_MLP',
  'CPA' =  'CPA',
  'GEARS' =  'GEARS',
  'scELMO' =  'scELMO',
  'PerturbNet' =  'PerturbNet',
  'scGPT' =  'scGPT',
  'scFoundation' =  'scFoundation',
  'Biolord' =  'Biolord'
)

base_path <- "/data2/lanxiang/perturb_benchmark_v2/SA_review/Round1/Metrics_out/Table/Task1_metrics"
trade_base_path <- "/data2/lanxiang/perturb_benchmark_v2/model/Trade"
output_base_path <- "/data2/lanxiang/perturb_benchmark_v2/SA_review/Round1/Fig/Fig2/Trade_plot"


dir.create(output_base_path, showWarnings = FALSE, recursive = TRUE)


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


categories <- c("Allgenes", "Top100", "Top50", "Top20")


metrics <- c('R_squared', 'Pearson_Correlation', "Pearson_Correlation_delta", 'Cosine_Similarity',
             'MSE', 'RMSE', 'MAE', 'L2', 'MMD', 'Wasserstein')


plot_single_metric_trend <- function(data, metric_name, category_name, dataset_name) {
  data_filtered <- data %>% 
    filter(category == category_name, Metric == metric_name) %>%
    mutate(condition_label = as.character(condition))
  

  p <- ggplot(data_filtered, aes(x = condition, y = Value, color = model_display, group = model_display)) +
    geom_line(size = 1.2, alpha = 0.8) +
    geom_point(size = 3, alpha = 0.9) +
    scale_color_manual(values = COLOR_MAP, name = "Model") +
    theme_minimal() +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      axis.line = element_line(color = "black", size = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, color = "black", face = "bold"), 
      axis.text.y = element_text(size = 12, color = "black", face = "bold"),  
      axis.title.x = element_text(size = 14, color = "black", margin = margin(t = 10), face = "bold"),  
      axis.title.y = element_text(size = 14, color = "black", margin = margin(r = 10), face = "bold"), 
      plot.title = element_text(size = 14, hjust = 0.5, color = "black", face = "bold"),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "grey40"),
      legend.position = "bottom",
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 10),
      legend.box.background = element_blank(),
      legend.box.margin = margin(t = 10),
      plot.margin = margin(20, 20, 20, 20),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    ) +
    labs(
      title = paste(metric_name, "Performance Across Perturbation Genes -", dataset_name),
      subtitle = paste("Category:", category_name, "| Genes ordered by decreasing perturbation strength"),
      x = "Perturbation Gene (High → Low Impact)",
      y = metric_name,
      color = "Model"
    )
  
  return(p)
}

# Train/Test distribution plot
plot_train_test_distribution <- function(impact_df, test_genes, dataset_name) {
  impact_df_single <- impact_df %>%
    filter(!str_detect(perturb_gene, "\\+"))
  

  impact_df_single <- impact_df_single %>%
    mutate(split = if_else(perturb_gene %in% test_genes, "test", "train"))

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
      legend.position = "none" 
    )
  
  return(p)
}


for(dataset in datasets) {
  cat("Processing dataset:", dataset, "\n")
  

  dataset_output_path <- file.path(output_base_path, dataset)
  dir.create(dataset_output_path, showWarnings = FALSE, recursive = TRUE)
  

  impact_file <- file.path(trade_base_path, dataset, "transcriptome_wide_impact_summary.csv")

  if(!file.exists(impact_file)) {
    cat(" Warning: Impact file does not exist, skipping dataset", dataset, "\n")
    next
  }
  
  impact_df <- read_csv(impact_file)
  
 
  impact_df_sorted <- impact_df %>%
    mutate(
      perturb_gene = str_replace(perturb_gene, "\\+ctrl", ""),
      transcriptome_wide_impact = as.numeric(transcriptome_wide_impact)
    ) %>%
    arrange(desc(transcriptome_wide_impact))
  
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
  
  if(length(data_list) == 0) {
    cat("  Warning: No data files found, skipping dataset", dataset, "\n")
    next
  }
  

  all_data <- bind_rows(data_list)
  

  dataset_data <- all_data %>% 
    filter(tolower(dataset) == tolower(!!dataset)) %>%  

    filter(model != "scGPT_epoch5") %>%
    mutate(model = case_when(
      model == "scGPT_epoch2" ~ "scGPT",
      TRUE ~ model
    )) %>%

    mutate(model_display = case_when(
      model %in% names(MODEL_DISPLAY_MAPPING) ~ MODEL_DISPLAY_MAPPING[model],
      TRUE ~ model
    ))
  

  if(nrow(dataset_data) == 0) {
    cat("  Warning: No matching data found, skipping dataset", dataset, "\n")
    next
  }
  

  dataset_data <- dataset_data %>%
    mutate(condition = factor(condition, levels = impact_df_sorted$perturb_gene))
  

  dataset_long <- dataset_data %>%
    pivot_longer(cols = all_of(metrics),
                 names_to = "Metric",
                 values_to = "Value")
  

  dataset_long$Metric <- factor(dataset_long$Metric, levels = metrics)
  

  for(category in categories) {
    if(!category %in% dataset_long$category) next
    
    cat("  Processing category:", category, "\n")
    
    for(metric in metrics) {
      metric_data <- dataset_long %>% 
        filter(category == category, Metric == metric)
      
      if(nrow(metric_data) == 0) next
      
      cat("    Generating metric:", metric, "\n")
      
      tryCatch({
        p <- plot_single_metric_trend(dataset_long, metric, category, dataset)
        
        filename <- paste0("trend_", category, "_", metric, ".pdf")
        ggsave(file.path(dataset_output_path, filename), 
               p, width = 10, height = 8, dpi = 300)
      }, error = function(e) {
        cat("      Error: Unable to generate", metric, "plot:", e$message, "\n")
      })
    }
  }
  

  cat("  Generating Train/Test distribution plot\n")
  tryCatch({

    test_genes <- dataset_data$condition %>% unique() %>% as.character()
    

    train_test_plot <- plot_train_test_distribution(impact_df, test_genes, dataset)
    

    ggsave(file.path(dataset_output_path, "TRADE_train_test_plot.pdf"), 
           train_test_plot, width = 6, height = 4, dpi = 300)
    ggsave(file.path(dataset_output_path, "TRADE_train_test_plot.png"), 
           train_test_plot, width = 6, height = 4, dpi = 300)
    
  }, error = function(e) {
    cat("    Error: Unable to generate Train/Test distribution plot:", e$message, "\n")
  })
  
  cat("  Error: Unable to generate Train/Test distribution plot:", dataset, "\n\n")
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
