#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
import glob
import os

# Task definitions
TASK1_DATASETS = {
    "Arce_MM_CRISPRi_sub": "Maya M. Arce",
    "Adamson": "Adamson",
    "DatlingerBock2017": "Datlinger_2017",
    "DatlingerBock2021": "Datlinger_2021",
    'DixitRegev2016': 'Dixit',
    "FrangiehIzar2021_RNA": "Frangieh",
    'NormanWeissman2019_filtered': 'Norman',
    "PapalexiSatija2021_eccite_RNA": "Papalexi",
    "ReplogleWeissman2022_rpe1": "Replogle",
    'Sunshine2023_CRISPRi_sarscov2': 'Sunshine',
    "TianKampmann2021_CRISPRa": "Tian_crispra",
    "TianKampmann2021_CRISPRi": "Tian_crispri",
    "vcc_train_filtered": "Virtual Cell Challenge",
    "Junyue_Cao": "Xu",
    "Fengzhang2023": "Joung"
}

TASK3_DATASETS = {
    "Kang": "Kang",
    "Haber": "Haber",
    "Hagai": "Hagai",
    "Weinreb_time": "Weinreb",
    "Burkhardt_sub10": "Burkhardt",
    "Srivatsan_sciplex3_sub10": "Srivatsan",
    "Perturb_KHP_sub10": "Perturb_KHP",
    "Perturb_cmo_V1_sub10": "Perturb_cmo_V1",
    "Tahoe100_sub10": "Tahoe100",
    "Parse_10M_PBMC_sub10": "Parse"
}

# Task2 seen condition mapping
TASK2_SEEN_MAPPING = {
    'Arce_MM_CRISPRi_sub': {
        'seen0': ['NFKB2+KLF2', 'MYC+NFKB2', 'MYC+KLF2'],
        'seen1': ['IL2RA+NFKB2', 'BATF+NFKB2', 'LEF1+NFKB2', 'BATF+MYC', 'IRF4+NFKB2', 'IRF1+MYC', 'PRDM1+KLF2', 'NFKB2+PRDM1', 'MYC+PRDM1', 'LEF1+KLF2', 'BACH2+NFKB2', 'IRF1+KLF2', 'IRF1+NFKB2', 'BACH2+KLF2', 'BATF+KLF2', 'BACH2+MYC', 'LEF1+MYC'],
        'seen2': ['BATF+IRF1', 'BATF+PRDM1', 'IL2RA+PRDM1', 'IRF1+LEF1']
    },
    'Sunshine2023_CRISPRi_sarscov2': {
        'seen0': ['TMPRSS2+ACE2', 'IFNAR2+IFNAR1'],
        'seen1': ['FURIN+TMPRSS2', 'TMPRSS2+CTSL', 'BRD2+BRD4', 'CTSB+TMPRSS2', 'GPR89B+GPR89A', 'FURIN+ACE2', 'CCZ1+CCZ1B', 'RELB+RELA', 'IRF3+IRF9', 'COG6+COG5'],
        'seen2': ['SLC35B2+B3GALT6', 'TMEM97+SIGMAR1']
    },
    'NormanWeissman2019_filtered': {
        'seen0': ['ETS2+CNN1', 'ETS2+PRTG', 'CBL+CNN1', 'CBL+PTPN9', 'ZBTB10+ELMSAN1', 'CDKN1C+CDKN1B', 'ZBTB10+DLX2', 'CEBPB+CEBPA'],
        'seen1': ['DUSP9+ETS2', 'LHX1+ELMSAN1', 'MAP2K3+ELMSAN1', 'MAPK1+PRTG', 'BCL2L11+TGFBR2', 'ETS2+MAPK1', 'MAP2K6+ELMSAN1', 'ETS2+IKZF3', 'ETS2+IGDCC3', 'CEBPB+MAPK1', 'CBL+UBASH3B', 'UBASH3B+CNN1', 'CNN1+MAPK1', 'PTPN12+PTPN9', 'SGK1+S1PR2', 'ETS2+CEBPE', 'CBL+PTPN12', 'CEBPB+PTPN12', 'UBASH3B+PTPN9', 'ETS2+MAP7D1', 'RHOXF2+SET', 'TGFBR2+ETS2', 'KLF1+CEBPA', 'MAP2K6+SPI1', 'DUSP9+PRTG', 'TGFBR2+PRTG', 'ZBTB10+PTPN12', 'RHOXF2+ZBTB25', 'CEBPE+CNN1', 'CEBPB+OSR2', 'LYL1+CEBPB', 'CBL+TGFBR2', 'CEBPE+CEBPA', 'CEBPE+SPI1', 'CNN1+UBASH3A', 'BCL2L11+BAK1', 'IGDCC3+PRTG', 'CDKN1B+CDKN1A', 'SNAI1+DLX2', 'CDKN1C+CDKN1A', 'CEBPE+CEBPB', 'ZBTB10+SNAI1', 'KIF18B+KIF2C', 'ZC3HAV1+CEBPA', 'PLK4+STIL', 'FOSB+CEBPB', 'CBL+UBASH3A', 'JUN+CEBPB', 'JUN+CEBPA'],
        'seen2': ['AHR+KLF1', 'CEBPE+PTPN12', 'CEBPE+RUNX1T1', 'DUSP9+MAPK1', 'FEV+CBFA2T3', 'FEV+ISL2', 'FOSB+OSR2', 'FOSB+UBASH3B', 'FOXA1+FOXL2', 'FOXA3+HOXB9', 'FOXL2+HOXB9', 'FOXL2+MEIS1', 'KLF1+MAP2K6', 'MAP2K3+SLC38A2', 'PTPN12+SNAI1', 'SNAI1+UBASH3B', 'TMSB4X+BAK1', 'ZNF318+FOXL2']
    },
    'DixitRegev2016': {
        'seen0': ['NR2C2+IRF1', 'NR2C2+ETS1', 'IRF1+ETS1'],
        'seen1': ['NR2C2+ELF1', 'IRF1+ELF1', 'ETS1+ELF1', 'NR2C2+ELK1', 'CREB1+ETS1', 'EGR1+ETS1', 'ELK1+ETS1', 'EGR1+NR2C2', 'NR2C2+E2F4', 'ETS1+GABPA', 'EGR1+IRF1', 'NR2C2+GABPA', 'NR2C2+CREB1', 'CREB1+IRF1', 'IRF1+ELK1', 'YY1+ETS1', 'IRF1+E2F4', 'NR2C2+YY1', 'ETS1+E2F4', 'IRF1+GABPA'],
        'seen2': ['CREB1+YY1', 'EGR1+EGR1', 'EGR1+ELF1', 'EGR1+GABPA', 'EGR1+YY1', 'YY1+ELK1', 'YY1+GABPA']
    },
    'Fengzhang2023': {
        'seen0': [],
        'seen1': ['ZNF559+ZNF177'],
        'seen2': []
    }
}

def refine_model_name(model, task):
    """Refine model names based on task category"""
    if task == 'Task1' and model.startswith('Baseline'):
        return 'Baseline_1'
    elif task.startswith('Task2') and model.startswith('Baseline'):
        return 'Baseline_2'
    elif task == 'Task3' and model.startswith('Baseline'):
        return 'Baseline_3'
    elif model.startswith('CPA'):
        return 'CPA'
    else:
        return model
    
def assign_task_category(dataset, condition=None):
    """Assign task category based on dataset and condition"""
    # First check if this is a Task2 dataset with seen splits
    if dataset in TASK2_SEEN_MAPPING:
        if condition:
            # Check which seen category this condition belongs to
            for seen_type, conditions in TASK2_SEEN_MAPPING[dataset].items():
                if condition in conditions:
                    return f'Task2'
            # If condition not found in any seen category, return general Task2
            return 'Task1'
        else:
            # No condition provided, return general Task2
            return 'Task2'
    
    # Task 1 datasets (single genetic perturbations)  
    elif dataset in TASK1_DATASETS:
        return 'Task1'
    
    # Task 3 datasets (chemical perturbations)
    elif dataset in TASK3_DATASETS:
        return 'Task3'
    
    else:
        return 'Other'

def calculate_ccc(y_true, y_pred):
    """Calculate Concordance Correlation Coefficient"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(mask) < 2:
        return np.nan
    
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Calculate means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    # Calculate variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # Calculate covariance
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # Calculate CCC
    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(mask) < 2:
        return np.nan
    
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Avoid division by zero
    mask_nonzero = y_true != 0
    if np.sum(mask_nonzero) == 0:
        return np.nan
    
    y_true_nz = y_true[mask_nonzero]
    y_pred_nz = y_pred[mask_nonzero]
    
    mape = np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100
    return mape

def process_csv_files():
    """Process all CSV files and calculate correlation metrics"""
    
    # Find all *deg_overlap_results.csv files
    csv_files = glob.glob("../accuracy_updated_csv/*deg_overlap_results.csv")
    
    results = []
    
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            if 'NsigDEG_true' not in df.columns or 'NsigDEG_pred' not in df.columns:
                print(f"Warning: Required columns not found in {csv_file}")
                continue
            
            # Extract model name from filename
            model = csv_file.replace('_deg_overlap_results.csv', '').replace('../accuracy_updated_csv/', '')
            
            # Group by dataset and calculate metrics per task/model combination
            if 'dataset' in df.columns:
                # First, assign task categories to each row based on dataset and condition
                conditions_to_check = []
                if 'condition' in df.columns:
                    conditions_to_check = df['condition'].tolist()
                
                # Calculate metrics grouped by dataset and task
                for dataset_name, dataset_df in df.groupby('dataset'):
                    # For Task2 datasets, further group by seen splits
                    if dataset_name in TASK2_SEEN_MAPPING:
                        if 'condition' in df.columns:
                            # Group by task category (seen0, seen1, seen2)
                            task_groups = {}
                            for _, row in dataset_df.iterrows():
                                condition = row['condition']
                                task = assign_task_category(dataset_name, condition)
                                if task not in task_groups:
                                    task_groups[task] = []
                                task_groups[task].append(row)
                            
                            # Calculate metrics for each task group
                            for task, rows in task_groups.items():
                                if len(rows) == 0:
                                    continue
                                
                                rows_df = pd.DataFrame(rows)
                                y_true = rows_df['NsigDEG_true'].values
                                y_pred = rows_df['NsigDEG_pred'].values
                                
                                # Remove rows where either value is NaN
                                # mask = ~(pd.isna(y_true) | pd.isna(y_pred))
                                mask = ~(pd.isna(y_true) | pd.isna(y_pred) | (y_true == 0) | (y_pred == 0))
                                y_true_clean = y_true[mask]
                                y_pred_clean = y_pred[mask]
                                
                                if len(y_true_clean) < 5:
                                    print(f"Warning: Not enough valid data points for {model}/{dataset_name}/{task}")
                                    continue
                                
                                # Calculate metrics
                                try:
                                    spearman_r, spearman_p = spearmanr(y_true_clean, y_pred_clean)
                                except:
                                    spearman_r, spearman_p = np.nan, np.nan
                                
                                try:
                                    pearson_r, pearson_p = pearsonr(y_true_clean, y_pred_clean)
                                except:
                                    pearson_r, pearson_p = np.nan, np.nan
                                
                                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                                mape = calculate_mape(y_true_clean, y_pred_clean)
                                ccc = calculate_ccc(y_true_clean, y_pred_clean)
                                
                                result = {
                                    'dataset': dataset_name,
                                    'model': model,
                                    'Task': task,
                                    'n_samples': len(y_true_clean),
                                    'spearman_r': spearman_r,
                                    'spearman_p': spearman_p,
                                    'pearson_r': pearson_r,
                                    'pearson_p': pearson_p,
                                    'mae': mae,
                                    'mape': mape,
                                    'ccc': ccc
                                }
                                result['model'] = refine_model_name(result['model'], result['Task'])
                                results.append(result)
                        else:
                            # No condition column, treat as mixed Task2
                            task = assign_task_category(dataset_name)
                            y_true = dataset_df['NsigDEG_true'].values
                            y_pred = dataset_df['NsigDEG_pred'].values
                            
                            # Remove rows where either value is NaN
                            # mask = ~(pd.isna(y_true) | pd.isna(y_pred))
                            mask = ~(pd.isna(y_true) | pd.isna(y_pred) | (y_true == 0) | (y_pred == 0))
                            y_true_clean = y_true[mask]
                            y_pred_clean = y_pred[mask]
                            
                            if len(y_true_clean) < 5:
                                print(f"Warning: Not enough valid data points for {model}/{dataset_name}")
                                continue
                            
                            # Calculate metrics
                            try:
                                spearman_r, spearman_p = spearmanr(y_true_clean, y_pred_clean)
                            except:
                                spearman_r, spearman_p = np.nan, np.nan
                            
                            try:
                                pearson_r, pearson_p = pearsonr(y_true_clean, y_pred_clean)
                            except:
                                pearson_r, pearson_p = np.nan, np.nan
                            
                            mae = mean_absolute_error(y_true_clean, y_pred_clean)
                            mape = calculate_mape(y_true_clean, y_pred_clean)
                            ccc = calculate_ccc(y_true_clean, y_pred_clean)
                            
                            result = {
                                'dataset': dataset_name,
                                'model': model,
                                'Task': task,
                                'n_samples': len(y_true_clean),
                                'spearman_r': spearman_r,
                                'spearman_p': spearman_p,
                                'pearson_r': pearson_r,
                                'pearson_p': pearson_p,
                                'mae': mae,
                                'mape': mape,
                                'ccc': ccc
                            }
                            result['model'] = refine_model_name(result['model'], result['Task'])
                            results.append(result)
                    else:
                        # Task1 or Task3 datasets - calculate metrics for entire dataset
                        task = assign_task_category(dataset_name)
                        y_true = dataset_df['NsigDEG_true'].values
                        y_pred = dataset_df['NsigDEG_pred'].values
                        
                        # Remove rows where either value is NaN
                        # mask = ~(pd.isna(y_true) | pd.isna(y_pred))
                        mask = ~(pd.isna(y_true) | pd.isna(y_pred) | (y_true == 0) | (y_pred == 0))
                        y_true_clean = y_true[mask]
                        y_pred_clean = y_pred[mask]
                        
                        if len(y_true_clean) < 5:
                            print(f"Warning: Not enough valid data points for {model}/{dataset_name}")
                            continue
                        
                        # Calculate metrics
                        try:
                            spearman_r, spearman_p = spearmanr(y_true_clean, y_pred_clean)
                        except:
                            spearman_r, spearman_p = np.nan, np.nan
                        
                        try:
                            pearson_r, pearson_p = pearsonr(y_true_clean, y_pred_clean)
                        except:
                            pearson_r, pearson_p = np.nan, np.nan
                        
                        mae = mean_absolute_error(y_true_clean, y_pred_clean)
                        mape = calculate_mape(y_true_clean, y_pred_clean)
                        ccc = calculate_ccc(y_true_clean, y_pred_clean)
                        
                        result = {
                            'dataset': dataset_name,
                            'model': model,
                            'Task': task,
                            'n_samples': len(y_true_clean),
                            'spearman_r': spearman_r,
                            'spearman_p': spearman_p,
                            'pearson_r': pearson_r,
                            'pearson_p': pearson_p,
                            'mae': mae,
                            'mape': mape,
                            'ccc': ccc
                        }
                        result['model'] = refine_model_name(result['model'], result['Task'])
                        results.append(result)
            
            else:
                # No dataset column, calculate for entire file
                y_true = df['NsigDEG_true'].values
                y_pred = df['NsigDEG_pred'].values
                
                # Remove rows where either value is NaN
                # mask = ~(pd.isna(y_true) | pd.isna(y_pred))
                mask = ~(pd.isna(y_true) | pd.isna(y_pred) | (y_true == 0) | (y_pred == 0))
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]
                
                if len(y_true_clean) < 5:
                    print(f"Warning: Not enough valid data points in {csv_file}")
                    continue
                
                # Calculate metrics
                try:
                    spearman_r, spearman_p = spearmanr(y_true_clean, y_pred_clean)
                except:
                    spearman_r, spearman_p = np.nan, np.nan
                
                try:
                    pearson_r, pearson_p = pearsonr(y_true_clean, y_pred_clean)
                except:
                    pearson_r, pearson_p = np.nan, np.nan
                
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                mape = calculate_mape(y_true_clean, y_pred_clean)
                ccc = calculate_ccc(y_true_clean, y_pred_clean)
                
                result = {
                    'dataset': 'all',
                    'model': model,
                    'Task': 'Other',
                    'n_samples': len(y_true_clean),
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'mae': mae,
                    'mape': mape,
                    'ccc': ccc
                }
                result['model'] = refine_model_name(result['model'], result['Task'])
                results.append(result)
                
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    # Refine model names in the final DataFrame
    results_df['model'] = results_df.apply(lambda row: refine_model_name(row['model'], row['Task']), axis=1)
    # Save results
    output_file = 'deg_correlation_metrics_noseensplit_nozeros.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Processed {len(results)} dataset/model/condition combinations")
    
    # Display summary statistics
    if not results_df.empty:
        print("\nTask distribution:")
        print(results_df['Task'].value_counts())
        print("\nSummary statistics:")
        print(results_df.describe())
    else:
        print("No results to display.")
    
    return results_df

if __name__ == "__main__":
    results = process_csv_files()