# Benchmarking

This directory contains the execution pipeline for our perturbation response prediction benchmark.

## ⚠️ Important Prerequisites

### 1. Install Model-Specific Environments

**CRITICAL**: Each model requires its own specific environment and dependencies. The environment provided here is intended for the benchmarking pipeline only and does NOT include the full set of dependencies required to run individual models. You **MUST** install the required packages for each model according to their official documentation **BEFORE** running this benchmark.

### 2. Environment Setup Steps

```bash
# Step 1: Create a base environment for the pipeline
conda create -n perturbation_bench python=3.8
conda activate perturbation_bench

# Step 2: Install pipeline dependencies
pip install -r requirements.txt

# Step 3: Install model-specific dependencies
# For each model you want to run, follow its official installation guide
# Example for scGen:
pip install scgen

# ... repeat for all models you want to use
```

**Note**: Some models may require conflicting dependencies. You may need to create separate conda environments for different models.

---

## 📁 Directory Structure

```
benchmarking/
├── benchmark/              # Core pipeline code
│   ├── config/            # Configuration management
│   ├── models/            # Model registry
│   ├── metrics/           # Metrics computation
│   ├── data/              # Data loading
│   ├── pipeline/          # Execution pipeline
│   └── utils/             # Utilities
├── run_benchmark.py       # Main execution script
├── requirements.txt       # Pipeline dependencies
├── config_example.yaml    # Configuration example
└── README.md              # This file
```

---

## ⚙️ Configuration

Before running, you **MUST** configure paths:

### Edit Configuration File

Copy and edit the configuration:
```bash
cp benchmark/config/datasets.yaml benchmark/config/datasets.yaml.local
nano benchmark/config/datasets.yaml
```

Update these paths:
```yaml
# Line 5-6: Path to model scripts
model_scripts_dir: "../Model_predict_code"
metrics_scripts_dir: "../Metrics_code"

# Line 10, 22, 30: Path to datasets
task1:
  data_dir: "/your/path/to/Task1_data"
task2:
  data_dir: "/your/path/to/Task2_data"
task3:
  data_dir: "/your/path/to/Task3_data"
```

### Update Model Scripts

Each model script in `Model_predict_code/` contains hardcoded paths. You need to update:
- Data input paths
- Model checkpoint paths
- Output directory paths

---

## 🚀 Usage

### List Available Tasks
```bash
python run_benchmark.py --list-tasks
```

### List Models for a Task
```bash
python run_benchmark.py --list-models task1
python run_benchmark.py --list-models task2
python run_benchmark.py --list-models task3
```

### Run a Single Model
```bash
python run_benchmark.py --task task3 --model scgen --output-dir ./results
```

### Task-Model Mapping

**Task 1** : Single perturbation prediction
- baseline1_contextmean_linear, baseline1_mlp
- biolord, gears, perturbnet, scelmo, scfoundation, scgpt

**Task 2** : Combinatorial perturbation prediction
- baseline2_contextmean_additivemean_linear, baseline2_mlp
- biolord, cpa, gears, perturbnet, scelmo, scfoundation, scgpt

**Task 3** : Cell state transfer
- baseline3_contextmean_perturbmean_linear, baseline3_mlp
- perturbnet_task3, scgen, scpram, scpregan, scvidr, squidiff, trvae


---

## ⚠️ Important Notes

### About This Pipeline

This pipeline is designed for **our benchmark evaluation workflow**. It:

1. **Wraps existing model implementations**: Does not re-implement models
2. **Requires model-specific setup**: Each model needs its dependencies installed
3. **Uses original code**: Executes the prediction scripts in `Model_predict_code/`
4. **Preserves experimental setup**: Maintains exact conditions from our experiments

### Before Running

- ✅ Install ALL required model environments
- ✅ Update paths in `benchmark/config/datasets.yaml`
- ✅ Update paths in individual model scripts (`Model_predict_code/*.py`)
- ✅ Download required datasets
- ✅ Verify test data runs successfully

### Common Issues

**"ModuleNotFoundError"**
→ Install the model-specific dependencies (see Prerequisites section)

**"FileNotFoundError: dataset not found"**
→ Update data paths in configuration file

**"Model script failed"**
→ Check that paths in the individual model script are updated
