# scPerturbBench
This repository contains the code and resources for "A Systematic Comparison of Single-Cell Perturbation Response Prediction Models".
A website hosting the data and the visualization can be viewed at: luyitian.github.io/PerturbArena
<img width="1168" height="1416" alt="image" src="https://github.com/user-attachments/assets/9fe22bc9-7e68-4610-bf92-49e34134b2f0" />


## Tasks
### Task 1: Unseen Perturbation Prediction
Evaluation of model ability to predict responses to unseen single-gene perturbations.
### Task 2: Combinatorial Perturbation Prediction
Assessment of models in predicting combinatorial perturbation effects and interactions.
### Task 3: Cell state transfer
Testing model generalization across different cell types and contexts.

## Evaluation Metrics
Our benchmark employs 24 comprehensive metrics organized into four categories:
#### Absolute Accuracy
#### Relative Effect Capture
#### Differential Expression Recovery
#### Distribution Similarity

## Repository Structure
#### Benchmarking: Execution pipeline for perturbation benchmark
#### Metrics_code: Evaluation metrics implementation
#### Model_predict_code: Model prediction implementations
#### Figure: Generated figures and visualizations
#### Plots: Code for generating plots
#### .github：Continuous integration & community pull-request template

## Usage Instructions
### Single-Model Execution and Evaluation
Note: Some models may require conflicting dependencies. You may need to create separate conda environments for different models. In such cases, or if you only wish to run a single model, we recommend using the workflow below.
#### Step 1: Create a Model-Specific Environment
Create and activate a dedicated environment for the target model following its official installation instructions.
#### Step 2: Run Model Prediction
Use the corresponding scripts in `./Model_predict_code` to generate predictions.
#### Step 3: Evaluate Predictions
Compute evaluation metrics using the scripts provided in `./Metrics_code`.

### Benchmark Execution
#### Step 1: Environment Setup
Each model requires its own specific environment and dependencies. Please install the required environments for each model according to their official code repositories.
#### Step 2: Run Benchmarking
Please refer to `./Benchmarking`. Before running, you **MUST** configure the required paths.

## Acknowledge
We acknowledge the support of the Data Science Platform of Guangzhou National Laboratory and the Bio-medical Big Data Operating System (Bio-OS). We thank Lead Healthcare.AI for their help on experiments. We also acknowledge Claude (Anthropic) for assistance with GitHub repository setup and code organization.


## Citation
If you use this code or benchmark in your research, please cite:

@article{li2024systematic,
  title={A systematic comparison of single-cell perturbation response prediction models},
  author={Li, Lanxiang and You, Yue and Liao, Wenyu and Fan, Xueying and Lu, Shihong and Cao, Ye and Li, Bo and Ren, Wenle and Fu, Yunlin and Kong, Jiaming and others},
  journal={bioRxiv},
  pages={2024--12},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
