# scPerturbBench
This repository contains the code and resources for "A Systematic Comparison of Single-Cell Perturbation Response Prediction Models".
A website hosting the data and the visualization can be viewed at: luyitian.github.io/PerturbArena
<img width="1238" height="1518" alt="image" src="https://github.com/user-attachments/assets/76adb7ac-ec0c-423a-bbda-157c4bee6b5e" />

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
#### Metrics_code: Evaluation metrics implementation
#### Model_predict_code: Model prediction implementations
#### Figure: Generated figures and visualizations
#### Plots: Code for generating plots

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
