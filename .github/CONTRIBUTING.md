# Contributing to scPerturbBench

Thank you for your interest in contributing to scPerturbBench! This document provides guidelines for contributing to our single-cell perturbation prediction benchmark.

## 🎯 How to Contribute

We welcome several types of contributions:

- **🧠 New Model Implementations**: Add new perturbation prediction models
- **📊 New Datasets**: Integrate additional single-cell perturbation datasets
- **📏 Evaluation Metrics**: Contribute new evaluation metrics
- **🐛 Bug Fixes**: Fix issues in existing code
- **📚 Documentation**: Improve documentation and examples
- **⚡ Performance**: Optimize existing implementations

## 🚀 Getting Started

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/scPerturbBench.git
   cd scPerturbBench
   ```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Note: This project uses model-specific environments
# Each model in Model_predict_code/ has its own dependencies
# Install dependencies as needed for the specific model you're working with
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

## 📋 Contribution Guidelines

### Adding a New Model

When contributing a new model, please ensure:

**Required Files:**
- `Model_predict_code/your_model/model.py` - Main model implementation
- `Model_predict_code/your_model/train.py` - Training script
- `Model_predict_code/your_model/README.md` - Model-specific documentation
- `Model_predict_code/your_model/requirements.txt` - Model dependencies

**Implementation Requirements:**
- Follow the established API structure (see existing models for reference)
- Support all applicable benchmark tasks:
  - Task 1: Unseen Perturbation Prediction
  - Task 2: Combinatorial Perturbation Prediction  
  - Task 3: Cell State Transfer
- Include comprehensive evaluation on all 24 metrics
- Provide working training scripts with clear documentation

**Code Quality:**
- Write clean, well-commented code
- Include docstrings for all functions and classes
- Follow PEP 8 style guidelines
- Add appropriate error handling

### Adding a New Dataset

For dataset contributions:
- Provide preprocessing scripts
- Ensure data format consistency with existing datasets
- Include dataset metadata and documentation
- Verify licensing compatibility
- Add appropriate citations

### Adding Evaluation Metrics

For new metrics:
- Implement in `Metrics_code/`
- Follow existing metric structure
- Include comprehensive tests
- Provide clear documentation of what the metric measures
- Validate against known benchmarks

## 🧪 Testing

Before submitting your contribution:

1. **Test your model environment:**
   ```bash
   # Set up model-specific environment
   cd Model_predict_code/your_model/
   # Follow model-specific setup instructions
   
   # Test model runs successfully
   python train.py --help  # or your training script
   ```

2. **Test model predictions:**
   ```bash
   # Run your model to generate predictions
   # Verify output format is compatible with evaluation pipeline
   ```

3. **Test evaluation integration:**
   ```bash
   # Test that your predictions work with Metrics_code/
   cd ../../Metrics_code/
   # Run evaluation on your model's predictions
   ```

4. **Validate complete workflow:**
   - Model training/prediction → Metrics evaluation → Results
   - Test on a small subset of data
   - Verify all 24 metrics can be computed

## 📊 Evaluation Requirements

All model submissions must include evaluation results for:

### Absolute Accuracy Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Pearson Correlation
- Spearman Correlation

### Relative Effect Capture
- Direction Accuracy
- Magnitude Correlation
- Top-k Gene Overlap
- Rank Correlation

### Differential Expression Recovery
- DEG F1 Score
- DEG Precision/Recall
- Pathway Enrichment

### Distribution Similarity
- Wasserstein Distance
- KL Divergence
- Jensen-Shannon Divergence

## 📝 Documentation Standards

- Use clear, concise language
- Include code examples where appropriate
- Document all parameters and return values
- Provide usage examples
- Update README files as needed

## 🔄 Submission Process

1. **Create Pull Request:**
   - Use the provided PR template
   - Fill out all relevant sections
   - Include evaluation results
   - Reference any related issues

2. **Review Process:**
   - Automated CI checks must pass
   - Code review by maintainers
   - Performance validation
   - Integration testing

3. **Approval Requirements:**
   - All tests passing
   - Code quality standards met
   - Complete evaluation results provided
   - Documentation updated

## 🏗️ Development Setup

### Repository Structure
```
scPerturbBench/
├── Metrics_code/           # Evaluation metrics
├── Model_predict_code/     # Model implementations
├── Figure/                 # Generated figures
├── Plots/                  # Plotting code
└── .github/               # GitHub automation
    ├── workflows/         # CI/CD pipelines
    └── ISSUE_TEMPLATE/    # Issue templates
```

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Keep functions focused and modular
- Include type hints where appropriate
- Maximum line length: 88 characters

### Commit Messages

Use conventional commit format:
```
feat: add new transformer-based model
fix: resolve memory leak in evaluation
docs: update model integration guide
test: add unit tests for new metrics
```

## 🤝 Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and improve
- Follow the code of conduct
- Ask questions if unsure

## 🆘 Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check existing documentation first
- **Examples**: Look at existing implementations as reference

## 📜 License

By contributing to scPerturbBench, you agree that your contributions will be licensed under the same license as the project.

## 🙏 Recognition

All contributors will be acknowledged in our documentation and releases. Significant contributions may be included in academic publications (with contributor consent).

---

Thank you for helping make scPerturbBench better! 🧬✨