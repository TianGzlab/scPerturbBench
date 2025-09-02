# Pull Request Template

## Description
Brief description of the changes in this PR.

## Type of Change
Please check the relevant option(s):

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 📈 New model implementation (adds a new perturbation prediction model)
- [ ] 📊 New dataset integration (adds support for a new dataset)
- [ ] 📏 New evaluation metric (adds a new evaluation metric)
- [ ] 🔧 Refactoring (code change that neither fixes a bug nor adds a feature)
- [ ] 📚 Documentation update
- [ ] ⚡ Performance improvement
- [ ] 🚨 Breaking change (fix or feature that would cause existing functionality to not work as expected)

## Model/Dataset Addition Checklist
*Complete this section if adding a new model or dataset:*

### For New Models:
- [ ] Model implementation is placed in `Model_predict_code/your_model/` directory
- [ ] Training script is included with clear documentation
- [ ] Model-specific environment setup instructions are provided
- [ ] Model can handle all three benchmark tasks (unseen perturbation, combinatorial, cell state transfer)
- [ ] Evaluation results are provided for all 24 metrics using `Metrics_code/`
- [ ] Model dependencies are documented in model-specific requirements or setup instructions
- [ ] Model paper/reference is cited appropriately

### For New Datasets:
- [ ] Dataset preprocessing script is included
- [ ] Data format is consistent with existing datasets
- [ ] Dataset metadata is documented
- [ ] Licensing information is provided
- [ ] Dataset source and citation are included

## Testing
- [ ] I have verified my model code runs successfully in its environment
- [ ] I have tested model predictions on sample data
- [ ] I have verified integration with the evaluation pipeline (`Metrics_code/`)
- [ ] My code follows the existing code style and conventions
- [ ] I have tested the complete workflow: model prediction → metrics evaluation

## Evaluation Results
*For model additions, please provide evaluation results:*

### Task 1: Unseen Perturbation Prediction
- [ ] Results provided for all absolute accuracy metrics
- [ ] Results provided for all relative effect capture metrics
- [ ] Results provided for all differential expression recovery metrics
- [ ] Results provided for all distribution similarity metrics

### Task 2: Combinatorial Perturbation Prediction
- [ ] Results provided (if applicable)

### Task 3: Cell State Transfer
- [ ] Results provided (if applicable)

## Documentation
- [ ] I have updated the README if necessary
- [ ] I have added appropriate code comments
- [ ] I have included example usage (if adding new functionality)

## Additional Notes
Any additional information, context, or screenshots that would help reviewers understand the changes.

## Checklist Before Submitting
- [ ] My code follows the project's coding standards
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

---
**By submitting this PR, I confirm that my contributions are made under the same license as the project and that I have read and agreed to the contributing guidelines.**