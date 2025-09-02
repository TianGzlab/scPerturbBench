# .github Directory

This directory contains GitHub-specific configuration files that enable continuous integration, community contributions, and automated testing for the scPerturbBench repository.

## 📁 Directory Structure

```
.github/
├── workflows/              # GitHub Actions CI/CD pipelines
│   └── ci.yml             # Main CI workflow
├── ISSUE_TEMPLATE/         # Issue templates for community
│   ├── bug_report.yml     # Bug report template
│   ├── feature_request.yml # Feature request template
│   └── model_submission.yml # Model submission template
├── pull_request_template.md # PR template for contributions
├── CONTRIBUTING.md         # Contribution guidelines
└── README.md              # This file
```

## 🚀 Continuous Integration (CI)

### Workflow: `ci.yml`

**Triggers:**
- Pushes to `main` and `develop` branches
- Pull requests targeting `main` and `develop`

**Testing Matrix:**
- Python version: 3.9
- OS: Ubuntu Latest

**Checks Performed:**
1. **Repository Structure**: Validate required directories exist
2. **Code Quality**: Basic Python syntax validation
3. **Model Discovery**: List available models and metrics
4. **Community Setup**: Verify GitHub templates
5. **Integration Tests**: Basic path configuration verification

**Validation Steps:**
- Validate directory structure (Metrics_code, Model_predict_code, Figure, Plots)
- Run Python syntax checks with flake8
- Discover and list Python files in key directories  
- Verify community templates are present
- Test basic Python path configuration (no dependency installation required)

## 🤝 Community Templates

### Issue Templates

1. **Bug Report** (`bug_report.yml`)
   - Structured bug reporting with environment details
   - Component-specific categorization
   - Reproduction steps and error logging

2. **Feature Request** (`feature_request.yml`)
   - Feature categorization (model, dataset, metric, etc.)
   - Use case motivation and detailed specifications
   - Priority assessment

3. **Model Submission** (`model_submission.yml`)
   - Comprehensive model contribution checklist
   - Performance requirements and evaluation results
   - Integration and documentation requirements

### Pull Request Template

The PR template (`pull_request_template.md`) provides:
- Change type classification
- Specific checklists for model/dataset additions
- Testing and evaluation requirements
- Documentation standards
- Pre-submission verification

## 📋 Contribution Workflow

1. **Issue Creation**: Use appropriate template for bugs, features, or model submissions
2. **Development**: Follow guidelines in `CONTRIBUTING.md`
3. **Pull Request**: Use PR template with complete checklist
4. **CI Validation**: Automated testing on multiple Python versions
5. **Review Process**: Community and maintainer review
6. **Integration**: Merge after passing all checks

## 🔧 Setting Up Local Development

To work with the CI setup locally:

```bash
# Install basic linting tools
pip install flake8

# Run linting (same as CI)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=__pycache__,.git
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics --exclude=__pycache__,.git

# Test basic path configuration (same as CI)
python -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'Model_predict_code'))
sys.path.insert(0, os.path.join(os.getcwd(), 'Metrics_code'))
print('✅ Python path configuration successful')
"

# Validate repository structure
ls -la Metrics_code Model_predict_code Figure Plots
```

## 🎯 Benefits for Community

This GitHub automation provides:

1. **Quality Assurance**: Automated testing prevents broken contributions
2. **Consistency**: Templates ensure complete information for issues and PRs
3. **Accessibility**: Clear guidelines lower contribution barriers
4. **Maintainability**: Structured process scales with community growth
5. **Transparency**: Public CI results build trust and reliability

## 📈 Continuous Improvement

The automation supports the benchmark's goal of being a "living" resource by:
- Streamlining new model additions
- Ensuring integration quality
- Facilitating community engagement
- Maintaining code quality standards
- Enabling rapid iteration and improvement

## 🚀 Future Enhancements

Potential additions to this automation:
- Performance benchmarking workflows
- Automated result validation
- Documentation generation
- Release automation
- Model comparison reports

---

This automation infrastructure ensures scPerturbBench remains a high-quality, community-driven benchmark that can evolve with the rapidly advancing field of single-cell perturbation prediction.