# Contributing to OpenSports

Thank you for your interest in contributing to OpenSports! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional, for containerized development)
- Node.js 18+ (for documentation)

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/OpenSports.git
   cd OpenSports
   ```

2. **Set up the development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize the database**
   ```bash
   opensports system init
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes or new features
- **Documentation**: Improve or add documentation
- **Testing**: Add or improve test coverage
- **Performance**: Optimize existing code

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** for significant changes to discuss the approach
3. **Start small** with your first contribution
4. **Follow our coding standards** and conventions

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Your Changes

- Write clear, concise commit messages
- Follow our coding standards
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run the full test suite
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Check code quality
black opensports/
ruff check opensports/ --fix
mypy opensports/

# Run security checks
bandit -r opensports/
```

### 4. Update Documentation

- Update docstrings for new functions/classes
- Add examples for new features
- Update README.md if needed
- Add entries to CHANGELOG.md

### 5. Submit Pull Request

1. Push your branch to your fork
2. Create a pull request against the main branch
3. Fill out the pull request template
4. Wait for review and address feedback

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports, group by standard/third-party/local
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Required for all public functions

### Code Formatting

We use automated tools for consistent formatting:

```bash
# Format code
black opensports/

# Sort imports
isort opensports/

# Lint code
ruff check opensports/ --fix

# Type checking
mypy opensports/
```

### Naming Conventions

- **Classes**: PascalCase (`PlayerAnalyzer`)
- **Functions/Variables**: snake_case (`get_player_stats`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_RETRIES`)
- **Private methods**: Leading underscore (`_internal_method`)

### Documentation Standards

#### Docstring Format

```python
def analyze_player_performance(
    player_id: str,
    season: str,
    metrics: List[str]
) -> Dict[str, float]:
    """
    Analyze player performance for a given season.
    
    Args:
        player_id: Unique identifier for the player
        season: Season in format "2023-24"
        metrics: List of metrics to analyze
        
    Returns:
        Dictionary mapping metric names to values
        
    Raises:
        ValueError: If player_id is invalid
        DataNotFoundError: If season data is unavailable
        
    Example:
        >>> analyzer = PlayerAnalyzer()
        >>> stats = analyzer.analyze_player_performance(
        ...     "lebron-james", "2023-24", ["points", "assists"]
        ... )
        >>> print(stats["points"])
        25.7
    """
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── performance/    # Performance tests
├── security/       # Security tests
└── fixtures/       # Test data and fixtures
```

### Writing Tests

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark critical paths
- **Security tests**: Test for vulnerabilities

### Test Guidelines

```python
import pytest
from opensports.modeling import PlayerPerformanceModel

class TestPlayerPerformanceModel:
    """Test suite for PlayerPerformanceModel."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return PlayerPerformanceModel()
    
    def test_predict_performance_valid_input(self, model):
        """Test prediction with valid input."""
        # Arrange
        player_data = {"name": "Test Player", "stats": {...}}
        
        # Act
        result = model.predict_performance(player_data)
        
        # Assert
        assert isinstance(result, dict)
        assert "predicted_points" in result
        assert result["predicted_points"] > 0
    
    def test_predict_performance_invalid_input(self, model):
        """Test prediction with invalid input."""
        with pytest.raises(ValueError):
            model.predict_performance({})
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=opensports --cov-report=html

# Run specific test file
pytest tests/unit/test_modeling.py

# Run tests matching pattern
pytest -k "test_player"

# Run tests with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and inline comments
2. **API Documentation**: Auto-generated from docstrings
3. **User Documentation**: README, tutorials, examples
4. **Developer Documentation**: Architecture, contributing guides

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# Serve documentation locally
python -m http.server 8080 -d docs/_build/html/
```

### Documentation Guidelines

- **Be clear and concise**
- **Include examples** for complex features
- **Keep documentation up-to-date** with code changes
- **Use proper grammar and spelling**
- **Include diagrams** for complex architectures

## Release Process

### Versioning

We use Semantic Versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release branch
5. Submit pull request
6. Tag release after merge
7. Deploy to PyPI

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: nikjois@llamaearch.ai for private matters

### Resources

- [Project Documentation](https://opensports.readthedocs.io)
- [API Reference](https://opensports.readthedocs.io/api)
- [Architecture Guide](ARCHITECTURE.md)
- [Examples Repository](https://github.com/llamasearchai/opensports-examples)

## Recognition

Contributors will be recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project documentation
- Annual contributor highlights

Thank you for contributing to OpenSports! Your efforts help make sports analytics more accessible and powerful for everyone.

---

**Questions?** Feel free to reach out to the maintainers or create an issue for clarification. 