# Test Suite Documentation

This directory contains the test suite for the modular machine learning project. The tests are written using Python's built-in `unittest` framework.

## Test Structure

The test suite consists of three main test modules and a test runner:

- `test_preprocess.py`: Tests for data preprocessing functionality
- `test_model.py`: Tests for machine learning model operations
- `test_evaluate.py`: Tests for model evaluation metrics
- `run_tests.py`: Script to run all tests

## Test Coverage

### DataPreprocessor Tests (`test_preprocess.py`)
- Tests data preprocessing pipeline
- Validates scaling operations
- Checks handling of new data
- Tests error handling for invalid inputs

### MLModel Tests (`test_model.py`)
- Tests model initialization with default/custom parameters
- Validates training and prediction functionality
- Tests model persistence (save/load operations)
- Checks error handling for invalid inputs

### ModelEvaluator Tests (`test_evaluate.py`)
- Tests evaluation metrics calculation
- Validates perfect and imperfect prediction scenarios
- Tests multi-class classification support
- Checks error handling for invalid inputs

## Running Tests

You can run the tests in several ways:
```bash
python tests/run_tests.py
```

2. Run individual test modules:
```bash
python -m unittest tests/test_preprocess.py
python -m unittest tests/test_model.py
python -m unittest tests/test_evaluate.py
```

3. Run specific test cases:
```bash
python -m unittest tests.test_preprocess.TestDataPreprocessor
```

## Test Dependencies

The test suite requires the following packages:
- numpy
- pandas
- scikit-learn (via the main project dependencies)

These dependencies are included in the project's `requirements.txt` file. 