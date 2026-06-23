# Testing with VSCode and PyTest

This project is configured with comprehensive testing support for VSCode using PyTest.

## VSCode Test Explorer Integration

The project includes VSCode configuration for seamless test integration:

### Test Configuration Files:
- `.vscode/settings.json` - PyTest configuration for VSCode
- `.vscode/launch.json` - Debug configurations for testing
- `pyproject.toml` - Project-level PyTest configuration

### Test Directories:
- `tests/` - Main test directory
- `unit_tests/` - Unit test directory

### Running Tests

#### Using VSCode Test Explorer:
1. Open the Test Explorer view (Ctrl+Shift+P → "Test: Open Test Explorer")
2. Click the "Refresh Tests" button to discover tests
3. Run individual tests, test files, or all tests

#### Using VSCode Debug Configurations:
- **Python: PyTest Current File** - Run tests in the current file
- **Python: PyTest All Tests** - Run all tests in both test directories

#### Using Command Line:
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_sample.py

# Run with coverage
uv run pytest --cov=src
```

### Test Features:
- **Parametrized Tests**: Support for parameterized test cases
- **Test Discovery**: Automatic discovery in both `tests/` and `unit_tests/` directories
- **Parallel Execution**: Tests run in parallel for better performance
- **Rerun Failed Tests**: Automatic rerun of failed tests (3 attempts)
- **Detailed Output**: Clean, readable test output with short traceback format

### Sample Tests:
The `tests/test_sample.py` file contains examples of:
- Simple passing and failing tests
- Parametrized tests
- Different assertion patterns

### Configuration Details:
- PyTest path: `uv` (using uv package manager)
- Test paths: `tests`, `unit_tests`
- Parallel workers: 12
- Rerun attempts: 3
- Output format: Short traceback