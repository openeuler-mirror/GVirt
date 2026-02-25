# xlite Static Checker Documentation

## Overview

The xlite static checker provides automated code quality checks for both C++ and Python code. It helps maintain code consistency, detect potential bugs, and ensure adherence to coding standards.

## Features

### C++ Static Checks

- **Code Formatting (Clang-Format)**: Checks code formatting including:
  - Indentation (4 spaces)
  - Bracket placement
  - Line breaks
  - Trailing spaces
  - Line length (max 100 characters)
  - Comment formatting

- **Copyright Check**: Verifies that all source files contain the proper copyright header:
  ```
  * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
  ```

- **Header Guard Check**: Ensures header files have proper include guards:
  - Checks for `#ifndef` or `#pragma once`
  - Verifies matching `#define` for `#ifndef` guards

- **Static Analysis (Clang-Tidy)**: Performs comprehensive static analysis:
  - Bug detection (bugprone-* checks)
  - Security checks (cert-* checks)
  - C++ core guidelines (cppcoreguidelines-* checks)
  - Google style checks (google-* checks)
  - Modern C++ suggestions (modernize-* checks)
  - Performance optimizations (performance-* checks)
  - Readability improvements (readability-* checks)

### Python Static Checks

- **Syntax Check**: Validates Python syntax
- **Flake8**: Basic linting including:
  - PEP 8 style compliance
  - Code complexity
  - Unused imports
  - Line length (max 100 characters)
- **MyPy**: Type checking (optional)

## Usage

### Basic Usage

Run all checks (C++ and Python):

```bash
cd /workspaces/code/opencode/GVirt/xlite
python tests/run_static_checks.py
```

### Running C++ Checks Only

```bash
python tests/run_static_checks.py --cpp-only
```

### Running Python Checks Only

```bash
python tests/run_static_checks.py --python-only
```

### Advanced Options

#### Skip Specific C++ Checks

```bash
# Skip code format check
python tests/run_static_checks.py --no-cpp-format

# Skip copyright check
python tests/run_static_checks.py --no-cpp-copyright

# Skip header guard check
python tests/run_static_checks.py --no-cpp-header-guard

# Skip clang-tidy analysis
python tests/run_static_checks.py --no-cpp-tidy

# Skip multiple checks
python tests/run_static_checks.py --no-cpp-format --no-cpp-tidy
```

#### Skip Specific Python Checks

```bash
# Skip flake8 check
python tests/run_static_checks.py --no-python-flake8

# Skip mypy type check
python tests/run_static_checks.py --no-python-mypy
```

### Individual Check Scripts

You can also run the check scripts directly:

#### C++ Checks

```bash
python tests/cpp_check.py
python tests/cpp_check.py --no-tidy  # Skip clang-tidy
python tests/cpp_check.py --no-format --no-copyright  # Skip format and copyright
```

#### Python Checks

```bash
python tests/python_check.py
python tests/python_check.py --no-mypy  # Skip type checking
python tests/python_check.py --no-flake8  # Skip flake8
```

## Configuration Files

### .clang-format

Located at `xlite/.clang-format`, this file defines the C++ code formatting rules based on Google style with customizations:

- Indent width: 4 spaces
- Max line length: 100 characters
- Tab width: 4 spaces
- Various style customizations for brackets, spaces, and alignment

### .clang-tidy

Located at `xlite/.clang-tidy`, this file configures clang-tidy checks:

- Enables bugprone, cert, clang-analyzer, cppcoreguidelines, google, misc, modernize, performance, and readability checks
- Disables overly strict or incompatible checks
- Custom thresholds for function size and complexity

## Output Format

The checker provides clear output with emojis for easy identification:

- ✅ **PASSED**: Check completed successfully
- ❌ **FAILED**: Check failed with errors
- ⚠️ **WARNING**: Check passed with warnings

### Too many warnings

You can disable specific checks:
```bash
# Skip clang-tidy if it produces too many warnings
python run_static_checks.py --no-cpp-tidy

# Skip mypy type checking
python run_static_checks.py --no-python-mypy
```

## Best Practices

1. **Run checks locally** before committing changes
2. **Fix errors immediately** - don't accumulate technical debt
3. **Review warnings** - they often indicate potential issues
4. **Keep configuration updated** - adjust as project evolves
5. **Use in CI/CD** - catch issues early in development pipeline

## Contributing

When contributing to xlite:

1. Ensure all code passes static checks
2. Add copyright headers to new files
3. Use proper header guards in header files
4. Follow the configured code style
5. Address all warnings before submitting PRs

## Support

For issues or questions about the static checker:
- Check this documentation
- Review configuration files (.clang-format, .clang-tidy)
- Consult tool documentation (clang-format, clang-tidy, flake8, mypy)
