```markdown
# pw-plan-cleaner-01 Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill documents the development patterns and workflows for the `pw-plan-cleaner-01` Python project. The repository focuses on data cleaning features, with a simple UI and a modular backend. It uses conventional commits, maintains a consistent code style, and includes tests for pipeline stages. This guide will help you contribute new features, update the UI, and maintain code quality in line with established practices.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for Python files and modules.  
  *Example:*  
  ```
  cleaner/data_cleaner.py
  tests/test_pipeline_stages.py
  ```

- **Import Style:**  
  Use relative imports within modules.  
  *Example:*  
  ```python
  from .utils import clean_text
  ```

- **Export Style:**  
  Use named exports (explicitly define what is exported from a module).  
  *Example:*  
  ```python
  def clean_data(...):
      ...
  __all__ = ["clean_data"]
  ```

- **Commit Messages:**  
  Follow [Conventional Commits](https://www.conventionalcommits.org/) with `feat` and `fix` prefixes.  
  *Example:*  
  ```
  feat: add advanced whitespace cleaner to pipeline
  fix: correct UI layout for cleaner options
  ```

## Workflows

### Feature Addition with UI and Tests
**Trigger:** When adding a new processing mode or feature to the application  
**Command:** `/add-feature-with-tests`

1. **Create new processing logic**  
   - Add a new module under `cleaner/` for the feature.  
   - *Example:*  
     ```
     cleaner/new_mode.py
     ```
2. **Update the UI**  
   - Edit `app.py` to add controls and integrate the new feature into the interface.
   - *Example:*  
     ```python
     # In app.py
     from cleaner.new_mode import process_new_mode
     # Add UI elements and hook up processing
     ```
3. **Write or update tests**  
   - Add or update tests in `tests/test_pipeline_stages.py` to cover the new feature.
   - *Example:*  
     ```python
     def test_new_mode():
         result = process_new_mode("input")
         assert result == "expected"
     ```
4. **Commit your changes**  
   - Use a conventional commit message:
     ```
     feat: add new_mode processing with UI and tests
     ```

### UI Restyle or Fix
**Trigger:** When updating the look and feel or fixing UI bugs  
**Command:** `/restyle-ui`

1. **Edit `app.py`**  
   - Adjust styles, layout, or component behavior as needed.
   - *Example:*  
     ```python
     # Change layout or add style tweaks in app.py
     ```
2. **Test visually**  
   - Run the app and confirm the UI changes work as intended.
3. **Commit your changes**  
   - Use a descriptive conventional commit message:
     ```
     fix: improve button alignment in main UI
     ```

## Testing Patterns

- **Test File Naming:**  
  Test files follow the pattern `*.test.*` or are placed in the `tests/` directory.  
  *Example:*  
  ```
  tests/test_pipeline_stages.py
  ```

- **Framework:**  
  The specific testing framework is not specified, but tests are written as functions (compatible with `pytest`).

- **Example Test:**  
  ```python
  def test_clean_data_removes_empty_lines():
      input_data = "line1\n\nline2"
      result = clean_data(input_data)
      assert result == "line1\nline2"
  ```

## Commands

| Command                  | Purpose                                               |
|--------------------------|-------------------------------------------------------|
| /add-feature-with-tests  | Add a new processing feature with UI and tests        |
| /restyle-ui              | Update or fix the UI layout and styles                |
```
