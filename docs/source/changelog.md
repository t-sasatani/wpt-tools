# Changelog

All notable changes to this project will be documented in this file.

## [0.1.8] - 2025-10-11

### Changed
- **Naming consistency**: Renamed `solvers.py` → `solver.py` and `plotter.py` → `plotter.py` for consistency
- **Class naming**: Fixed `WPTToolsLogger` to follow proper Python conventions (WPT acronym preserved)
- **Method naming**: Fixed `print_tables()` → `print_table()` for consistency with other methods
- **Class structure**: Renamed `nw_tools` class → `NwTools` (PascalCase) with backward compatibility
- **Documentation**: Updated all imports and references to use new naming conventions

### Fixed
- **File naming**: Removed duplicate `CHANGELOG.md` and generic `mymodule.log` files
- **Test cleanup**: Removed duplicate test files after module renaming
- **Import consistency**: Updated all imports across codebase, tests, and documentation
- **Documentation links**: Fixed ReadTheDocs links and removed redundant CLI documentation

### Improved
- **Code organization**: Consistent singular module naming (`plotter.py`, `solver.py`)
- **Test suite**: Cleaned up duplicate tests, now 34 unique tests
- **Coverage**: Maintained 89% test coverage with cleaner structure
- **Linting**: All code passes ruff, black, and isort formatting

## [0.1.7] - 2025-10-11

### Changed
- Converted documentation from RST to Markdown
- Refactored CLI to use Click with workflow separation
- Enhanced test suite with regression testing
- Updated license from AGPL-3.0 to Apache 2.0
- Improved test coverage and documentation

## [0.1.6] - 2025-10-10

### Added
- Command line interface with `wpt` command
- Automated PyPI deployment with OpenID Connect
- Sphinx documentation with CLI reference
- Regression tests with snapshot validation
- PyPI deployment documentation

### Changed
- Refactored `nw_tools` to use static methods
- Improved GitHub Actions workflows
- Enhanced test coverage reporting

### Fixed
- Windows compatibility for LCR fitting
- Sphinx documentation configuration
- Test workflow parameters

## [0.1.5] - Previous Release

### Added
- Initial release with core functionality
- Basic wireless power analysis tools
- S-parameter processing capabilities
