# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- **RXC filter design**: `compute_rxc_filter()` resonated the fitted receiver inductance
  (`Im(Z_network + j*w*Lrx) -> 0`) instead of presenting the maximum-efficiency load
  `R_opt + j*X_opt` at the coil terminals. `X_opt` already includes the coil reactance, so
  the inductance was double counted and the design missed the optimum badly on self-tuned
  coils (0.4% vs 52% efficiency on the bundled sample, port 2). The network is now solved
  against `R_opt + j*X_opt` directly, with a closed-form `Cp`/`Cs` solution
  (`solve_cpcs_closed_form()`) used as the optimiser seed. A warning is logged when the
  `Cp`/`Cs` topology cannot realise the target (e.g. inductive `X_opt`).
  `RXCFilterResults.lrx` is kept for reference but no longer enters the design.
  The archived `rxc_filter` regression values were regenerated accordingly.

### Added
- Tests verifying the RXC design through independent routes: network impedance equals the
  optimal load, an algebraic closed form, an ABCD-matrix circuit simulation, and direct
  maximisation of the simulated efficiency.

## [0.1.11] - 2026-10-14

### Fixed
- **Optimal load reactance**: `efficiency_calculator()` computed `x_opt` as `Rm*Xm - X22`
  because of an operator-precedence slip. It now returns the correct maximum-efficiency
  load reactance `Rm*Xm/R11 - X22`. `max_eff_opt` and `r_opt` were unaffected. The error
  was invisible for purely inductive coupling (`Rm = 0`) and grows with `Rm*Xm`.
  The archived `max_x_opt` regression values were regenerated accordingly.

### Added
- Regression tests checking `x_opt` against the closed form, a brute-force efficiency
  maximum and the `rx_port` swap, using a measured network with non-zero `Rm`.

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
