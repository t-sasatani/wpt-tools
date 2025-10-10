# Testing

## Regression Test Precision

Regression tests use percentage-based relative tolerances (`rtol`) for numerical comparison:

- **High precision** (`rtol=1e-6` = 0.0001%): Efficiency analysis, load sweep, wrapper parity
- **Medium precision** (`rtol=1e-3` = 0.1%): LCR fitting parameters, RXC filter values  
- **Lower precision** (`rtol=1e-2` = 1%): R² values

Relative tolerance means the allowed difference is proportional to the magnitude of the values being compared.

## Windows Compatibility

LCR fitting tests are `@pytest.mark.xfail` on Windows due to numerical precision differences in optimization algorithms. This is expected behavior.

```python
@pytest.mark.xfail(
    condition=sys.platform == "win32",
    reason="LCR fitting results differ significantly on Windows due to numerical precision differences",
    strict=False,
)
```

## Archive System

Tests use binary pickle archives (`tests/data/analysis_regression_results.pkl`) for precise numerical comparison with automatic creation and updates.
