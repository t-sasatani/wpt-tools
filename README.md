# Wireless Power Tools

⚠️ **Alpha Software** - This project is in active development. Features and APIs may change without notice.

[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/t-sasatani/wpt-tools)
[![CI](https://github.com/t-sasatani/wpt-tools/actions/workflows/format.yml/badge.svg)](https://github.com/t-sasatani/wpt-tools/actions/workflows/format.yml)
![coverage](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Ft-sasatani%2Fwpt-tools%2Fcoverage-data%2Fcoverage%2Fsummary.yml&query=%24.coverage.percent&label=coverage)
[![PyPI version](https://badge.fury.io/py/wpt-tools.svg)](https://badge.fury.io/py/wpt-tools)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

2-port parameter based analysis tools for wireless power transfer systems

## Features

- LCR circuit fitting and modeling
- Z-parameter visualization
- Efficiency calculation with load matching
- Receiver-side capacitor filter design
- Load sweep analysis (input/output power, efficiency)
- Plotting and visualization

## Installation

```bash
pip install wpt-tools
```

## Quick Start

```bash
# Run the demo
wpt demo
```

## Documentation

- **Full documentation**: [https://wpt-tools.readthedocs.io/](https://wpt-tools.readthedocs.io/)
- **PyPI package**: [https://pypi.org/project/wpt-tools/](https://pypi.org/project/wpt-tools/)