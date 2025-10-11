# Usage

## Command Line Interface

The easiest way to get started is using the command line interface:

```bash
# Run the demo with sample data
wpt demo

# Run demo with interactive plots
wpt demo --show-plots
```

## Python API

For programmatic usage, you can import and use the modules directly:

```python
from wpt_tools.data_classes import RichNetwork
from wpt_tools.analysis import nw_tools

# Load S-parameter data
network = RichNetwork.from_touchstone("path/to/your/file.s2p")
network.set_f_target_range(target_f=6.78e6, range_f=1e6)

# Run efficiency analysis
results = nw_tools.analyze_efficiency(network, show_plot=True, rx_port=1)
```

## Examples

See the [examples directory](https://github.com/t-sasatani/wpt-tools/tree/main/examples) for Jupyter notebooks and sample data.
