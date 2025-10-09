"""
Example script for using wpt-tools.
"""
from wpt_tools.data_classes import RichNetwork
from wpt_tools.analysis import nw_tools

if __name__ == "__main__":
    example_nw = RichNetwork.from_touchstone('./assets/sample.s2p')
    example_nw.set_f_target_range(target_f=6.78e6, range_f=1e6)

    results = nw_tools.analyze_efficiency(
        rich_nw=example_nw,
        show_plot=True,
        rx_port=1
        )

    # For maximum efficiency analysis
    target_f = 6.78e6
    range_f = 1e6

    ls1, cs1, rs1, ls2, cs2, rs2, lm = nw_tools.fit_z_narrow(
        rich_nw=example_nw,
        show_plot=True,
    )
