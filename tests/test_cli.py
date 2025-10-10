"""Tests for CLI functionality."""

import pytest
from unittest.mock import patch, MagicMock
from wpt_tools.cli.main import cli, demo
from wpt_tools.workflow import demo_workflow


class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_structure(self):
        """Test that CLI is properly structured."""
        assert cli is not None
        assert hasattr(cli, 'commands')
        assert demo is not None
        assert hasattr(demo, 'name')
        assert demo.name == 'demo'
        
        # Check if version option is present
        assert hasattr(cli, 'params')
        version_params = [p for p in cli.params if hasattr(p, 'opts') and '--version' in p.opts]
        assert len(version_params) > 0


class TestWorkflowModule:
    """Test the workflow module functionality."""
    
    @patch('wpt_tools.workflow.plot_impedance')
    @patch('wpt_tools.workflow.RichNetwork')
    @patch('wpt_tools.workflow.nw_tools')
    def test_demo_workflow(self, mock_nw_tools, mock_rich_network, mock_plot_impedance):
        """Test demo workflow execution with and without plots."""
        # Mock the network and tools
        mock_nw_instance = MagicMock()
        mock_rich_network.from_touchstone.return_value = mock_nw_instance
        
        # Test without plots
        demo_workflow(show_plot=False)
        mock_plot_impedance.assert_not_called()
        
        # Test with plots
        demo_workflow(show_plot=True)
        mock_plot_impedance.assert_called_once()
        
        # Verify the analysis was called
        assert mock_rich_network.from_touchstone.call_count == 2
        assert mock_nw_instance.set_f_target_range.call_count == 2
        assert mock_nw_tools.analyze_efficiency.call_count == 2
        assert mock_nw_tools.fit_z_narrow.call_count == 2
        assert mock_nw_tools.sweep_load.call_count == 2
        assert mock_nw_tools.calc_rxc_filter.call_count == 2
