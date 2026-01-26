from unittest.mock import MagicMock

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.catalog.catalog import sample_cable_catalog

from mechaphlowers.plotting.plot import PlotEngine


def test_plot_engine_reset_called_on_notify(
    balance_engine_base_test: BalanceEngine,
):
    plot_engine = PlotEngine(balance_engine_base_test)

    # Replace reset with mock after construction to track calls from notify
    original_reset = plot_engine.reset
    plot_engine.reset = MagicMock(wraps=original_reset)  # type: ignore[assignment]

    balance_engine_base_test.notify()

    plot_engine.reset.assert_called_once_with(balance_engine=balance_engine_base_test)
    
 