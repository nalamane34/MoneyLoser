"""Report generation for backtest results and daily operations.

Produces human-readable Markdown reports summarizing system performance,
calibration health, and model drift status.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from moneygone.monitoring.calibration_monitor import CalibrationMonitor
from moneygone.monitoring.drift import DriftDetector
from moneygone.monitoring.pnl import PnLTracker

log = structlog.get_logger(__name__)


class ReportGenerator:
    """Generates formatted reports from system state.

    All reports are returned as Markdown strings suitable for logging,
    file output, or webhook delivery.
    """

    # ------------------------------------------------------------------
    # Backtest report
    # ------------------------------------------------------------------

    @staticmethod
    def generate_backtest_report(result: Any) -> str:
        """Generate a comprehensive backtest report.

        Parameters
        ----------
        result:
            A ``BacktestResult`` object with attributes:
            ``total_pnl``, ``num_trades``, ``win_rate``, ``sharpe_ratio``,
            ``max_drawdown``, ``trades`` (list), ``daily_pnl`` (list),
            ``config`` (dict).

        Returns
        -------
        str
            Formatted Markdown report.
        """
        lines: list[str] = []
        lines.append("# Backtest Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total PnL**: ${_get(result, 'total_pnl', 0.0):.2f}")
        lines.append(f"- **Number of Trades**: {_get(result, 'num_trades', 0)}")
        lines.append(f"- **Win Rate**: {_get(result, 'win_rate', 0.0):.1%}")
        lines.append(f"- **Sharpe Ratio**: {_get(result, 'sharpe_ratio', 0.0):.2f}")
        lines.append(f"- **Max Drawdown**: {_get(result, 'max_drawdown', 0.0):.2%}")
        lines.append("")

        # Configuration
        config = _get(result, "config", {})
        if config:
            lines.append("## Configuration")
            lines.append("")
            if isinstance(config, dict):
                for key, value in config.items():
                    lines.append(f"- **{key}**: {value}")
            lines.append("")

        # Trade statistics
        trades = _get(result, "trades", [])
        if trades:
            lines.append("## Trade Statistics")
            lines.append("")

            pnls = [_get(t, "pnl", 0.0) for t in trades if isinstance(t, dict)]
            if pnls:
                winners = [p for p in pnls if p > 0]
                losers = [p for p in pnls if p < 0]

                lines.append(f"- **Total trades**: {len(pnls)}")
                lines.append(f"- **Winners**: {len(winners)}")
                lines.append(f"- **Losers**: {len(losers)}")
                if winners:
                    lines.append(
                        f"- **Avg winner**: ${sum(winners) / len(winners):.2f}"
                    )
                if losers:
                    lines.append(
                        f"- **Avg loser**: ${sum(losers) / len(losers):.2f}"
                    )
                if winners and losers:
                    avg_win = sum(winners) / len(winners)
                    avg_loss = abs(sum(losers) / len(losers))
                    ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")
                    lines.append(f"- **Win/Loss ratio**: {ratio:.2f}")
            lines.append("")

        # Top and bottom trades
        if trades and len(trades) >= 2:
            sorted_trades = sorted(
                [t for t in trades if isinstance(t, dict)],
                key=lambda t: _get(t, "pnl", 0.0),
                reverse=True,
            )

            lines.append("## Top 5 Trades")
            lines.append("")
            lines.append("| Ticker | Side | PnL | Edge |")
            lines.append("|--------|------|-----|------|")
            for t in sorted_trades[:5]:
                lines.append(
                    f"| {_get(t, 'ticker', '?')} "
                    f"| {_get(t, 'side', '?')} "
                    f"| ${_get(t, 'pnl', 0.0):.2f} "
                    f"| {_get(t, 'edge', 0.0):.4f} |"
                )
            lines.append("")

            lines.append("## Bottom 5 Trades")
            lines.append("")
            lines.append("| Ticker | Side | PnL | Edge |")
            lines.append("|--------|------|-----|------|")
            for t in sorted_trades[-5:]:
                lines.append(
                    f"| {_get(t, 'ticker', '?')} "
                    f"| {_get(t, 'side', '?')} "
                    f"| ${_get(t, 'pnl', 0.0):.2f} "
                    f"| {_get(t, 'edge', 0.0):.4f} |"
                )
            lines.append("")

        report = "\n".join(lines)
        log.info("report.backtest_generated", length=len(report))
        return report

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    @staticmethod
    def generate_daily_report(
        pnl_tracker: PnLTracker,
        calibration_monitor: CalibrationMonitor,
        drift_detector: DriftDetector,
    ) -> str:
        """Generate a daily operations report.

        Parameters
        ----------
        pnl_tracker:
            PnL tracker with today's trade data.
        calibration_monitor:
            Calibration monitor with prediction/outcome pairs.
        drift_detector:
            Drift detector with recent prediction buffer.

        Returns
        -------
        str
            Formatted Markdown report.
        """
        lines: list[str] = []
        lines.append("# Daily Trading Report")
        lines.append("")
        lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
        lines.append("")

        # PnL summary
        daily_summary = pnl_tracker.get_summary(period="daily")
        overall_summary = pnl_tracker.get_summary(period="all")

        lines.append("## PnL Summary")
        lines.append("")
        lines.append("### Today")
        lines.append("")
        lines.append(f"- **Net PnL**: ${daily_summary.net_pnl:.2f}")
        lines.append(f"- **Gross PnL**: ${daily_summary.gross_pnl:.2f}")
        lines.append(f"- **Fees Paid**: ${daily_summary.fees_paid:.2f}")
        lines.append(f"- **Trades**: {daily_summary.num_trades}")
        lines.append(f"- **Win Rate**: {daily_summary.win_rate:.1%}")
        lines.append(f"- **Sharpe**: {daily_summary.sharpe:.2f}")
        lines.append("")

        lines.append("### Overall")
        lines.append("")
        lines.append(f"- **Net PnL**: ${overall_summary.net_pnl:.2f}")
        lines.append(f"- **Trades**: {overall_summary.num_trades}")
        lines.append(f"- **Win Rate**: {overall_summary.win_rate:.1%}")
        lines.append(f"- **Avg Edge (predicted)**: {overall_summary.avg_edge_predicted:.4f}")
        lines.append(f"- **Avg Edge (realized)**: {overall_summary.avg_edge_realized:.4f}")
        lines.append(f"- **Sharpe**: {overall_summary.sharpe:.2f}")
        lines.append("")

        # Attribution
        attribution = pnl_tracker.get_attribution()
        if attribution:
            lines.append("## Category Attribution")
            lines.append("")
            lines.append("| Category | Net PnL | Trades | Win Rate |")
            lines.append("|----------|---------|--------|----------|")
            for cat, summary in sorted(
                attribution.items(), key=lambda x: x[1].net_pnl, reverse=True
            ):
                lines.append(
                    f"| {cat} | ${summary.net_pnl:.2f} "
                    f"| {summary.num_trades} "
                    f"| {summary.win_rate:.1%} |"
                )
            lines.append("")

        # Calibration
        cal_metrics = calibration_monitor.get_rolling_metrics()
        lines.append("## Calibration")
        lines.append("")
        lines.append(f"- **Brier Score**: {cal_metrics.brier_score:.4f}")
        lines.append(f"- **ECE**: {cal_metrics.ece:.4f}")
        lines.append(f"- **Log Loss**: {cal_metrics.log_loss:.4f}")
        lines.append(f"- **Resolved Predictions**: {cal_metrics.n_resolved}")
        is_degraded = calibration_monitor.is_degraded()
        status = "DEGRADED" if is_degraded else "OK"
        lines.append(f"- **Status**: {status}")
        lines.append("")

        # Drift
        drift_result = drift_detector.check_drift()
        lines.append("## Model Drift")
        lines.append("")
        lines.append(f"- **Metric**: {drift_result.metric_name}")
        lines.append(f"- **Value**: {drift_result.metric_value:.4f}")
        lines.append(f"- **Threshold**: {drift_result.threshold:.4f}")
        lines.append(f"- **Severity**: {drift_result.severity}")
        lines.append(f"- **Drifted**: {'YES' if drift_result.is_drifted else 'NO'}")
        lines.append("")

        report = "\n".join(lines)
        log.info("report.daily_generated", length=len(report))
        return report


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute or dict key from an object."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
