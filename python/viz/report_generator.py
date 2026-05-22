"""
PDF report generator using matplotlib and reportlab.

Generates a multi-page PDF containing:
  - Cover page with strategy summary
  - Equity curve and drawdown chart
  - Walk-forward fold table
  - Return distribution histogram
  - Statistical significance table
  - Pairs cointegration status table
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Generates a multi-page PDF report from walk-forward backtest results.

    Parameters
    ----------
    output_path : path to the output PDF file
    style       : matplotlib style ('dark_background', 'seaborn-v0_8', etc.)
    """

    TITLE_FONT_SIZE = 20
    HEADER_FONT_SIZE = 14
    BODY_FONT_SIZE = 10

    def __init__(
        self,
        output_path: str = "stat_arb_report.pdf",
        style: str = "dark_background",
    ) -> None:
        self.output_path = Path(output_path)
        self.style = style
        plt.style.use(self.style)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        fold_results: list,
        strategy_name: str = "Statistical Arbitrage Engine",
    ) -> Path:
        """
        Generate a full PDF report.

        Parameters
        ----------
        fold_results  : list of FoldResult from WalkForwardBacktester
        strategy_name : title string for the report

        Returns
        -------
        Path to the generated PDF.
        """
        from python.backtest.metrics import compute_metrics
        from python.backtest.statistical_tests import (
            walk_forward_t_test,
            bootstrap_confidence_interval,
            ljung_box_test,
        )

        all_rets = np.concatenate([f.daily_returns for f in fold_results])
        all_trades = [t for f in fold_results for t in f.trades]
        metrics = compute_metrics(all_rets, all_trades)
        sharpes = [f.sharpe for f in fold_results]
        t_stat, p_val = walk_forward_t_test(sharpes)
        ci_lo, ci_hi = bootstrap_confidence_interval(np.array(sharpes))
        lb_stat, lb_p = ljung_box_test(all_rets)

        with PdfPages(str(self.output_path)) as pdf:
            self._page_cover(pdf, strategy_name, metrics)
            self._page_equity(pdf, all_rets)
            self._page_fold_table(pdf, fold_results)
            self._page_statistics(pdf, all_rets, metrics, sharpes, t_stat, p_val,
                                   ci_lo, ci_hi, lb_stat, lb_p)
            self._page_return_distribution(pdf, all_rets)
            pdf.infodict()["Title"] = strategy_name
            pdf.infodict()["Author"] = "Stat-Arb Engine"
            pdf.infodict()["CreationDate"] = datetime.now(timezone.utc)

        return self.output_path

    # ------------------------------------------------------------------
    # Page builders
    # ------------------------------------------------------------------

    def _page_cover(self, pdf: PdfPages, title: str, metrics: dict) -> None:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("#0d0d1a")
        ax.axis("off")

        # Title
        ax.text(0.5, 0.85, title, transform=ax.transAxes,
                fontsize=28, fontweight="bold", ha="center", va="center",
                color="#00c4b4")

        ax.text(0.5, 0.77, "Walk-Forward Backtest Report",
                transform=ax.transAxes, fontsize=16, ha="center",
                color="#cccccc")

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        ax.text(0.5, 0.72, f"Generated: {ts}",
                transform=ax.transAxes, fontsize=11, ha="center",
                color="#888888")

        # Key metrics table on cover
        cover_metrics = [
            ("Annualised Return", f"{metrics['annualised_return']:.2%}"),
            ("Annualised Volatility", f"{metrics['annualised_vol']:.2%}"),
            ("Sharpe Ratio", f"{metrics['sharpe']:.2f}"),
            ("Sortino Ratio", f"{metrics['sortino']:.2f}"),
            ("Calmar Ratio", f"{metrics['calmar']:.2f}"),
            ("Maximum Drawdown", f"{metrics['max_drawdown']:.2%}"),
            ("Total Trades", str(metrics['n_trades'])),
            ("Win Rate", f"{metrics['win_rate']:.1%}"),
        ]

        y0 = 0.62
        for i, (label, val) in enumerate(cover_metrics):
            row = i % 4
            col = i // 4
            x = 0.15 + col * 0.5
            y = y0 - row * 0.08
            ax.text(x, y, label + ":", transform=ax.transAxes,
                    fontsize=12, ha="right", color="#aaaaaa")
            ax.text(x + 0.02, y, val, transform=ax.transAxes,
                    fontsize=12, ha="left", color="#00c4b4", fontweight="bold")

        # Horizontal rule
        ax.axhline(y=0.68, xmin=0.1, xmax=0.9, color="#333366", linewidth=1.5)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

    def _page_equity(self, pdf: PdfPages, daily_returns: np.ndarray) -> None:
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.2, 1.2], hspace=0.4)

        # Equity curve
        equity = np.cumprod(1.0 + daily_returns)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.maximum(peak, 1e-10)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(equity, color="#00c4b4", linewidth=1.5, label="Portfolio")
        ax1.plot(peak, color="#555588", linewidth=0.8, linestyle="--", label="Peak")
        ax1.fill_between(range(len(equity)), equity, peak, alpha=0.15, color="#00c4b4")
        ax1.set_title("Equity Curve", fontsize=self.HEADER_FONT_SIZE, color="white")
        ax1.set_ylabel("NAV", color="white")
        ax1.legend(loc="upper left", fontsize=8)
        self._style_axes(ax1)

        # Drawdown
        ax2 = fig.add_subplot(gs[1])
        ax2.fill_between(range(len(drawdown)), drawdown * 100, 0,
                         alpha=0.7, color="#ff4444", label="Drawdown")
        ax2.set_title("Drawdown (%)", fontsize=self.HEADER_FONT_SIZE, color="white")
        ax2.set_ylabel("DD %", color="white")
        self._style_axes(ax2)

        # Rolling 63-day Sharpe
        ax3 = fig.add_subplot(gs[2])
        roll_sharpe = self._rolling_sharpe(daily_returns, window=63)
        ax3.plot(roll_sharpe, color="#ffaa00", linewidth=1.2)
        ax3.axhline(y=0, color="#555555", linewidth=0.8)
        ax3.set_title("Rolling 63-Day Sharpe", fontsize=self.HEADER_FONT_SIZE, color="white")
        ax3.set_ylabel("Sharpe", color="white")
        self._style_axes(ax3)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

    def _page_fold_table(self, pdf: PdfPages, fold_results: list) -> None:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        headers = ["Fold", "Period", "Sharpe", "Sortino", "Max DD",
                   "Ann. Ret.", "Trades", "Win Rate"]
        rows = []
        for f in fold_results:
            rows.append([
                str(f.fold_id),
                f"{f.test_start.date()} → {f.test_end.date()}",
                f"{f.sharpe:.2f}",
                f"{f.sortino:.2f}",
                f"{f.max_drawdown:.1%}",
                f"{f.annualised_return:.1%}",
                str(f.n_trades),
                f"{f.win_rate:.1%}",
            ])

        if rows:
            table = ax.table(
                cellText=rows,
                colLabels=headers,
                cellLoc="center",
                loc="center",
                bbox=[0, 0.05, 1, 0.90],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)

            # Colour header
            for j in range(len(headers)):
                table[0, j].set_facecolor("#1a1a3d")
                table[0, j].set_text_props(color="#00c4b4", fontweight="bold")

            # Colour data rows by Sharpe sign
            for i, f in enumerate(fold_results):
                colour = "#0d2b0d" if f.sharpe > 0 else "#2b0d0d"
                for j in range(len(headers)):
                    table[i + 1, j].set_facecolor(colour)
                    table[i + 1, j].set_text_props(color="white")

        ax.set_title("Walk-Forward Fold Results", fontsize=self.HEADER_FONT_SIZE,
                     color="white", pad=20)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

    def _page_statistics(
        self,
        pdf: PdfPages,
        daily_returns: np.ndarray,
        metrics: dict,
        sharpes: List[float],
        t_stat: float,
        p_val: float,
        ci_lo: float,
        ci_hi: float,
        lb_stat: float,
        lb_p: float,
    ) -> None:
        from scipy.stats import skew, kurtosis
        from python.backtest.metrics import deflated_sharpe_ratio

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Statistical Analysis", fontsize=16, color="white")

        # -- Fold Sharpe bar chart --
        ax = axes[0, 0]
        colours = ["#00c4b4" if s > 0 else "#ff4444" for s in sharpes]
        ax.bar(range(len(sharpes)), sharpes, color=colours)
        ax.axhline(0, color="white", linewidth=0.8)
        ax.set_title("Fold Sharpe Ratios", color="white")
        ax.set_xlabel("Fold", color="white")
        ax.set_ylabel("Sharpe", color="white")
        self._style_axes(ax)

        # -- Statistical summary table --
        ax = axes[0, 1]
        ax.axis("off")
        sk = float(skew(daily_returns))
        ku = float(kurtosis(daily_returns))
        dsr = deflated_sharpe_ratio(
            sharpe_obs=float(np.mean(sharpes)),
            n_trials=len(sharpes),
            n_obs=len(daily_returns),
            skew=sk,
            excess_kurt=ku,
        )
        stat_rows = [
            ["Mean Fold Sharpe", f"{np.mean(sharpes):.3f}"],
            ["t-statistic", f"{t_stat:.3f}"],
            ["p-value", f"{p_val:.4f}"],
            ["95% CI Sharpe", f"[{ci_lo:.2f}, {ci_hi:.2f}]"],
            ["Skewness", f"{sk:.3f}"],
            ["Excess Kurtosis", f"{ku:.3f}"],
            ["Ljung-Box Q", f"{lb_stat:.2f} (p={lb_p:.3f})"],
            ["DSR", f"{dsr:.4f}"],
        ]
        t = ax.table(cellText=stat_rows, colLabels=["Statistic", "Value"],
                     cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        for j in range(2):
            t[0, j].set_facecolor("#1a1a3d")
            t[0, j].set_text_props(color="#00c4b4", fontweight="bold")
        for i in range(1, len(stat_rows) + 1):
            for j in range(2):
                t[i, j].set_facecolor("#0d0d1a")
                t[i, j].set_text_props(color="white")
        ax.set_title("Statistical Tests", color="white")

        # -- Monthly return heatmap --
        ax = axes[1, 0]
        self._monthly_returns_heatmap(ax, daily_returns)

        # -- QQ plot --
        ax = axes[1, 1]
        from scipy.stats import probplot
        (osm, osr), _ = probplot(daily_returns, dist="norm")
        ax.scatter(osm, osr, color="#00c4b4", s=5, alpha=0.7)
        ax.plot(osm, osm, color="red", linewidth=1, linestyle="--")
        ax.set_title("Q-Q Plot vs Normal", color="white")
        ax.set_xlabel("Theoretical quantiles", color="white")
        ax.set_ylabel("Sample quantiles", color="white")
        self._style_axes(ax)

        plt.tight_layout()
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

    def _page_return_distribution(self, pdf: PdfPages, daily_returns: np.ndarray) -> None:
        from scipy.stats import norm

        fig, ax = plt.subplots(figsize=(11, 6))
        n, bins, _ = ax.hist(daily_returns * 100, bins=60, density=True,
                              color="#00c4b4", alpha=0.6, edgecolor="#333333",
                              label="Empirical")

        # Overlay normal
        mu = daily_returns.mean() * 100
        sig = daily_returns.std() * 100
        x = np.linspace(bins[0], bins[-1], 200)
        ax.plot(x, norm.pdf(x, mu, sig), color="orange", linewidth=2,
                label=f"Normal N({mu:.3f}, {sig:.3f})")

        # VaR lines
        var_95 = np.percentile(daily_returns * 100, 5)
        var_99 = np.percentile(daily_returns * 100, 1)
        ax.axvline(var_95, color="yellow", linestyle="--", linewidth=1.5,
                   label=f"VaR 95% = {var_95:.2f}%")
        ax.axvline(var_99, color="red", linestyle="--", linewidth=1.5,
                   label=f"VaR 99% = {var_99:.2f}%")

        ax.set_title("Daily Return Distribution", fontsize=self.HEADER_FONT_SIZE, color="white")
        ax.set_xlabel("Daily Return (%)", color="white")
        ax.set_ylabel("Density", color="white")
        ax.legend(fontsize=9)
        self._style_axes(ax)

        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_sharpe(returns: np.ndarray, window: int = 63) -> np.ndarray:
        n = len(returns)
        rs = np.full(n, np.nan)
        for t in range(window, n):
            w = returns[t - window:t]
            sig = w.std()
            rs[t] = w.mean() / sig * np.sqrt(252) if sig > 1e-10 else 0.0
        return rs

    @staticmethod
    def _monthly_returns_heatmap(ax: plt.Axes, daily_returns: np.ndarray) -> None:
        """Pivot monthly returns into a year x month heatmap."""
        idx = pd.date_range("2022-01-01", periods=len(daily_returns), freq="B")
        s = pd.Series(daily_returns, index=idx)
        monthly = s.resample("ME").apply(lambda x: float(np.prod(1 + x) - 1))
        if monthly.empty:
            ax.set_title("Monthly Returns — no data", color="white")
            return

        df = monthly.to_frame("ret")
        df["year"] = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot_table(values="ret", index="year", columns="month", aggfunc="first")
        pivot = pivot.fillna(0)

        im = ax.imshow(pivot.values * 100, aspect="auto", cmap="RdYlGn",
                       vmin=-10, vmax=10)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)],
            color="white", fontsize=7
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist(), color="white", fontsize=8)
        ax.set_title("Monthly Returns (%)", color="white")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j] * 100
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(val) < 5 else "white")

    @staticmethod
    def _style_axes(ax: plt.Axes) -> None:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from python.backtest.walk_forward import (
        WalkForwardBacktester, _generate_synthetic_prices
    )

    print("Generating synthetic data and running backtest...")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    pairs = [("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "BNBUSDT"), ("ETHUSDT", "SOLUSDT")]
    price_data = _generate_synthetic_prices(symbols, n_days=700)

    wf = WalkForwardBacktester()
    fold_results = wf.run(price_data, pairs)

    print(f"Completed {len(fold_results)} folds. Generating PDF report...")
    gen = ReportGenerator("stat_arb_report.pdf")
    out = gen.generate(fold_results)
    print(f"Report saved to: {out}")
