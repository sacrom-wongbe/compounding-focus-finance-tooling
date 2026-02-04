import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

# --- Constants for trading day approximations ---
TRADING_DAYS_1M = 21
TRADING_DAYS_6M = 126
TRADING_DAYS_12M = 252
SMA200_WINDOW = 200
EXCLUDE_RECENT = 21  # Exclude most recent month

@dataclass
class DataFlags:
    has_6m: bool
    has_12m: bool
    has_sma200: bool


def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetches Close prices for the given tickers from yfinance.
    Returns a DataFrame (dates x tickers) with aligned dates (inner join).
    """
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all")

    # Remove tickers with insufficient data for 12-month rolling window
    min_required = SMA200_WINDOW + TRADING_DAYS_6M + EXCLUDE_RECENT + 1
    tickers_to_remove = []
    for ticker in data.columns:
        # Only consider non-NaN values
        series = data[ticker].dropna()
        if series.rolling(min_required).mean().isna().all():
            print(f"Skipping {ticker}: not enough data for 12-month lookback in fetch_prices.")
            tickers_to_remove.append(ticker)
    if tickers_to_remove:
        data = data.drop(columns=tickers_to_remove)
    return data


def compute_metrics(
    prices: pd.DataFrame,
    benchmark: str = "VOO",
    exclude_recent: int = EXCLUDE_RECENT
) -> pd.DataFrame:
    """
    Computes a robust set of relative strength, trend, and stability metrics for each ticker vs the benchmark.
    Returns a tidy DataFrame (tickers x metrics).
    """
    tickers = [t for t in prices.columns if t != benchmark]
    results = []
    last_idx = prices.index[-1]
    # Align all series on common dates (inner join)
    prices = prices.dropna(axis=0, how="any")
    # Compute benchmark SMA200 and gate
    voo = prices[benchmark]
    voo_sma200 = voo.rolling(SMA200_WINDOW).mean()
    voo_above_sma200 = int(voo.iloc[-1] > voo_sma200.iloc[-1]) if not np.isnan(voo_sma200.iloc[-1]) else np.nan
    # Precompute benchmark for all tickers
    for ticker in tickers:
        # Only use rows where both ticker and benchmark have data
        valid_idx = prices[[ticker, benchmark]].dropna().index
        P = prices.loc[valid_idx, ticker]
        B = prices.loc[valid_idx, benchmark]
        row = {"ticker": ticker}
        # ...existing code...
        # --- E) PE Ratio (from yfinance, set to NaN if unavailable) ---
        try:
            yf_ticker = yf.Ticker(ticker)
            pe_ratio = yf_ticker.info.get("trailingPE", np.nan)
            if pe_ratio is None:
                pe_ratio = np.nan
        except Exception:
            pe_ratio = np.nan
        row["pe_ratio"] = pe_ratio
        # --- F) FCF Margin (Free Cash Flow / Revenue) ---
        try:
            # Use quarterly data for TTM calculations
            q_cashflow = yf_ticker.quarterly_cashflow
            q_financials = yf_ticker.quarterly_financials
            
            fcf_ttm = np.nan
            rev_ttm = np.nan

            if not q_cashflow.empty:
                # yfinance often provides a pre-calculated 'Free Cash Flow' row now
                if "Free Cash Flow" in q_cashflow.index:
                    fcf_series = q_cashflow.loc["Free Cash Flow"]
                    fcf_ttm = fcf_series.iloc[:4].sum()
                else:
                    # Fallback to OCF - CapEx
                    # Note: Index names can be 'Capital Expenditure' (singular)
                    ocf_label = "Operating Cash Flow"
                    capex_label = "Capital Expenditure" # common label in newer yfinance versions
                    
                    if ocf_label in q_cashflow.index and capex_label in q_cashflow.index:
                        ocf = q_cashflow.loc[ocf_label].iloc[:4].sum()
                        # Use abs() because yfinance often records CapEx as negative
                        capex = abs(q_cashflow.loc[capex_label].iloc[:4].sum())
                        fcf_ttm = ocf - capex

            if not q_financials.empty:
                rev_label = "Total Revenue"
                if rev_label in q_financials.index:
                    rev_ttm = q_financials.loc[rev_label].iloc[:4].sum()

            # Calculate Margin
            if rev_ttm and rev_ttm > 0 and not np.isnan(fcf_ttm):
                fcf_margin = fcf_ttm / rev_ttm
            else:
                fcf_margin = np.nan

        except Exception as e:
            print(f"Error calculating FCF for {yf_ticker.ticker}: {e}")
            fcf_margin = np.nan

        row["fcf_margin"] = fcf_margin

        # --- G) ROIC (Return on Invested Capital) ---
        try:
            # 1. Get the correct DataFrames
            q_fin = yf_ticker.quarterly_financials
            q_bs = yf_ticker.quarterly_balance_sheet
            
            roic = np.nan

            if not q_fin.empty and not q_bs.empty:
                # --- NOPAT Calculation ---
                # Using EBIT * (1 - Tax Rate) is the standard formula for NOPAT
                ebit_label = "EBIT"
                pretax_label = "Pretax Income"
                tax_label = "Tax Provision" # Often called Tax Provision or Income Tax Expense

                ebit_ttm = q_fin.loc[ebit_label].iloc[:4].sum() if ebit_label in q_fin.index else np.nan
                
                # Calculate Effective Tax Rate
                if pretax_label in q_fin.index and tax_label in q_fin.index:
                    pretax_ttm = q_fin.loc[pretax_label].iloc[:4].sum()
                    tax_ttm = q_fin.loc[tax_label].iloc[:4].sum()
                    # Avoid division by zero or negative tax rates
                    eff_tax_rate = max(0, tax_ttm / pretax_ttm) if pretax_ttm > 0 else 0.21 
                else:
                    eff_tax_rate = 0.21 # Default to corporate standard if missing

                nopat = ebit_ttm * (1 - eff_tax_rate)

                # --- Invested Capital Calculation (from Balance Sheet) ---
                ta_label = "Total Assets"
                cl_label = "Current Liabilities" # Note: check if 'Total Current Liabilities' in your version

                if ta_label in q_bs.index and cl_label in q_bs.index:
                    # We take the most recent quarter (iloc[0]) for the denominator
                    total_assets = q_bs.loc[ta_label].iloc[0]
                    current_liab = q_bs.loc[cl_label].iloc[0]
                    invested_capital = total_assets - current_liab
                else:
                    invested_capital = np.nan

                # Final ROIC
                if invested_capital > 0 and not np.isnan(nopat):
                    roic = nopat / invested_capital

            row["roic"] = roic

        except Exception as e:
            print(f"Error calculating ROIC for {ticker}: {e}")
            row["roic"] = np.nan
        # Ratio and log ratio
        R = P / B
        LR = np.log(R)
        dLR = LR.diff()
        # SMA200 of ratio
        sma200_R = R.rolling(SMA200_WINDOW).mean()
        # Data flags
        has_12m = len(R) >= (TRADING_DAYS_12M + EXCLUDE_RECENT + 1)
        has_6m = len(R) >= (TRADING_DAYS_6M + EXCLUDE_RECENT + 1)
        has_sma200 = len(R) >= (SMA200_WINDOW + 1)
        row["data_ok_has_12m"] = has_12m
        row["data_ok_has_6m"] = has_6m
        row["data_ok_has_sma200"] = has_sma200
        # --- A) Market Gate ---
        row["voo_above_sma200"] = voo_above_sma200
        # --- B) Relative Strength ---
        # 1) RS 12M ex-1M
        if has_12m:
            t_1M = -EXCLUDE_RECENT
            t_13M = -(TRADING_DAYS_12M + EXCLUDE_RECENT)
            rs_12m_ex1 = 100 * (LR.iloc[t_1M] - LR.iloc[t_13M])
        else:
            rs_12m_ex1 = np.nan
        row["rs_12m_ex1"] = rs_12m_ex1
        # 2) RS 6M ex-1M
        if has_6m:
            t_1M = -EXCLUDE_RECENT
            t_7M = -(TRADING_DAYS_6M + EXCLUDE_RECENT)
            rs_6m_ex1 = 100 * (LR.iloc[t_1M] - LR.iloc[t_7M])
        else:
            rs_6m_ex1 = np.nan
        row["rs_6m_ex1"] = rs_6m_ex1
        # 3) RS Slope 6M ex-1M
        if has_6m:
            start = -(TRADING_DAYS_6M + EXCLUDE_RECENT)
            end = -EXCLUDE_RECENT
            LR_window = LR.iloc[start:end]
            x = np.arange(len(LR_window))
            if len(LR_window) == TRADING_DAYS_6M:
                slope = np.polyfit(x, LR_window, 1)[0]
                row["rs_slope_6m_ex1"] = slope * 100  # per 100d
            else:
                row["rs_slope_6m_ex1"] = np.nan
        else:
            row["rs_slope_6m_ex1"] = np.nan
        # --- C) Structural Trend Gate ---
        if has_sma200:
            ratio_above_sma200_binary = int(R.iloc[-1] > sma200_R.iloc[-1]) if not np.isnan(sma200_R.iloc[-1]) else np.nan
            ratio_above_sma200 = R.iloc[-1] / sma200_R.iloc[-1] if not np.isnan(sma200_R.iloc[-1]) else np.nan
            ratio_dist_sma200 = 100 * np.log(R.iloc[-1] / sma200_R.iloc[-1]) if not np.isnan(sma200_R.iloc[-1]) else np.nan
        else:
            ratio_above_sma200_binary = np.nan
            ratio_above_sma200 = np.nan
            ratio_dist_sma200 = np.nan
        row["ratio_above_sma200_binary"] = ratio_above_sma200_binary
        row["ratio_above_sma200"] = ratio_above_sma200
        row["ratio_dist_sma200"] = ratio_dist_sma200
        # 5) SMA200 slope on ratio (6M)
        if has_sma200 and has_6m:
            sma200_ln = np.log(sma200_R)
            start = -(TRADING_DAYS_6M + EXCLUDE_RECENT)
            end = -EXCLUDE_RECENT
            sma200_window = sma200_ln.iloc[start:end]
            x = np.arange(len(sma200_window))
            if len(sma200_window) == TRADING_DAYS_6M:
                slope = np.polyfit(x, sma200_window, 1)[0]
                row["ratio_sma200_slope_6m"] = slope * 100
            else:
                row["ratio_sma200_slope_6m"] = np.nan
        else:
            row["ratio_sma200_slope_6m"] = np.nan
        # --- D) Persistence & Stability ---
        # 6) % Days above SMA200 (6M)
        if has_sma200 and has_6m:
            window_R = R.iloc[-(TRADING_DAYS_6M + EXCLUDE_RECENT):-EXCLUDE_RECENT]
            window_SMA = sma200_R.iloc[-(TRADING_DAYS_6M + EXCLUDE_RECENT):-EXCLUDE_RECENT]
            valid = ~np.isnan(window_R) & ~np.isnan(window_SMA)
            if valid.sum() > 0:
                pct = 100 * np.mean(window_R[valid] > window_SMA[valid])
                row["pct_days_ratio_above_sma200_6m"] = pct
            else:
                row["pct_days_ratio_above_sma200_6m"] = np.nan
        else:
            row["pct_days_ratio_above_sma200_6m"] = np.nan
        # 7) RS Volatility 6M ex-1M
        if has_6m:
            start = -(TRADING_DAYS_6M + EXCLUDE_RECENT)
            end = -EXCLUDE_RECENT
            dLR_window = dLR.iloc[start:end]
            if dLR_window.notna().sum() == TRADING_DAYS_6M:
                rs_vol = 100 * dLR_window.std() * np.sqrt(252)
                row["rs_vol_6m_ex1"] = rs_vol
                # RS Volatility Slope (linear regression) (10000x scaling)
                x = np.arange(len(dLR_window))
                rs_vol_slope = np.polyfit(x, dLR_window, 1)[0] * 10000 if len(dLR_window) == TRADING_DAYS_6M else np.nan
                row["rs_vol_slope_6m_ex1"] = rs_vol_slope
            else:
                row["rs_vol_6m_ex1"] = np.nan
                row["rs_vol_slope_6m_ex1"] = np.nan
        else:
            row["rs_vol_6m_ex1"] = np.nan
            row["rs_vol_slope_6m_ex1"] = np.nan
        # 8) RS Max Drawdown 6M ex-1M
        if has_6m:
            start = -(TRADING_DAYS_6M + EXCLUDE_RECENT)
            end = -EXCLUDE_RECENT
            LR_window = LR.iloc[start:end]
            equity = np.exp(LR_window - LR_window.iloc[0])
            peak = np.maximum.accumulate(equity)
            dd = equity / peak - 1
            max_dd = 100 * dd.min()
            row["rs_max_dd_6m_ex1"] = max_dd
        else:
            row["rs_max_dd_6m_ex1"] = np.nan
        # 9) Momentum Efficiency 6M ex-1M
        if has_6m:
            start = -(TRADING_DAYS_6M + EXCLUDE_RECENT)
            end = -EXCLUDE_RECENT
            LR_window = LR.iloc[start:end]
            net_change = np.abs(LR_window.iloc[-1] - LR_window.iloc[0])
            path_length = np.sum(np.abs(np.diff(LR_window)))
            if path_length > 0:
                mom_eff = net_change / path_length
                row["mom_eff_6m_ex1"] = mom_eff
                # --- Compute rolling efficiency over a shorter subwindow (e.g., 21D) ---
                ROLL_EFF_WINDOW = 21
                if len(LR_window) >= ROLL_EFF_WINDOW:
                    rolling_eff = []
                    for i in range(len(LR_window) - ROLL_EFF_WINDOW + 1):
                        sub = LR_window.iloc[i:i+ROLL_EFF_WINDOW]
                        net = np.abs(sub.iloc[-1] - sub.iloc[0])
                        path = np.sum(np.abs(np.diff(sub)))
                        eff = net / path if path > 0 else np.nan
                        rolling_eff.append(eff)
                    rolling_eff = np.array(rolling_eff)
                    x = np.arange(len(rolling_eff))
                    if np.isfinite(rolling_eff).sum() == len(rolling_eff) and len(rolling_eff) > 1:
                        slope = np.polyfit(x, rolling_eff, 1)[0]
                        row["mom_eff_slope_6m_ex1"] = slope * 100  # 100x scaling
                    else:
                        row["mom_eff_slope_6m_ex1"] = np.nan
                else:
                    row["mom_eff_slope_6m_ex1"] = np.nan
            else:
                row["mom_eff_6m_ex1"] = np.nan
                row["mom_eff_slope_6m_ex1"] = np.nan
        else:
            row["mom_eff_6m_ex1"] = np.nan
            row["mom_eff_slope_6m_ex1"] = np.nan

        # --- NEW: Distribution Days (last 30 days) ---
        # Only compute if both price and volume data are available
        try:
            # Try to fetch volume data for this ticker
            if ticker in prices.columns and ticker in prices.columns.get_level_values(0):
                # MultiIndex (not expected here), fallback to single index
                close = prices[ticker]
                volume = prices[ticker]  # fallback, but this is not correct
            else:
                close = prices[ticker]
                # Try to get volume from yfinance
                yf_hist = yf.Ticker(ticker).history(period="max")
                if not yf_hist.empty and "Volume" in yf_hist.columns:
                    # Align index to prices
                    volume = yf_hist["Volume"].reindex(close.index)
                else:
                    volume = pd.Series(index=close.index, data=np.nan)

            ret = close.pct_change()
            vol_avg_20 = volume.rolling(20).mean()
            distribution_day = (
                (ret <= -0.005) &
                (close < close.shift(1)) &
                (volume > vol_avg_20)
            )
            dist_days_30 = distribution_day.rolling(30).sum().iloc[-1]
            row["distribution_days_30"] = dist_days_30
        except Exception as e:
            row["distribution_days_30"] = np.nan
        # --- Up/Down Volume Pressure (UVP) ---
        try:
            w = 40  # ~3 months
            close = P
            yf_hist = yf.Ticker(ticker).history(period="max")
            
            if not yf_hist.empty and "Volume" in yf_hist.columns:
                # Normalize both indexes to date-only
                yf_hist.index = pd.to_datetime(yf_hist.index).date
                close_dates = pd.to_datetime(close.index).date
                
                volume_raw = pd.Series(yf_hist["Volume"].values, index=yf_hist.index)
                volume = volume_raw.reindex(close_dates)
                volume.index = close.index
            else:
                volume = pd.Series(index=close.index, data=np.nan)
            
            if volume.isna().all():
                row["uvp_pct_40d"] = np.nan
                row["uvp_slope_40d"] = np.nan
            else:
                ret = close.pct_change()
                up = ret > 0
                down = ret < 0
                
                # KEY FIX: Fill NaN with 0 so rolling can accumulate
                up_vol = volume.where(up, 0).rolling(w, min_periods=1).sum()
                down_vol = volume.where(down, 0).rolling(w, min_periods=1).sum()
                
                # Avoid division by zero
                total_vol = up_vol + down_vol
                uvp_pct = up_vol / total_vol.replace(0, np.nan)

                row["uvp_pct_40d"] = uvp_pct.iloc[-1] if not uvp_pct.empty and not pd.isna(uvp_pct.iloc[-1]) else np.nan

                uvp_pct_window = uvp_pct.dropna()
                
                if len(uvp_pct_window) > 1:
                    x = np.arange(len(uvp_pct_window))
                    uvp_slope = np.polyfit(x, uvp_pct_window, 1)[0] * 1000
                    row["uvp_slope_40d"] = uvp_slope
                else:
                    row["uvp_slope_40d"] = np.nan
                    
        except Exception as e:
            row["uvp_pct_40d"] = np.nan
            row["uvp_slope_40d"] = np.nan
        results.append(row)
    df = pd.DataFrame(results).set_index("ticker")
    return df