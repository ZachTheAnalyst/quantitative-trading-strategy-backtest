"""
=
Quantitative Trrading Strategy Backtest
BACKTESTING MOVING AVERAGE CROSSOVER STRATEGIES ON A DIVERSIFIED EQUITY PORTFOLIO
--
Paper: "Backtesting and Analysis of a Quantitative Trrading Strategy: 
    A Case Study in Moving Acerage Crossovers and Portfolio Diversification" 

Stocks: APPL, MSFT, JPM, XOM, JNJ (5-scetor diversified portfolio)
Method: Moving Acerage Crossover (50-day SMA / 200-day SMA)
Outputs: Charts, Tables, Statistical Tests, Equity Curves
Sources: Brock et al. (1992), Fama (1970), While (2000), Markowitz (1952)
==

HOW TO RUN:
    pip install pandas numpy matplotlib scipy yfinance tabulate
    python wuant_trading_backtest.py

charts are saved as png in same folder as script. 
tables are printed in the terminal - copy them to paper 

^^^^
tables i want saved as pdfs later 

"""



from cProfile import label
import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
from matplotlib.patches import Patch
from scipy import stats
import os 


# Try to download live data; fall back if no internet 
try:
    import yfinance as yf
    USE_LIVE_DATA = True
except ImportError:
    USE_LIVE_DATA = False

#=
#   CONFIGURATION    - CHANGE anything here
#=

TICKERS = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
SECTORS =  ['Technology', 'Technology', 'Financials', 'Energy', 'Healthcare']
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'
SHORT_WINDOW = 50   #days - short moving aavr
LONG_WINDOW = 200   #days - long moving avr
TRANS_COST = .001   #.1% per trade (round-trip ≈ .2%)
RISK_FREE = .02     #annual risk-free rate for Sharpe ratio
INSAMPLE_END = '2018-12-31' # in-samle / ount-of-samle split

CHART_STYLE = 'seaborn-v0_8-whitegrid'
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

#=
#   DATA DOWNLOAD
#=

def download_data(tickers, start, end):
    """Download adjusted closing prices. Falls back to offline data."""
    
    if USE_LIVE_DATA:
        print(f"\n[1/6] Downloading price data for {tickers} ...")
        try: 
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )['Close']

            raw = raw.dropna()

            print(f"     Downloaded {len(raw)} trading days "
                  f"({raw.index[0].date()} -> {raw.index[-1].date()})")
            return raw

        except Exception as e:
            print(f"     Live download failed ({e}). Using Offline Data.")

    # -- Synthetic fallback --
    print("\n[1/6] Generating synthetic price data (no internet connection) ...")

    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)

    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0003 + i * 0.00005
        vol = 0.015 - i * 0.001
        rets = np.random.normal(drift, vol, len(dates))
        prices = 100 * np.exp(np.cumsum(rets))
        data[t] = prices

    df = pd.DataFrame(data, index=dates)

    print(f"     Synthetic data: {len(df)} trading days "
          f"({df.index[0].date()} -> {df.index[-1].date()})")

    return df


# --
#   STRATEGY LOGIC 
# --

def compute_signals(prices, short_w, long_w):
    """

    Compute moving averages and crossover signals.

    Signal = +1 (long) when short MA > long MA
    signal = 0  (cash) when short MA < long MA
    Position changes generate a transaction cost.
    """

    short_ma = prices.rolling(window=short_w).mean()
    long_ma  = prices.rolling(window=long_w).mean()

    #raw signal: 1 = in market, 0 = in cash 
    signal = (short_ma> long_ma).astype(int)

    #shift by 1 day - trade on NEXT day's open to avoid look-ahead bias
    position = signal.shift(1)

    return short_ma, long_ma, position 


def compute_returns(prices, position, trans_cost):
    """
    Compute strategy daily ruturns after transaction costs.
    Also compute buy - and - hold returns for comparison. 
    """

    daily_ret = prices.pct_change()

    #number of trades = times position changes 
    trades = position.diff().abs()

    #strategy return = position * market return - cost when trade occurs 
    strat_ret = position * daily_ret - trades * trans_cost 

    #buy - and - hold return 
    bah_ret = daily_ret

    return strat_ret, bah_ret, trades


#--
#   PERFORMANCE METRICS
#-- 

def annualize(daily_ret):
    """Compound and annualize a daily return series."""

    cum = (1 + daily_ret.dropna()).prod()
    n = len(daily_ret.dropna())
    return cum ** (252 / n) - 1

def sharpe(daily_ret, rf_annual = RISK_FREE):
    """Annualized Sharpe ration."""
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    excess = daily_ret.dropna() - rf_daily
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(252)

def max_drawdown(daily_ret):
    """Maximum peak-to-through drawdown."""
    cumret = (1 + daily_ret.dropna()).cumprod()
    roll_max = cumret.cummax()
    drawdown = (cumret - roll_max) / roll_max
    return drawdown.min()

def win_rate(daily_ret):
    r = daily_ret.dropna()
    return (r > 0).sum() / len(r)

def t_test_excess(strat_ret, bah_ret):
    """
    Two-tailed t-test: H0 = mean excess return = 0
    Returns t-statistic, p-balue, 95% CI.
    """

    excess = (strat_ret - bah_ret).dropna()
    t_stat, p_val = stats.ttest_1samp(excess, 0)
    ci = stats.t.interval(0.95, df=len(excess)-1,
                          loc = excess.mean(),
                          scale = stats.sem(excess))

    return t_stat, p_val, ci, excess.mean(), excess.std()

def performance_summary(ticker, strat_ret, bah_ret, trades):
    """return a dict of all performance metrics for one ticker."""
    t, p, ci, ex_mean, ex_std = t_test_excess(strat_ret, bah_ret)
    n_trades = int(trades.sum())
    return{
        'Ticker' : ticker,
        'Strategy Ann. Return' : annualize(strat_ret),
        'B&H Ann. Return' : annualize(bah_ret),
        'Excess Return (Ann.)' : annualize(strat_ret) - annualize(bah_ret),
        'Strategy Sharpe' : sharpe(strat_ret),
        'B&H Sharpe' : sharpe(bah_ret),
        'Strategy Max Drawdown' : max_drawdown(strat_ret),
        'B&H Max Drawdown' : max_drawdown(bah_ret),
        'Win Rate (Strategy)' : win_rate(strat_ret),
        'Number of Trades' : n_trades,
        'Mean Daily Excess Ret' : ex_mean,
        'Std Daily Excess Ret' : ex_std,
        't-Statistic' : t,
        'p-Value' : p,
        '95% CI Lower' : ci[0],
        '95% CI Upper' : ci[1],
        'Significant (5%)' : 'Yes' if p < 0.05 else 'No',
        
        }


#--
#   DESCRIPTIVE STATISTICS
#--

def descriptive_stats(prices):
    """Table 1 - Descriptive statistics of daily log returns"""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    rows = []
    for col in log_ret.columns:
        r = log_ret[col].dropna()
        jb_stat, jb_p = stats.jarque_bera(r)
        ac1 = r.autocorr(lag=1)     #first-order auto corr
        rows.append({
            'Ticker' : col,
            'Mean (Daily)' : r.mean(),
            'Std Dev' : r.std(),
            'Skewness' : r.skew(),
            'Kurtosis' : r.kurtosis(),
            'JB Stat' : jb_stat,
            'JB p-Value' : jb_p,
            'AC(1)' : ac1,
            'Min' : r.min(),
            'Max' : r.max(),
            })
    return pd.DataFrame(rows).set_index('Ticker')


#--
#   Charts
#--

plt.rcParams.update({
    'font.family' : 'DejaVu Sans',
    'axes.titlesize' : 13,
    'axes.labelsize' : 11,
    'xtick.labelsize' : 9,
    'ytick.labelsize' : 9,
    'legend.fontsize' : 9,
    'figure.dpi' : 150,
    })

def chart_price_and_ma(ticker,prices, short_ma, long_ma, position, filename):
    """
    Figure 1(per stock) - price with MA lines and trade siggnals.
    This is your primary strategy illustration chart.
    """
    try:
        plt.style.use(CHART_STYLE)
    except:
        print('didnt work')
        pass

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (14, 8),
                                  gridspec_kw = {'height_ratios': [3,1]},
                                  sharex = True)

    # -- Top panel: price + MAs + shaded buy regions -- 
    ax1.plot(prices.index, prices, color='#333333', lw=1.0, 
             label = f'{ticker} Price', alpha = .8)
    ax1.plot(short_ma.index, short_ma, color='#2196F3', lw=1.5, 
            label = f'{SHORT_WINDOW}-Day SMA')
    ax1.plot(long_ma.index, long_ma, color='#FF5722', lw=1.5, 
            label = f'{LONG_WINDOW}-Day SMA')

    #Shade in market (long) periods 
    in_market = position.fillna(0)
    ax1.fill_between(prices.index, prices.min(), prices.max(),
                     where=(in_market == 1),
                     alpha = .08, color='#4CAF50', label = 'In-Market Period')

    ax1.set_title(f'{ticker} - Prices & Moving Average Crossover Strategy '
                  f'({SHORT_WINDOW}/{LONG_WINDOW}-Day SMA)',
                  fontweight='bold', pad=12)
    ax1.set_ylabel('Adjusted Closing Price (USD)')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # -- Bottom panel: position (0/1) --
    ax2.fill_between(position.index, 0, position.fillna(0),
                     color='#4CAF50', alpha = .7, step='post')
    ax2.set_ylabel("position\n(1=Long, 0=Cash)")
    ax2.set_yticks([0, 1])
    ax2.set_xlabel('Date')

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def chart_equity_curves(results_dict, prices_dict, filename):
    """
    Figure 2 - Equity curves: strategy vs buy-and-hold for all stocks.
    One of the most important charts for my paper.
    """
    try:
        plt.style.use(CHART_STYLE)
    except:
        pass

    fig, axes = plt.subplots(3, 2, figsize = (16, 14))
    axes = axes.flatten()

    colors_strat = '#1565C0'
    colors_bah = '#B71C1C'

    for i, (ticker, res) in enumerate(results_dict.items()):
        ax = axes[i]

        strat_curve = (1 + res['strat_ret'].dropna()).cumprod()
        bah_curve = (1 + res['bah_ret'].dropna()).cumprod()

        ax.plot(strat_curve.index, strat_curve, color=colors_strat,
                lw=1.8, label = 'MA Crossover Strategy')
        ax.plot(bah_curve.index, bah_curve, color=colors_bah,
                lw=1.8, label='Buy-and-Hold', linestyle='--')

        # mark in-sample / ount-of-samle split
        split_data = pd.Timestamp(INSAMPLE_END)
        ax.axvline(x=split_data, color = 'gray', linestyle=':', lw=1.2,
                   label='In/Out-of-samle-Split')

        ax.set_title(f'{ticker}', fontweight='bold')
        ax.set_ylabel('Cumulative Return (Base = 1.0)')
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.1f}x'))

    # hide unused subplot 
    if len(results_dict) < len(axes):
        axes[-1].set_visible(False)

    fig.suptitle(
        f'Figure 2: Equity Curves - MA Crossover Strategy vs. Buy-andHold\n'
        f'({SHORT_WINDOW}/{LONG_WINDOW}-Day SMA, Jan 2010 - Dec 2023)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def chart_portfolio_equity(port_strat, port_bah, filename):
    """
    Figure 3 - Equal-weighted portfolio equity curve + drawdown
    the portfolio - level summary chart for the paper 
    """

    try: 
        plt.style.use(CHART_STYLE)
    except:
        pass

    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (14, 9),
                                   gridspec_kw = {'height_ratios': [3, 1]},
                                   sharex = True)

    strat_curve = (1 + port_strat.dropna()).cumprod()
    bah_curve = (1 + port_bah.dropna()).cumprod()

    # drawdown calculation
    roll_max = strat_curve.cummax()
    drawdown = (strat_curve - roll_max) / roll_max

    ax1.plot(strat_curve.index, strat_curve, color='#1565C0',
             lw=2, label='MA Crossover Portfolio')
    ax1.plot(bah_curve.index, bah_curve, color='#B71C1C',
             lw=2, linestyle='--', label='Buy-and-Hold Portfolio')
    ax1.axvline(x=pd.Timestamp(INSAMPLE_END), color='gray',
                linestyle=':', lw=1.5, label='In/Out-of-Samle Split')
    ax1.set_title(
        f'Figure 3: Equal-Weighted Portfolio Equity Curve\n'
        f'MA Crossover({SHORT_WINDOW}/{LONG_WINDOW}-Day SMA) vs. Buy-and-Hold',
        fontweight='bold', pad=12)
    ax1.set_ylabel('Cumulative Portfolio Return (Base = 1.0)')
    ax1.legend()
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.1f}x'))

    ax2.fill_between(drawdown.index, drawdown, 0,
                     color='#B71C1C', alpha = .5)
    ax2.set_ylabel('Strategy\nDrawdown')
    ax2.set_xlabel('Date')
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def chart_performance_bar(summary_df, filename):
    """
    Figure 4 - Bar chart comparing Sharge ratios and annual returns.
    A Clean summary visual for results section
    """

    try: 
        plt.style.use(CHART_STYLE)
    except:
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    tickers = summary_df['Ticker'].tolist()
    x = np.arange(len(tickers))
    w=.35

    #sharpe ratios
    ax1.bar(x - w/2, summary_df['Strategy Sharpe'], w, 
            label='MA Strategy', color='#1565C0', alpha=.85)
    ax1.bar(x + w/2, summary_df['B&H Sharpe'], w,
            label='Buy-and-Hold', color='#B71C1C', alpha =.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers)
    ax1.axhline(0, color='black', lw=.8)
    ax1.set_title('Sharpe Ratio Comparison', fontweight='bold')
    ax1.set_ylabel('Annualized Sharpe Ratio')
    ax1.legend()

    #annualized returns 
    ax2.bar(x - w/2, summary_df['Strategy Ann. Return'] * 100, w,
            label='MA Strategy', color='#1565C0', alpha=.85)
    ax2.bar(x + w/2, summary_df['B&H Ann. Return'] * 100, w,
            label='Buy-and-Hold', color='#B71C1C', alpha=.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers)
    ax2.axhline(0, color='black', lw=.8)
    ax2.set_title('Annualized Return Comparison', fontweight='bold')
    ax2.set_ylabel('Annualized Return (%)')
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    ax2.legend()

    fig.suptitle(
        f'Figure 4: Strategy vs. Buy-and-Hold Performance Summary\n'
        f'{SHORT_WINDOW}/{LONG_WINDOW}-Day MA Crossover, 2010-2023)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR,filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")

def chart_robustness(prices, filename):
    """
    Figure 5 - robustness: sharge ratio heatmap across MA window Combinations
    This directly mirrors the robustness checks in brock et al. (1992)
    """

    try: 
        plt.style.use(CHART_STYLE)
    except:
        pass

    short_windows = [10, 20, 30, 50]
    long_windows = [50, 100, 150, 200]

    #use first ticker for robustness illustration
    p = prices[TICKERS[0]]

    results_matrix = np.full((len(short_windows), len(long_windows)), np.nan)

    for i, sw in enumerate(short_windows): 
        for j, lw in enumerate(long_windows):
            if sw >= lw:
                continue
            sma_s = p.rolling(sw).mean()
            sma_l = p.rolling(lw).mean()
            sig = (sma_s > sma_l).astype(int).shift(1)
            dr = p.pct_change()
            tr = sig.diff().abs()
            sr = sig * dr - tr * TRANS_COST
            results_matrix[i, j] = sharpe(sr)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(results_matrix, cmap='RdYlGn', aspect='auto', 
                   vmin=-.5, vmax=1.5)
    ax.set_xticks(range(len(long_windows)))
    ax.set_yticks(range(len(short_windows)))
    ax.set_xticklabels([f'{w}d' for w in long_windows])
    ax.set_yticklabels([f'{w}d' for w in short_windows])
    ax.set_xlabel('Long MA Window')
    ax.set_ylabel('Short MA Window')
    ax.set_title(
        f'Figure 5: Robustness Check - Sharge Ratio Across MA Paarameter Combinations\n'
        f'({TICKERS[0]}, 2010-2023)',
        fontweight='bold', pad=12)

    for i in range(len(short_windows)):
        for j in range(len(long_windows)):
            if not np.isnan(results_matrix[i, j]):
                ax.text(j, i, f'{results_matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=10,
                        color='black', fontweight='bold')

    plt.colorbar(im, ax=ax, label='Annualized Sharge Ratio')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filename}")


#--
#   Table Printing 
#--

def fmt_pct(x): return f"{x*100:+.2f}%"
def fmt_dec(x): return f"{x:.4f}"
def fmt_int(x): return f"{int(x):,}"
def fmt_p(x): return f"{x:.4f}"
def fmt_sr(x): return f"{x:.3f}"

def print_table_1(desc_df):
    print("\n" + "=" * 80)
    print("TABLE 1: Descriptive Statistics of Daily Log Returns")
    print(f"    Sample Period: {START_DATE} to {END_DATE}")
    print("=" * 80)
    display = pd.DataFrame({
        'Mean' : desc_df['Mean (Daily)'].map(fmt_dec),
        'Std Dev' : desc_df['Std Dev'].map(fmt_dec),
        'Skewness' : desc_df['Skewness'].map(fmt_dec),
        'Kurtosis' : desc_df['Kurtosis'].map(fmt_dec),
        'JB Stat' : desc_df['JB Stat'].map(lambda x: f"{x:.1f}"),
        'JB p-Val' : desc_df['JB p-Value'].map(fmt_p),
        'AC(1)' : desc_df['AC(1)'].map(fmt_dec)
    })
    print(display.to_string())
    print("Notes: Mean and Std Dev are daily. JB = Jarque-Bera normality test.")
    print("     AC(1) = first-order autocorrelation of daily log returns.")

def print_table_2(summary_df):
    print("\n" + "="*80)
    print("TABLE 2: Strategy vs. Buy-and-Hold Performance")
    print(f"    {SHORT_WINDOW}/{LONG_WINDOW}-Day SMA Crossover | "
          f"Transaction Cost: {TRANS_COST *100:1f}% per trade")
    print('='*80)
    display = pd.DataFrame({
        'Ticker' : summary_df['Ticker'],
        'Strat Ann Ret' : summary_df['Strategy Ann. Return'].map(fmt_pct),
        'B&H Ann Ret' : summary_df['B&H Ann. Return'].map(fmt_pct),
        'Excess Ret' : summary_df['Excess Return (Ann.)'].map(fmt_pct),
        'Strat Sharpe' : summary_df['Strategy Sharpe'].map(fmt_sr),
        'B&H Sharpe' : summary_df['B&H Sharpe'].map(fmt_sr),
        'Max DD (Strat)' : summary_df['Strategy Max Drawdown'].map(fmt_pct),
        'Max DD (B&H)' : summary_df['B&H Max Drawdown'].map(fmt_pct),
        'N Trades' : summary_df['Number of Trades'].map(fmt_int),
        }).set_index('Ticker')
    print(display.to_string())
    print("\nNotes: Ann Ret = annualized return. Sharpe ratio uses RF = "
          f"{RISK_FREE*100:.1f}% p.a.")
    print("     Max DD = maximum peak-to-through drawdown.")

def print_table_3(summary_df):
    print('\n' + '='*80)
    print('TABLE 3: Statistical Significance of Daily Excess Returns')
    print("     H0: Mean daily excess return = 0 (Two-tailed t-test)")
    print('='*80)
    display = pd.DataFrame({
        'Ticker' : summary_df['Ticker'],
        'Mean Excess' : summary_df['Mean Daily Excess Ret'].map(fmt_dec),
        'Std Excess' : summary_df['Std Daily Excess Ret'].map(fmt_dec),
        't-Statistic' : summary_df['t-Statistic'].map(fmt_sr),
        'p-Value' : summary_df['p-Value'].map(fmt_p),
        '95% CI Lower' : summary_df['95% CI Lower'].map(fmt_dec),
        '95% CI Upper' : summary_df['95% CI Upper'].map(fmt_dec),
        'Sig. at 5%?' : summary_df['Significant (5%)'],
        }).set_index('Ticker')
    print(display.to_string())
    print("\nNotes: Excess return = strategy daily return minus buy-and-hold return.")
    print("     ***p<.01, **p.05, *p<.10")

def print_table_4(results_dict):
    """in-simple vs out-of-sample performance split."""
    print("\n"+'='*80)
    print("TABLE 4: In-Sample vs. Out-of-Sample Performance")
    print(f"    In-Sample: {START_DATE} to {INSAMPLE_END}")
    print(f"    Out-of-Sample: {INSAMPLE_END} to {END_DATE}")
    print("="*80)
    rows = []
    for ticker, res in results_dict.items():
        split = pd.Timestamp(INSAMPLE_END)
        for label, mask in [('In-Sample', res['strat_ret'].index <= split),
                            ('Out-of-Sample', res['strat_ret'].index > split)]:
            sr = res['strat_ret'][mask]
            bh = res['bah_ret'][mask]
            rows.append({
                'Ticker' : ticker,
                'Period' : label,
                'Strat Sharpe' : fmt_sr(sharpe(sr)),
                'B&H Sharpe' : fmt_sr(sharpe(bh)),
                'Strat Ann Ret' : fmt_pct(annualize(sr)),
                'B&H Ann Ret' : fmt_pct(annualize(bh)),
                })
    df = pd.DataFrame(rows).set_index(['Ticker', 'Period'])
    print(df.to_string())
    print("\nNotes: Out-of-sample period used for validation only --"
    "no parameter re-fitting.")

#--
#   Main Execution
#--

def main():
    print("\n" + '='*80)
    print(" QUANTITATIVE TRADING STRATEGY BACKTEST")
    print(" Moving Average Crossover | Brock et al. (1992) Methodology")
    print('='*80)

    # -- step 1: Download data
    prices = download_data(TICKERS, START_DATE, END_DATE)

    # -- step 2: Descriptive sttatistics
    print("\n[2/6] Computing descriptive statistics ...")
    desc = descriptive_stats(prices)
    print_table_1(desc)

    # -- step 3: Run strategy for each ticker 
    print("\n[3/6] Running strategy backtest ...")
    results = {}
    summary_rows = []

    port_strat_list = []
    port_bah_list = []

    for ticker in TICKERS: 
        p = prices[ticker]
        short_ma, long_ma, position = compute_signals(p, SHORT_WINDOW, LONG_WINDOW)
        strat_ret, bah_ret, trades = compute_returns(p, position, TRANS_COST)

        results[ticker] = {
            'prices' : p,
            'short_ma' : short_ma,
            'long_ma' : long_ma,
            'position' : position,
            'strat_ret' : strat_ret,
            'bah_ret' : bah_ret,
            'trades' : trades,
            }

        perf = performance_summary(ticker, strat_ret, bah_ret, trades)
        summary_rows.append(perf)
        port_strat_list.append(strat_ret)
        port_bah_list.append(bah_ret)

        print(f"    {ticker}: Strategy Sharpe ="
              f"{perf['Strategy Sharpe']:.3f}|"
              f"Trades = {int(perf['Number of Trades'])}")

    summary_df = pd.DataFrame(summary_rows)

    # equal weighted portfolio 
    port_strat = pd.concat(port_strat_list, axis=1).mean(axis=1)
    port_bah = pd.concat(port_bah_list, axis=1).mean(axis=1)

    # -- step 4: Print  tables
    print('\n[4/6] Printing performance tables...')
    print_table_2(summary_df)
    print_table_3(summary_df)
    print_table_4(results)

    # Portfolio level stats
    print('\n'+ '='*80)
    port_perf = performance_summary("PORTFOLIO", port_strat, port_bah,
                                    pd.Series(0, index=port_strat.index))
    for k,v in port_perf.items():
        if k =='Ticker': continue 
        if isinstance(v, float):
            print(f" {k:<30} {v:>12.4f}")
        else:
            print(f" {k:<30} {str(v):>12}")

    # -- step 5: Generate charts
    print("\n[5/6] Generating charts...")

    #Figure 1:  pre-stock price + MA + signals 
    for ticker, res in results.items():
        chart_price_and_ma(
            ticker,
            res['prices'], res['short_ma'], res['long_ma'], res['position'],
            filename=f"fig1_{ticker}_price_and_MA.png"
            )

    #Figure 2: all equity curves 
    chart_equity_curves(results, prices, filename='fig2_equity_curves.png')

    #Figure 3: portfolio equity + drawdown
    chart_portfolio_equity(port_strat, port_bah,
                           filename="fig3_portfolio_equity.png")

    #Figure 4: bar chart comparison 
    chart_performance_bar(summary_df, filename='fig4_performance_comparison.png')

    #Figure 5: Robustness heatmap
    chart_robustness(prices, filename="fig5_robustness_heatmap.png")

    # -- step 6: Done
    print('\n[6/6]All outputs complete!')
    print('='*80)
    print('\n CHARTS SAVED:')

    for ticker in TICKERS:
        print(f"    fig1_{ticker}_price_and_MA.png -> Paper: Figure 1 (results 5.2")

    print(' fig2_equity_curves.png      -> Paper: Figure 2 (results 5.2)')
    print(' fig3_portfolio_equity.png      -> Paper: Figure 3 (results 5.3)')
    print(' fig4_performance_comparison.png -> Paper: Figure 4 (results 5.2)')
    print(' fig5_robustness_heatmap.png      -> Paper: Figure 5 (results 5.6)')

    print('\n TABLES PRINTED ABOVE:')
    print(' table 1 -> Paper: descriptive Statistics (5.1)')
    print(' table 2 -> Paper: Strategy vs B&H        (5.2)')
    print(' table 3 -> Paper: Statistical Significance (5.4)')
    print(' table 4 -> Paper: In/Out-of-Sample Split (5.5)')

    print("\n Copy terminal output into my paper's Empirical Results section.")
    print('='*80 + '\n')



if __name__=="__main__":
    main()