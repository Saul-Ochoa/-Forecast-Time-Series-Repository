"""
================================================================================
  S&P 500 — Análisis de Regímenes Estructurales con Ridge Regularization
  Versión: 2.0 (Production-Grade)
================================================================================
  Mejoras respecto a v1:
  - Indicadores macro ampliados: Yield Curve, HYG, Breadth proxy, Skew/Kurt
  - VIX term structure (VIX3M/VIX), RSI, autocorrelación rolling
  - RidgeCV corregido (sin store_cv_values con scoring custom)
  - Validación walk-forward out-of-sample (25% de cada régimen)
  - IC de Sharpe por régimen (bootstrap 1000 iter)
  - Dashboard de 4 paneles con tema oscuro institucional
  - CSV completo + log de diagnósticos
================================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ruptures as rpt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

START_DATE      = "2000-01-01"
END_DATE        = pd.Timestamp.today().strftime("%Y-%m-%d")
MIN_REGIME_DAYS = 252  # ~6 meses mínimo por régimen - 126
PELT_JUMP       = 5
PELT_PEN_MULT   = 2           # pen = mult * log(N). puede variar
OOS_FRACTION    = 0.25         # 25% hold-out dentro de cada régimen
BOOTSTRAP_N     = 1000         # iteraciones para IC de Sharpe
QUIEBRES_N      =8
RIDGE_ALPHAS    = np.logspace(-4, 3, 50)

DARK_BG         = "#0d1117"
PANEL_BG        = "#161b22"
ACCENT_BLUE     = "#58a6ff"
ACCENT_GREEN    = "#3fb950"
ACCENT_RED      = "#f85149"
ACCENT_YELLOW   = "#e3b341"
ACCENT_PURPLE   = "#bc8cff"
TEXT_PRIMARY    = "#e6edf3"
TEXT_SECONDARY  = "#8b949e"
GRID_COLOR      = "#21262d"

REGIME_COLORS   = [
    "#58a6ff", "#3fb950", "#f85149",
    "#e3b341", "#bc8cff", "#ffa657",
    "#79c0ff", "#56d364", "#ff7b72"
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. DESCARGA DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

def download_data() -> pd.DataFrame:
    """
    Descarga SPX, VIX, VIX3M, TNX (10Y yield), IRX (3M yield), HYG, IEI.
    Aplica shift(1) donde corresponde para evitar look-ahead bias.
    """
    print("📥  Descargando datos de mercado...")

    tickers = {
        "^GSPC":  "spx",
        "^VIX":   "vix",
        "^VIX3M": "vix3m",   # VIX a 3 meses (term structure)
        "^TNX":   "tnx",     # Yield 10Y
        "^IRX":   "irx",     # Yield 3M
        "HYG":    "hyg",     # High-Yield ETF
        "IEI":    "iei",     # Investment-Grade 3-7Y ETF
    }

    raw = {}
    for ticker, name in tickers.items():
        try:
            df_t = yf.download(ticker, start=START_DATE, end=END_DATE,
                               auto_adjust=True, progress=False)
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = df_t.columns.get_level_values(0)
            raw[name] = df_t["Close"]
            print(f"   ✓ {ticker:8s}  ({len(df_t)} obs)")
        except Exception as e:
            print(f"   ✗ {ticker:8s}  Error: {e}")
            raw[name] = None

    df = pd.DataFrame()

    # Precios SPX
    df["close"]  = raw["spx"]
    df["ret"]    = np.log(df["close"]).diff()

    # VIX (shift 1 → sin look-ahead)
    df["vix"]    = raw["vix"].shift(1) if raw["vix"] is not None else np.nan

    # VIX Term Structure: VIX3M / VIX − 1
    # > 0 → contango (mercado tranquilo), < 0 → backwardation (pánico)
    if raw["vix3m"] is not None and raw["vix"] is not None:
        df["vix_ts"] = (raw["vix3m"] / raw["vix"] - 1).shift(1)
    else:
        df["vix_ts"] = np.nan

    # Yield Curve: TNX − IRX (10Y − 3M)
    # < 0 → invertida → señal recesiva
    if raw["tnx"] is not None and raw["irx"] is not None:
        df["yield_curve"] = (raw["tnx"] - raw["irx"]).shift(1)
    else:
        df["yield_curve"] = np.nan

    # Credit Spread proxy: log(HYG/IEI) → risk-on / risk-off
    if raw["hyg"] is not None and raw["iei"] is not None:
        df["credit_spread"] = np.log(raw["hyg"] / raw["iei"]).shift(1)
    else:
        df["credit_spread"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["close", "ret"])

    print(f"\n   Rango de datos: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"   Total de observaciones: {len(df)}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUCCIÓN DE FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Construye el conjunto completo de features con justificación económica.
    Devuelve df enriquecido y lista de columnas de features.
    """
    # ── Momentum / Mean-Reversion ──────────────────────────────────────────
    df["lag1"]     = df["ret"].shift(1)               # autocorrelación orden 1
    df["lag2"]     = df["ret"].shift(2)               # autocorrelación orden 2
    df["lag5"]     = df["ret"].shift(5)               # efecto semana

    # ── Volatilidad ────────────────────────────────────────────────────────
    df["vol20"]    = df["ret"].rolling(20).std()       # vol realizada 20d
    df["vol60"]    = df["ret"].rolling(60).std()       # vol realizada 60d
    df["vol_ratio"]= df["vol20"] / df["vol60"]         # régimen vol: >1 = stress

    # ── Apalancamiento (leverage effect) ──────────────────────────────────
    df["abs_lag1"] = df["lag1"].abs()                  # |ret_t-1| → ARCH effect
    df["neg_lag1"] = (df["lag1"] < 0).astype(float) * df["lag1"]  # asimetría

    # ── Tendencia ─────────────────────────────────────────────────────────
    df["ma20"]     = df["close"] / df["close"].rolling(20).mean() - 1
    df["ma60"]     = df["close"] / df["close"].rolling(60).mean() - 1
    df["ma200"]    = df["close"] / df["close"].rolling(200).mean() - 1  # bull/bear

    # ── RSI (14 días) ─────────────────────────────────────────────────────
    delta  = df["ret"].fillna(0)
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).shift(1)     # shift→sin look-ahead

    # ── Skewness y Kurtosis rolling (tail risk) ───────────────────────────
    df["skew20"]   = df["ret"].rolling(20).skew().shift(1)
    df["kurt20"]   = df["ret"].rolling(20).kurt().shift(1)

    # ── Autocorrelación rolling (trending vs mean-reverting) ───────────────
    df["autocorr5"] = (
        df["ret"]
        .rolling(20)
        .apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
        .shift(1)
    )

    # ── VIX enriched ──────────────────────────────────────────────────────
    df["vix_ma20"]  = df["vix"] / df["vix"].rolling(20).mean() - 1   # shock de miedo
    df["vix_vol"]   = df["vix"] * df["vol20"]                         # interacción

    df["resid_lag1"] = 0.0 # placeholde
    
    # ── Macro (ya con shift en download) ──────────────────────────────────
    # vix_ts, yield_curve, credit_spread ya están en df
    
    # ── Limpieza final ─────────────────────────────────────────────────────
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = [
        # Momentum
        "lag1", "lag2", "lag5",
        # Volatilidad
        "vol20", "vol60", "vol_ratio",
        # Leverage
        "abs_lag1", "neg_lag1",
        # Tendencia
        "ma20", "ma60", "ma200",
        # Técnicos
        "rsi14",
        # Tail risk
        "skew20", "kurt20",
        # Régimen
        "autocorr5",
        # VIX
        "vix", "vix_ma20", "vix_vol", "vix_ts",
        # Macro
        "yield_curve", "credit_spread",
        # residuos 
        "resid_lag1"
    ]

    # Eliminar columnas con demasiados NaN
    valid_cols = [c for c in feature_cols if c in df.columns
                  and df[c].notna().mean() > 0.5]
    missing = set(feature_cols) - set(valid_cols)
    if missing:
        print(f"\n   ⚠️  Features excluidas por NaN > 50%: {missing}")

    df = df.dropna(subset=valid_cols + ["ret"])

    print(f"\n   Features activas: {len(valid_cols)}")
    print(f"   Observaciones post-limpieza: {len(df)}")

    return df, valid_cols


# ──────────────────────────────────────────────────────────────────────────────
# 3. DETECCIÓN DE RUPTURAS (PELT vs BINSEG) — versión l2
# ──────────────────────────────────────────────────────────────────────────────

def detect_breakpoints(df: pd.DataFrame, feature_cols: list[str],
                       method: str = "binseg") -> list[int]:
    """
    Detecta quiebres estructurales con l2 (cambio de media).

    method = "binseg" → rápido, controlas nº de quiebres
    method = "pelt" → óptimo exacto, controlas penalización
    """
    print(f"\n🔍 Detectando rupturas estructurales ({method.upper()})...")

    y = df["ret"].values.reshape(-1, 1) # ← l2 solo usa y

    if method.lower() == "binseg":
        # ── BINSEG l2 ─────────────────────
        algo = rpt.Binseg(model="l2", min_size=MIN_REGIME_DAYS, jump=PELT_JUMP).fit(y)
        bkps = algo.predict(n_bkps=QUIEBRES_N)

    elif method.lower() == "pelt":
        # ── PELT l2 ─────────────────────
        algo = rpt.Pelt(model="l2", min_size=MIN_REGIME_DAYS, jump=PELT_JUMP).fit(y)
        pen = PELT_PEN_MULT * np.log(len(y))
        bkps = algo.predict(pen=pen)

    else:
        raise ValueError("method debe ser 'binseg' o 'pelt'")

    print(f" Quiebres encontrados: {len(bkps) - 1}")
    for b in bkps[:-1]:
        print(f" → {df.index[b].date()}")

    return bkps


# ──────────────────────────────────────────────────────────────────────────────
# 4. BOOTSTRAP SHARPE RATIO
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_sharpe(returns: np.ndarray, n: int = BOOTSTRAP_N,
                     alpha: float = 0.05) -> dict:
    """
    Calcula Sharpe anualizado con IC bootstrap (percentil).
    """
    sharpes = []
    for _ in range(n):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        std    = sample.std()
        if std > 0:
            sharpes.append(sample.mean() / std * np.sqrt(252))
    sharpes = np.array(sharpes)
    return {
        "mean":    sharpes.mean(),
        "lower":   np.percentile(sharpes, alpha / 2 * 100),
        "upper":   np.percentile(sharpes, (1 - alpha / 2) * 100),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. ESTIMACIÓN POR REGÍMENES
# ──────────────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox

def estimate_regimes(
    df: pd.DataFrame,
    bkps: list[int],
    feature_cols: list[str]
) -> list[dict]:
    """
    Estima un modelo Ridge para cada régimen.

    Flujo:
        1. Divide cada régimen en IS/OOS.
        2. Escala usando SOLO el conjunto IS.
        3. Entrena un Ridge inicial.
        4. Calcula residuos IS.
        5. Añade resid_lag1.
        6. Reentrena Ridge final.
        7. Evalúa OOS.
        8. Calcula métricas y devuelve resultados.
    """

    y_all = df["ret"].values

    resultados = []

    print("\n" + "=" * 80)
    print(" ESTIMACIÓN POR REGÍMENES — RIDGE + RESID_LAG1")
    print("=" * 80)

    for i, b in enumerate(bkps):

        inicio = 0 if i == 0 else bkps[i - 1]
        fin = b

        yr = y_all[inicio:fin]
        idx = df.index[inicio:fin]

        Xr = df.iloc[inicio:fin][feature_cols].values

        n_obs = len(yr)

        if n_obs < 10:
            continue

        split = max(10, int(n_obs * (1 - OOS_FRACTION)))

        if split >= n_obs:
            split = n_obs - 1

        # ============================================================
        # Split IS / OOS
        # ============================================================

        X_is = Xr[:split]
        X_oos = Xr[split:]

        y_is = yr[:split]
        y_oos = yr[split:]

        # ============================================================
        # Escalado SOLO con IS (evita data leakage)
        # ============================================================

        scaler = StandardScaler()

        X_is = scaler.fit_transform(X_is)

        if len(X_oos):
            X_oos = scaler.transform(X_oos)

        # ============================================================
        # Ridge inicial
        # ============================================================

        ridge = RidgeCV(
            alphas=RIDGE_ALPHAS,
            cv=TimeSeriesSplit(n_splits=5)
        )

        ridge.fit(X_is, y_is)

        yhat_is = ridge.predict(X_is)

        resid_is = y_is - yhat_is

        # ============================================================
        # Feature resid_lag1
        # ============================================================

        resid_lag_is = np.concatenate(([0.0], resid_is[:-1]))

        X_is_corr = np.column_stack((X_is, resid_lag_is))

        # ============================================================
        # Ridge final
        # ============================================================

        ridge_corr = RidgeCV(
            alphas=RIDGE_ALPHAS,
            cv=TimeSeriesSplit(n_splits=5)
        )

        ridge_corr.fit(X_is_corr, y_is)

        yhat_is = ridge_corr.predict(X_is_corr)

        resid_is = y_is - yhat_is

        # ============================================================
        # Predicción OOS
        # ============================================================

        if len(y_oos):

            yhat_oos = np.zeros(len(y_oos))

            resid_prev = resid_is[-1]

            for t in range(len(y_oos)):

                x = np.append(X_oos[t], resid_prev).reshape(1, -1)

                yhat_oos[t] = ridge_corr.predict(x)[0]

                # Ahora ya conocemos el error real
                resid_prev = y_oos[t] - yhat_oos[t]

        else:

            yhat_oos = np.array([])

        # ============================================================
        # Métricas
        # ============================================================

        r2_is = r2_score(y_is, yhat_is)

        r2_oos = (
            r2_score(y_oos, yhat_oos)
            if len(y_oos) > 1
            else np.nan
        )

        resid = y_is - yhat_is

        # ============================================================
        # Ljung-Box
        # ============================================================

        try:

            lag = min(10, len(resid) // 5)

            if lag < 1:
                lb_pval = np.nan
            else:
                lb = acorr_ljungbox(
                    resid,
                    lags=lag,
                    return_df=True
                )
                lb_pval = lb["lb_pvalue"].iloc[-1]

        except Exception:

            lb_pval = np.nan

        # ============================================================
        # Sharpe Bootstrap
        # ============================================================

        sh = bootstrap_sharpe(yr)

        # ============================================================
        # Coeficientes
        # ============================================================

        coefs = {
            col: ridge_corr.coef_[j]
            for j, col in enumerate(feature_cols)
        }

        coefs["resid_lag1"] = ridge_corr.coef_[-1]

        # ============================================================
        # Estadísticas del régimen
        # ============================================================

        ann_ret = yr.mean() * 252

        ann_vol = yr.std() * np.sqrt(252)

        max_dd = _max_drawdown(yr)

        skew_r = pd.Series(yr).skew()

        kurt_r = pd.Series(yr).kurt()

        # ============================================================
        # Resultado
        # ============================================================

        res = {

            "regimen": i + 1,

            "inicio": idx[0].date(),

            "fin": idx[-1].date(),

            "n_obs": n_obs,

            "n_is": split,

            "n_oos": n_obs - split,

            "alpha_opt": ridge_corr.alpha_,

            "intercepto": ridge_corr.intercept_,

            "r2_is": r2_is,

            "r2_oos": r2_oos,

            "lb_pval": lb_pval,

            "ann_ret": ann_ret,

            "ann_vol": ann_vol,

            "sharpe_mean": sh["mean"],

            "sharpe_lower": sh["lower"],

            "sharpe_upper": sh["upper"],

            "max_dd": max_dd,

            "skew": skew_r,

            "kurt": kurt_r,

            **coefs,

            "_idx": idx,

            "_yr": yr,

            "_yhat_is": yhat_is,

            "_yhat_oos": yhat_oos,

            "_resid": resid,

            "_split": split,

            "_ridge": ridge_corr,

            "_feature_cols": feature_cols,

            "_scaler": scaler

        }

        resultados.append(res)

        _print_regime(res)

    return resultados



def _max_drawdown(returns: np.ndarray) -> float:
    cum   = np.cumprod(1 + returns)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak) / peak
    return dd.min()


def _print_regime(res: dict):
    print(f"\n{'─'*80}")
    print(f"  RÉGIMEN {res['regimen']:>2d}  │  "
          f"{res['inicio']} → {res['fin']}  │  {res['n_obs']} obs")
    print(f"{'─'*80}")
    print(f"  Ridge α óptimo : {res['alpha_opt']:.4f}")
    print(f"  R² IS          : {res['r2_is']:.4f}   |   R² OOS: {res['r2_oos']:.4f}")
    lb_str = f"{res['lb_pval']:.4f}" if not np.isnan(res['lb_pval']) else "N/A"
    print(f"  Ljung-Box p    : {lb_str}  "
          f"{'(residuos OK ✓)' if not np.isnan(res['lb_pval']) and res['lb_pval'] > 0.05 else '(autocorr detectada ⚠️)'}")
    print(f"  Retorno anual  : {res['ann_ret']*100:+.2f}%   "
          f"Volatilidad: {res['ann_vol']*100:.2f}%")
    print(f"  Sharpe         : {res['sharpe_mean']:.3f}  "
          f"IC 95%: [{res['sharpe_lower']:.3f}, {res['sharpe_upper']:.3f}]")
    print(f"  Max Drawdown   : {res['max_dd']*100:.2f}%   "
          f"Skew: {res['skew']:.3f}   Kurt: {res['kurt']:.3f}")
    print(f"\n  Top coeficientes (|β| mayores):")
    feat_cols = res["_feature_cols"]
    coef_vals = [(c, res[c]) for c in feat_cols]
    coef_vals.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, val in coef_vals[:6]:
        bar = "█" * int(min(abs(val) * 500, 20))
        sign = "+" if val >= 0 else "-"
        print(f"    {col:>16s}: {sign}{abs(val):.6f}  {bar}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. DASHBOARD DE VISUALIZACIÓN (4 paneles)
# ──────────────────────────────────────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, bkps: list[int],
                   resultados: list[dict]):
    """
    Panel 1: Precio SPX + regímenes + rupturas
    Panel 2: Retornos con shading por régimen
    Panel 3: Coeficientes Ridge heatmap por régimen
    Panel 4: Métricas de riesgo comparativas (Sharpe, Drawdown, Volatilidad)
    """
    fig = plt.figure(figsize=(22, 18), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97,
                            top=0.94, bottom=0.05)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _style_ax(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.label.set_color(TEXT_SECONDARY)
        ax.xaxis.label.set_color(TEXT_SECONDARY)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)
        if title:
            ax.set_title(title, color=TEXT_PRIMARY, fontsize=10,
                         fontweight="bold", pad=8)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)

    # ── Panel 1 (span 2 cols): Precio + regímenes ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style_ax(ax1, title="S&P 500 — Regímenes Estructurales (PELT)")

    ax1.plot(df.index, df["close"], color=TEXT_PRIMARY, lw=1.0, zorder=3)

    for i, res in enumerate(resultados):
        idx   = res["_idx"]
        color = REGIME_COLORS[i % len(REGIME_COLORS)]
        ax1.axvspan(idx[0], idx[-1], alpha=0.12, color=color, zorder=1)
        mid   = idx[len(idx) // 2]
        ypos  = df["close"].max() * 0.98
        ax1.text(mid, ypos, f"R{res['regimen']}",
                 color=color, fontsize=7, ha="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc=DARK_BG, ec=color, lw=0.8))

    for b in bkps[:-1]:
        ax1.axvline(df.index[b], color=ACCENT_RED, lw=1.0,
                    linestyle="--", alpha=0.9, zorder=4)

    ax1.set_ylabel("Precio (USD)", fontsize=8, color=TEXT_SECONDARY)

    # ── Panel 2: Retornos log + VIX overlay ───────────────────────────────
    ax2  = fig.add_subplot(gs[1, :])
    ax2b = ax2.twinx()
    _style_ax(ax2, title="Retornos Diarios Log  +  VIX (eje derecho)")

    for i, res in enumerate(resultados):
        idx   = res["_idx"]
        color = REGIME_COLORS[i % len(REGIME_COLORS)]
        sub   = df.loc[idx[0]:idx[-1], "ret"]
        ax2.bar(sub.index, sub.values, color=color, alpha=0.4, width=1.0)

    if "vix" in df.columns:
        ax2b.plot(df.index, df["vix"], color=ACCENT_YELLOW,
                  lw=0.8, alpha=0.7, label="VIX")
        ax2b.set_ylabel("VIX", color=ACCENT_YELLOW, fontsize=8)
        ax2b.tick_params(axis="y", colors=ACCENT_YELLOW, labelsize=7)
        ax2b.spines["right"].set_color(GRID_COLOR)

    ax2.set_ylabel("Retorno Log", fontsize=8, color=TEXT_SECONDARY)
    ax2.axhline(0, color=TEXT_SECONDARY, lw=0.5, alpha=0.5)

    # ── Panel 3: Coeficientes heatmap ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    _style_ax(ax3, title="Coeficientes Ridge por Régimen (estandarizados)")

    feature_cols = resultados[0]["_feature_cols"]
    n_reg  = len(resultados)
    n_feat = len(feature_cols)
    coef_matrix = np.zeros((n_feat, n_reg))

    for j, res in enumerate(resultados):
        for k, col in enumerate(feature_cols):
            coef_matrix[k, j] = res.get(col, 0.0)

    # Normalizar por fila para visualización
    vmax = np.abs(coef_matrix).max()

    im = ax3.imshow(coef_matrix, aspect="auto",
                    cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")

    ax3.set_yticks(range(n_feat))
    ax3.set_yticklabels(feature_cols, fontsize=7, color=TEXT_PRIMARY)
    ax3.set_xticks(range(n_reg))
    ax3.set_xticklabels([f"R{r['regimen']}\n{str(r['inicio'])[:7]}"
                          for r in resultados], fontsize=7, color=TEXT_PRIMARY)

    cbar = fig.colorbar(im, ax=ax3, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    cbar.ax.set_ylabel("Coef.", color=TEXT_SECONDARY, fontsize=7)

    # Anotaciones de valores
    for k in range(n_feat):
        for j in range(n_reg):
            val = coef_matrix[k, j]
            ax3.text(j, k, f"{val:.3f}", ha="center", va="center",
                     fontsize=5.5, color="white" if abs(val) > vmax * 0.5 else "black")

    # ── Panel 4a: Sharpe por régimen ──────────────────────────────────────
    ax4a = fig.add_subplot(gs[3, 0])
    _style_ax(ax4a, title="Sharpe Ratio  +  IC 95% Bootstrap", ylabel="Sharpe")

    x_pos = range(n_reg)
    sh_means  = [r["sharpe_mean"]  for r in resultados]
    sh_lowers = [r["sharpe_mean"] - r["sharpe_lower"] for r in resultados]
    sh_uppers = [r["sharpe_upper"] - r["sharpe_mean"] for r in resultados]
    bar_colors = [ACCENT_GREEN if s >= 0 else ACCENT_RED for s in sh_means]

    bars = ax4a.bar(x_pos, sh_means, color=bar_colors, alpha=0.75, width=0.6)
    ax4a.errorbar(x_pos, sh_means,
                  yerr=[sh_lowers, sh_uppers],
                  fmt="none", color=TEXT_PRIMARY, capsize=4, lw=1.5)
    ax4a.axhline(0, color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax4a.set_xticks(x_pos)
    ax4a.set_xticklabels([f"R{r['regimen']}" for r in resultados], fontsize=8)

    for bar, val in zip(bars, sh_means):
        ax4a.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.02 * np.sign(val),
                  f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                  color=TEXT_PRIMARY, fontsize=7, fontweight="bold")

    # ── Panel 4b: R², Drawdown, Volatilidad ───────────────────────────────
    ax4b = fig.add_subplot(gs[3, 1])
    _style_ax(ax4b, title="R² OOS  |  Max Drawdown  |  Vol Anual")

    x  = np.arange(n_reg)
    w  = 0.25

    r2_oos = [r["r2_oos"] if not np.isnan(r["r2_oos"]) else 0 for r in resultados]
    mdd    = [abs(r["max_dd"]) for r in resultados]
    vols   = [r["ann_vol"] for r in resultados]

    # Normalizar para comparación visual en mismo eje
    ax4b.bar(x - w,   r2_oos, width=w, color=ACCENT_BLUE,   alpha=0.8, label="R² OOS")
    ax4b.bar(x,       mdd,    width=w, color=ACCENT_RED,    alpha=0.8, label="|Max DD|")
    ax4b.bar(x + w,   vols,   width=w, color=ACCENT_YELLOW, alpha=0.8, label="Vol Anual")

    ax4b.set_xticks(x)
    ax4b.set_xticklabels([f"R{r['regimen']}" for r in resultados], fontsize=8)
    leg = ax4b.legend(fontsize=7, facecolor=PANEL_BG,
                      edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)

    # ── Título global ──────────────────────────────────────────────────────
    fig.suptitle(
        "S&P 500 — Análisis de Regímenes Estructurales  ·  Ridge L2  ·  Walk-Forward OOS",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=0.975
    )

    plt.savefig("sp500_regimes_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print("\n   Dashboard guardado: sp500_regimes_dashboard.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. EXPORTACIÓN DE RESULTADOS
# ──────────────────────────────────────────────────────────────────────────────

def export_results(resultados: list[dict], feature_cols: list[str]):
    """
    Exporta tabla de regímenes a CSV (sin columnas internas _*).
    """
    rows = []
    for res in resultados:
        row = {k: v for k, v in res.items() if not k.startswith("_")}
        rows.append(row)

    tabla = pd.DataFrame(rows)
    tabla.to_csv("regimenes_sp500_ridge_v2.csv", index=False)

    print("\n" + "="*80)
    print("  TABLA RESUMEN DE REGÍMENES")
    print("="*80)

    cols_show = ["regimen", "inicio", "fin", "n_obs",
                 "ann_ret", "ann_vol", "sharpe_mean", "max_dd",
                 "r2_is", "r2_oos", "alpha_opt", "lb_pval"]
    cols_show = [c for c in cols_show if c in tabla.columns]

    print(tabla[cols_show].round(4).to_string(index=False))
    print(f"\n   ✓ CSV guardado: regimenes_sp500_ridge_v2.csv")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print("  S&P 500 — STRUCTURAL REGIME ANALYSIS  v2.0  (Production-Grade)")
    print("="*80 + "\n")

    # 1. Datos
    df = download_data()

    # 2. Features
    df, feature_cols = build_features(df)

    # 3. Rupturas
    bkps = detect_breakpoints(df, feature_cols)

    # 4. Estimación
    resultados = estimate_regimes(df, bkps, feature_cols)

    # 5. Dashboard
    print("\n📊  Generando dashboard...")
    plot_dashboard(df, bkps, resultados)

    # 6. Export
    export_results(resultados, feature_cols)

    print("\n✅  Análisis completo.\n")


if __name__ == "__main__":
    main()