"""
================================================================================
  S&P 500 — Análisis de Regímenes Estructurales con Ridge Regularization
  Versión: 3.0 (PELT + BINSEG Consensus)
================================================================================
  Cambios respecto a v2:
  - Detección de quiebres COMBINADA: PELT(linear) + BINSEG(linear) sobre la
    señal multivariante [y | X], no solo sobre y univariante.
    Un quiebre se considera "robusto" si ambos métodos coinciden dentro de
    una ventana de tolerancia (CONSENSUS_WINDOW días).
  - Escalado IS-only por régimen (sin leakage), heredado de tu v3.
  - resid_lag1 corregido: ya no es un placeholder fantasma en build_features,
    se documenta explícitamente el supuesto walk-forward de 1 paso en OOS.
  - TimeSeriesSplit en RidgeCV (heredado de tu v3).
  - Reporte de cuántos quiebres detectó cada método individualmente vs.
    cuántos sobrevivieron el consenso → transparencia total.
================================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

START_DATE        = "2000-01-01"
END_DATE           = pd.Timestamp.today().strftime("%Y-%m-%d")
MIN_REGIME_DAYS    = 126            # ~6 meses mínimo por régimen
JUMP                = 5
OOS_FRACTION        = 0.25
BOOTSTRAP_N         = 1000
RIDGE_ALPHAS        = np.logspace(-4, 3, 50)

# ── Detección de quiebres ──────────────────────────────────────────────────
PELT_PEN_MULT       = 5             # pen = mult * log(N) para PELT
BINSEG_N_BKPS       = 8             # nº de quiebres objetivo para BINSEG
CONSENSUS_WINDOW    = 21            # ±21 días de tolerancia para "coincidir"
CONSENSUS_MODE      = "union_near"  # "intersection" | "union_near" | "pelt_only" | "binseg_only"
# intersection : solo quiebres donde AMBOS métodos coinciden (±ventana) → conservador
# union_near   : todos los quiebres de ambos métodos, fusionando los que están cerca → exploratorio
# pelt_only / binseg_only : usar un solo método (para comparar contra v2/v3 originales)

DARK_BG         = "#0d1117"
PANEL_BG        = "#161b22"
ACCENT_BLUE     = "#58a6ff"
ACCENT_GREEN    = "#3fb950"
ACCENT_RED      = "#f85149"
ACCENT_YELLOW   = "#e3b341"
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
        "^VIX3M": "vix3m",
        "^TNX":   "tnx",
        "^IRX":   "irx",
        "HYG":    "hyg",
        "IEI":    "iei",
    }

    raw = {}
    for ticker, name in tickers.items():
        try:
            df_t = yf.download(ticker, start=START_DATE, end=END_DATE,
                               auto_adjust=True, progress=False)
            # yfinance >= 0.2.x devuelve MultiIndex en columnas → aplanar
            if isinstance(df_t.columns, pd.MultiIndex):
                df_t.columns = df_t.columns.get_level_values(0)
            raw[name] = df_t["Close"]
            print(f"   ✓ {ticker:8s}  ({len(df_t)} obs)")
        except Exception as e:
            print(f"   ✗ {ticker:8s}  Error: {e}")
            raw[name] = None

    df = pd.DataFrame()
    df["close"]  = raw["spx"]
    df["ret"]    = np.log(df["close"]).diff()
    df["vix"]    = raw["vix"].shift(1) if raw["vix"] is not None else np.nan

    if raw["vix3m"] is not None and raw["vix"] is not None:
        df["vix_ts"] = (raw["vix3m"] / raw["vix"] - 1).shift(1)
    else:
        df["vix_ts"] = np.nan

    if raw["tnx"] is not None and raw["irx"] is not None:
        df["yield_curve"] = (raw["tnx"] - raw["irx"]).shift(1)
    else:
        df["yield_curve"] = np.nan

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
    NOTA: resid_lag1 NO va aquí — se construye dinámicamente dentro de cada
    régimen en estimate_regimes(), porque depende de los residuos del Ridge
    ajustado en ESE régimen específico. Ponerlo aquí como placeholder de 0.0
    (como en el borrador anterior) lo dejaba muerto para la detección de
    quiebres y confundía su origen.
    """
    df["lag1"]     = df["ret"].shift(1)
    df["lag2"]     = df["ret"].shift(2)
    df["lag5"]     = df["ret"].shift(5)

    df["vol20"]    = df["ret"].rolling(20).std()
    df["vol60"]    = df["ret"].rolling(60).std()
    df["vol_ratio"]= df["vol20"] / df["vol60"]

    df["abs_lag1"] = df["lag1"].abs()
    df["neg_lag1"] = (df["lag1"] < 0).astype(float) * df["lag1"]

    df["ma20"]     = df["close"] / df["close"].rolling(20).mean() - 1
    df["ma60"]     = df["close"] / df["close"].rolling(60).mean() - 1
    df["ma200"]    = df["close"] / df["close"].rolling(200).mean() - 1

    delta  = df["ret"].fillna(0)
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).shift(1)

    df["skew20"]   = df["ret"].rolling(20).skew().shift(1)
    df["kurt20"]   = df["ret"].rolling(20).kurt().shift(1)

    df["autocorr5"] = (
        df["ret"]
        .rolling(20)
        .apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
        .shift(1)
    )

    df["vix_ma20"]  = df["vix"] / df["vix"].rolling(20).mean() - 1
    df["vix_vol"]   = df["vix"] * df["vol20"]

    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = [
        "lag1", "lag2", "lag5",
        "vol20", "vol60", "vol_ratio",
        "abs_lag1", "neg_lag1",
        "ma20", "ma60", "ma200",
        "rsi14",
        "skew20", "kurt20",
        "autocorr5",
        "vix", "vix_ma20", "vix_vol", "vix_ts",
        "yield_curve", "credit_spread",
    ]

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
# 3. DETECCIÓN DE RUPTURAS — CONSENSO PELT + BINSEG
# ──────────────────────────────────────────────────────────────────────────────

def _merge_nearby(points: list[int], window: int) -> list[int]:
    """Fusiona puntos que están a menos de `window` días entre sí (promedio)."""
    if not points:
        return []
    points = sorted(points)
    clusters = [[points[0]]]
    for p in points[1:]:
        if p - clusters[-1][-1] <= window:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.mean(c)) for c in clusters]


def detect_breakpoints(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Detecta quiebres con DOS métodos sobre la MISMA señal multivariante
    [y | X_scaled] (no solo y, para no perder la info de VIX/yield curve/etc):

      - PELT(model="linear")    → penalización, encuentra su propio nº óptimo
      - BINSEG(model="linear")  → nº de quiebres fijado a BINSEG_N_BKPS

    Luego combina según CONSENSUS_MODE:
      - "intersection": solo quiebres donde ambos coinciden (±CONSENSUS_WINDOW)
      - "union_near":   todos los quiebres, fusionando los cercanos
      - "pelt_only" / "binseg_only": un solo método puro

    Devuelve un dict con bkps finales + diagnóstico de cada método individual,
    para que puedas ver qué tan de acuerdo estuvieron.
    """
    print("\n🔍  Detectando rupturas estructurales (PELT + BINSEG, señal multivariante)...")

    y = df["ret"].values
    scaler_bp = StandardScaler()
    X_bp = scaler_bp.fit_transform(df[feature_cols].values)
    signal = np.column_stack([y, X_bp])
    n = len(y)

    # ── PELT ──────────────────────────────────────────────────────────────
    algo_pelt = rpt.Pelt(model="linear", min_size=MIN_REGIME_DAYS, jump=JUMP).fit(signal)
    pen = PELT_PEN_MULT * np.log(n)
    bkps_pelt = algo_pelt.predict(pen=pen)

    # ── BINSEG ────────────────────────────────────────────────────────────
    algo_binseg = rpt.Binseg(model="linear", min_size=MIN_REGIME_DAYS, jump=JUMP).fit(signal)
    bkps_binseg = algo_binseg.predict(n_bkps=BINSEG_N_BKPS)

    print(f"\n   PELT    encontró {len(bkps_pelt) - 1} quiebres (penalización adaptativa)")
    for b in bkps_pelt[:-1]:
        print(f"      → {df.index[b].date()}")
    print(f"\n   BINSEG  encontró {len(bkps_binseg) - 1} quiebres (fijado a n={BINSEG_N_BKPS})")
    for b in bkps_binseg[:-1]:
        print(f"      → {df.index[b].date()}")

    interior_pelt   = bkps_pelt[:-1]
    interior_binseg = bkps_binseg[:-1]

    if CONSENSUS_MODE == "pelt_only":
        final_interior = interior_pelt
    elif CONSENSUS_MODE == "binseg_only":
        final_interior = interior_binseg
    elif CONSENSUS_MODE == "intersection":
        final_interior = []
        for p in interior_pelt:
            if any(abs(p - b) <= CONSENSUS_WINDOW for b in interior_binseg):
                final_interior.append(p)
    elif CONSENSUS_MODE == "union_near":
        final_interior = _merge_nearby(interior_pelt + interior_binseg, CONSENSUS_WINDOW)
    else:
        raise ValueError(f"CONSENSUS_MODE inválido: {CONSENSUS_MODE}")

    final_interior = sorted(set(final_interior))
    bkps_final = final_interior + [n]

    # Enforce min_size en el resultado final (el consenso puede violar la
    # restricción mínima si dos quiebres de métodos distintos quedan muy
    # juntos tras el merge)
    bkps_final = _enforce_min_size(bkps_final, n, MIN_REGIME_DAYS)

    print(f"\n   CONSENSO ({CONSENSUS_MODE}) → {len(bkps_final) - 1} quiebres finales:")
    for b in bkps_final[:-1]:
        print(f"      ★ {df.index[b].date()}")

    return {
        "final": bkps_final,
        "pelt": bkps_pelt,
        "binseg": bkps_binseg,
    }


def _enforce_min_size(bkps: list[int], n: int, min_size: int) -> list[int]:
    """Elimina quiebres consecutivos que dejarían un régimen < min_size."""
    if not bkps:
        return [n]
    cleaned = []
    prev = 0
    for b in bkps:
        if b - prev >= min_size:
            cleaned.append(b)
            prev = b
    if not cleaned or cleaned[-1] != n:
        cleaned.append(n)
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# 4. BOOTSTRAP SHARPE RATIO
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_sharpe(returns: np.ndarray, n: int = BOOTSTRAP_N,
                     alpha: float = 0.05) -> dict:
    sharpes = []
    for _ in range(n):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        std    = sample.std()
        if std > 0:
            sharpes.append(sample.mean() / std * np.sqrt(252))
    sharpes = np.array(sharpes)
    return {
        "mean":  sharpes.mean(),
        "lower": np.percentile(sharpes, alpha / 2 * 100),
        "upper": np.percentile(sharpes, (1 - alpha / 2) * 100),
    }


def _max_drawdown(returns: np.ndarray) -> float:
    cum  = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / peak
    return dd.min()


# ──────────────────────────────────────────────────────────────────────────────
# 5. ESTIMACIÓN POR REGÍMENES
# ──────────────────────────────────────────────────────────────────────────────

def estimate_regimes(df: pd.DataFrame, bkps: list[int],
                     feature_cols: list[str]) -> list[dict]:
    """
    Para cada régimen:
      1. Split IS/OOS (75/25)
      2. Escalado fit SOLO en IS (sin leakage)
      3. Ridge inicial → residuos IS → resid_lag1 (in-sample, válido)
      4. Ridge final con resid_lag1 como feature extra
      5. Predicción OOS walk-forward de 1 paso:
         NOTA IMPORTANTE: para predecir y_oos[t] usamos resid_prev = el
         residuo REAL del día t-1 (que sí conocemos al llegar a t, porque
         ya cerró el mercado). Esto es válido como estrategia "refit diario"
         pero NO es una predicción pura n-pasos-adelante sin retroalimentación.
         Si vas a usar esto para señales de trading, ten claro que asume que
         conoces el cierre de ayer antes de operar hoy (válido para EOD,
         no válido para intradía sin ese dato).
    """
    y_all = df["ret"].values
    resultados = []

    print("\n" + "="*80)
    print("  ESTIMACIÓN POR REGÍMENES — RIDGE + RESID_LAG1 (walk-forward OOS)")
    print("="*80)

    for i, b in enumerate(bkps):
        inicio = 0 if i == 0 else bkps[i - 1]
        fin = b

        yr  = y_all[inicio:fin]
        idx = df.index[inicio:fin]
        Xr  = df.iloc[inicio:fin][feature_cols].values

        n_obs = len(yr)
        if n_obs < 10:
            print(f"\n   ⚠️  Régimen {i+1} descartado: solo {n_obs} obs (< 10 mínimo)")
            continue

        split = max(10, int(n_obs * (1 - OOS_FRACTION)))
        split = min(split, n_obs - 1)

        X_is, X_oos = Xr[:split], Xr[split:]
        y_is, y_oos = yr[:split], yr[split:]

        # ── Escalado SOLO con IS ────────────────────────────────────────
        scaler = StandardScaler()
        X_is   = scaler.fit_transform(X_is)
        X_oos  = scaler.transform(X_oos) if len(X_oos) else X_oos

        n_splits_cv = min(5, max(2, split // 30))  # evita crash si split es chico

        # ── Ridge inicial (sin resid_lag1) ──────────────────────────────
        ridge0 = RidgeCV(alphas=RIDGE_ALPHAS, cv=TimeSeriesSplit(n_splits=n_splits_cv))
        ridge0.fit(X_is, y_is)
        yhat_is0  = ridge0.predict(X_is)
        resid_is0 = y_is - yhat_is0

        # ── resid_lag1 in-sample (válido: usa solo residuos pasados) ────
        resid_lag_is = np.concatenate(([0.0], resid_is0[:-1]))
        X_is_corr = np.column_stack([X_is, resid_lag_is])

        # ── Ridge final con resid_lag1 ───────────────────────────────────
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=TimeSeriesSplit(n_splits=n_splits_cv))
        ridge.fit(X_is_corr, y_is)
        yhat_is  = ridge.predict(X_is_corr)
        resid_is = y_is - yhat_is

        # ── Predicción OOS walk-forward de 1 paso ────────────────────────
        if len(y_oos):
            yhat_oos = np.zeros(len(y_oos))
            resid_prev = resid_is[-1]
            for t in range(len(y_oos)):
                x_t = np.append(X_oos[t], resid_prev).reshape(1, -1)
                yhat_oos[t] = ridge.predict(x_t)[0]
                resid_prev = y_oos[t] - yhat_oos[t]   # conocido al cerrar el día t
        else:
            yhat_oos = np.array([])

        r2_is  = r2_score(y_is, yhat_is)
        r2_oos = r2_score(y_oos, yhat_oos) if len(y_oos) > 1 else np.nan

        try:
            lag = min(10, len(resid_is) // 5)
            lb_pval = (acorr_ljungbox(resid_is, lags=lag, return_df=True)["lb_pvalue"].iloc[-1]
                      if lag >= 1 else np.nan)
        except Exception:
            lb_pval = np.nan

        sh = bootstrap_sharpe(yr)

        coefs = {col: ridge.coef_[j] for j, col in enumerate(feature_cols)}
        coefs["resid_lag1"] = ridge.coef_[-1]

        ann_ret = yr.mean() * 252
        ann_vol = yr.std()  * np.sqrt(252)
        max_dd  = _max_drawdown(yr)
        skew_r  = pd.Series(yr).skew()
        kurt_r  = pd.Series(yr).kurt()

        res = {
            "regimen": i + 1, "inicio": idx[0].date(), "fin": idx[-1].date(),
            "n_obs": n_obs, "n_is": split, "n_oos": n_obs - split,
            "alpha_opt": ridge.alpha_, "intercepto": ridge.intercept_,
            "r2_is": r2_is, "r2_oos": r2_oos, "lb_pval": lb_pval,
            "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe_mean": sh["mean"], "sharpe_lower": sh["lower"], "sharpe_upper": sh["upper"],
            "max_dd": max_dd, "skew": skew_r, "kurt": kurt_r,
            **coefs,
            "_idx": idx, "_yr": yr, "_yhat_is": yhat_is, "_yhat_oos": yhat_oos,
            "_resid": resid_is, "_split": split, "_ridge": ridge,
            "_feature_cols": feature_cols + ["resid_lag1"], "_scaler": scaler,
        }
        resultados.append(res)
        _print_regime(res)

    return resultados


def _print_regime(res: dict):
    print(f"\n{'─'*80}")
    print(f"  RÉGIMEN {res['regimen']:>2d}  │  {res['inicio']} → {res['fin']}  │  {res['n_obs']} obs")
    print(f"{'─'*80}")
    print(f"  Ridge α óptimo : {res['alpha_opt']:.4f}")
    print(f"  R² IS          : {res['r2_is']:.4f}   |   R² OOS: {res['r2_oos']:.4f}")
    lb_str = f"{res['lb_pval']:.4f}" if not np.isnan(res['lb_pval']) else "N/A"
    ok = not np.isnan(res['lb_pval']) and res['lb_pval'] > 0.05
    print(f"  Ljung-Box p    : {lb_str}  {'(residuos OK ✓)' if ok else '(autocorr detectada ⚠️)'}")
    print(f"  Retorno anual  : {res['ann_ret']*100:+.2f}%   Volatilidad: {res['ann_vol']*100:.2f}%")
    print(f"  Sharpe         : {res['sharpe_mean']:.3f}  IC 95%: [{res['sharpe_lower']:.3f}, {res['sharpe_upper']:.3f}]")
    print(f"  Max Drawdown   : {res['max_dd']*100:.2f}%   Skew: {res['skew']:.3f}   Kurt: {res['kurt']:.3f}")
    print(f"\n  Top coeficientes (|β| mayores):")
    feat_cols = res["_feature_cols"]
    coef_vals = sorted([(c, res[c]) for c in feat_cols], key=lambda x: abs(x[1]), reverse=True)
    for col, val in coef_vals[:6]:
        bar = "█" * int(min(abs(val) * 500, 20))
        sign = "+" if val >= 0 else "-"
        print(f"    {col:>16s}: {sign}{abs(val):.6f}  {bar}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. DASHBOARD DE VISUALIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, bkps_dict: dict, resultados: list[dict]):
    bkps_final = bkps_dict["final"]

    fig = plt.figure(figsize=(22, 20), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35,
                            left=0.07, right=0.97, top=0.95, bottom=0.04)

    def _style_ax(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for s in ["bottom", "left"]:
            ax.spines[s].set_color(GRID_COLOR)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.yaxis.label.set_color(TEXT_SECONDARY)
        ax.xaxis.label.set_color(TEXT_SECONDARY)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)
        if title:
            ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, fontweight="bold", pad=8)
        if xlabel: ax.set_xlabel(xlabel, fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, fontsize=8)

    # ── Panel 0: comparación PELT vs BINSEG vs CONSENSO ──────────────────
    ax0 = fig.add_subplot(gs[0, :])
    _style_ax(ax0, title=f"Detección de Quiebres: PELT vs BINSEG vs Consenso ({CONSENSUS_MODE})")
    ax0.plot(df.index, df["close"], color=TEXT_PRIMARY, lw=0.9, zorder=2)

    for b in bkps_dict["pelt"][:-1]:
        ax0.axvline(df.index[b], color=ACCENT_BLUE, lw=1.2, ls=":", alpha=0.8, zorder=3)
    for b in bkps_dict["binseg"][:-1]:
        ax0.axvline(df.index[b], color=ACCENT_YELLOW, lw=1.2, ls=":", alpha=0.8, zorder=3)
    for b in bkps_final[:-1]:
        ax0.axvline(df.index[b], color=ACCENT_RED, lw=1.6, ls="-", alpha=0.95, zorder=4)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=ACCENT_BLUE, lw=1.5, ls=":", label=f"PELT ({len(bkps_dict['pelt'])-1})"),
        Line2D([0], [0], color=ACCENT_YELLOW, lw=1.5, ls=":", label=f"BINSEG ({len(bkps_dict['binseg'])-1})"),
        Line2D([0], [0], color=ACCENT_RED, lw=1.8, label=f"Consenso ({len(bkps_final)-1})"),
    ]
    ax0.legend(handles=legend_elems, fontsize=8, facecolor=PANEL_BG,
              edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY, loc="upper left")
    ax0.set_ylabel("Precio (USD)", fontsize=8, color=TEXT_SECONDARY)

    # ── Panel 1: Precio + regímenes finales ───────────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    _style_ax(ax1, title="S&P 500 — Regímenes Finales (post-consenso)")
    ax1.plot(df.index, df["close"], color=TEXT_PRIMARY, lw=1.0, zorder=3)

    for i, res in enumerate(resultados):
        idx   = res["_idx"]
        color = REGIME_COLORS[i % len(REGIME_COLORS)]
        ax1.axvspan(idx[0], idx[-1], alpha=0.12, color=color, zorder=1)
        mid  = idx[len(idx) // 2]
        ypos = df["close"].max() * 0.98
        ax1.text(mid, ypos, f"R{res['regimen']}", color=color, fontsize=7,
                 ha="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc=DARK_BG, ec=color, lw=0.8))
    for b in bkps_final[:-1]:
        ax1.axvline(df.index[b], color=ACCENT_RED, lw=1.0, ls="--", alpha=0.9, zorder=4)
    ax1.set_ylabel("Precio (USD)", fontsize=8, color=TEXT_SECONDARY)

    # ── Panel 2: Retornos + VIX ─────────────────────────────────────────
    ax2  = fig.add_subplot(gs[2, :])
    ax2b = ax2.twinx()
    _style_ax(ax2, title="Retornos Diarios Log  +  VIX (eje derecho)")
    for i, res in enumerate(resultados):
        idx   = res["_idx"]
        color = REGIME_COLORS[i % len(REGIME_COLORS)]
        sub   = df.loc[idx[0]:idx[-1], "ret"]
        ax2.bar(sub.index, sub.values, color=color, alpha=0.4, width=1.0)
    if "vix" in df.columns:
        ax2b.plot(df.index, df["vix"], color=ACCENT_YELLOW, lw=0.8, alpha=0.7)
        ax2b.set_ylabel("VIX", color=ACCENT_YELLOW, fontsize=8)
        ax2b.tick_params(axis="y", colors=ACCENT_YELLOW, labelsize=7)
        ax2b.spines["right"].set_color(GRID_COLOR)
    ax2.set_ylabel("Retorno Log", fontsize=8, color=TEXT_SECONDARY)
    ax2.axhline(0, color=TEXT_SECONDARY, lw=0.5, alpha=0.5)

    # ── Panel 3: Heatmap de coeficientes ──────────────────────────────────
    ax3 = fig.add_subplot(gs[3, :])
    _style_ax(ax3, title="Coeficientes Ridge por Régimen (incluye resid_lag1)")
    feature_cols = resultados[0]["_feature_cols"]
    n_reg, n_feat = len(resultados), len(feature_cols)
    coef_matrix = np.zeros((n_feat, n_reg))
    for j, res in enumerate(resultados):
        for k, col in enumerate(feature_cols):
            coef_matrix[k, j] = res.get(col, 0.0)
    vmax = max(np.abs(coef_matrix).max(), 1e-8)
    im = ax3.imshow(coef_matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax3.set_yticks(range(n_feat))
    ax3.set_yticklabels(feature_cols, fontsize=7, color=TEXT_PRIMARY)
    ax3.set_xticks(range(n_reg))
    ax3.set_xticklabels([f"R{r['regimen']}\n{str(r['inicio'])[:7]}" for r in resultados],
                        fontsize=7, color=TEXT_PRIMARY)
    cbar = fig.colorbar(im, ax=ax3, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    for k in range(n_feat):
        for j in range(n_reg):
            val = coef_matrix[k, j]
            ax3.text(j, k, f"{val:.3f}", ha="center", va="center", fontsize=5.5,
                     color="white" if abs(val) > vmax * 0.5 else "black")

    # ── Panel 4a: Sharpe ──────────────────────────────────────────────────
    ax4a = fig.add_subplot(gs[4, 0])
    _style_ax(ax4a, title="Sharpe Ratio  +  IC 95% Bootstrap", ylabel="Sharpe")
    x_pos = range(n_reg)
    sh_means  = [r["sharpe_mean"] for r in resultados]
    sh_lowers = [r["sharpe_mean"] - r["sharpe_lower"] for r in resultados]
    sh_uppers = [r["sharpe_upper"] - r["sharpe_mean"] for r in resultados]
    bar_colors = [ACCENT_GREEN if s >= 0 else ACCENT_RED for s in sh_means]
    bars = ax4a.bar(x_pos, sh_means, color=bar_colors, alpha=0.75, width=0.6)
    ax4a.errorbar(x_pos, sh_means, yerr=[sh_lowers, sh_uppers],
                  fmt="none", color=TEXT_PRIMARY, capsize=4, lw=1.5)
    ax4a.axhline(0, color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax4a.set_xticks(x_pos)
    ax4a.set_xticklabels([f"R{r['regimen']}" for r in resultados], fontsize=8)
    for bar, val in zip(bars, sh_means):
        ax4a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*np.sign(val),
                  f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                  color=TEXT_PRIMARY, fontsize=7, fontweight="bold")

    # ── Panel 4b: R² OOS / Drawdown / Vol ─────────────────────────────────
    ax4b = fig.add_subplot(gs[4, 1])
    _style_ax(ax4b, title="R² OOS  |  Max Drawdown  |  Vol Anual")
    x, w = np.arange(n_reg), 0.25
    r2_oos = [r["r2_oos"] if not np.isnan(r["r2_oos"]) else 0 for r in resultados]
    mdd    = [abs(r["max_dd"]) for r in resultados]
    vols   = [r["ann_vol"] for r in resultados]
    ax4b.bar(x - w, r2_oos, width=w, color=ACCENT_BLUE,   alpha=0.8, label="R² OOS")
    ax4b.bar(x,     mdd,    width=w, color=ACCENT_RED,    alpha=0.8, label="|Max DD|")
    ax4b.bar(x + w, vols,   width=w, color=ACCENT_YELLOW, alpha=0.8, label="Vol Anual")
    ax4b.set_xticks(x)
    ax4b.set_xticklabels([f"R{r['regimen']}" for r in resultados], fontsize=8)
    ax4b.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_PRIMARY)

    fig.suptitle(
        "S&P 500 — Regímenes Estructurales  ·  Ridge L2 + Consenso PELT/BINSEG  ·  Walk-Forward OOS",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", y=0.985
    )

    plt.savefig("sp500_regimes_dashboard_v3.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.show()
    print("\n   Dashboard guardado: sp500_regimes_dashboard_v3.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. EXPORTACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def export_results(resultados: list[dict]):
    rows = [{k: v for k, v in res.items() if not k.startswith("_")} for res in resultados]
    tabla = pd.DataFrame(rows)
    tabla.to_csv("regimenes_sp500_ridge_v3.csv", index=False)

    print("\n" + "="*80)
    print("  TABLA RESUMEN DE REGÍMENES")
    print("="*80)
    cols_show = ["regimen", "inicio", "fin", "n_obs", "ann_ret", "ann_vol",
                 "sharpe_mean", "max_dd", "r2_is", "r2_oos", "alpha_opt", "lb_pval"]
    cols_show = [c for c in cols_show if c in tabla.columns]
    print(tabla[cols_show].round(4).to_string(index=False))
    print(f"\n   ✓ CSV guardado: regimenes_sp500_ridge_v3.csv")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print("  S&P 500 — STRUCTURAL REGIME ANALYSIS  v3.0  (PELT + BINSEG Consensus)")
    print("="*80 + "\n")

    df = download_data()
    df, feature_cols = build_features(df)
    bkps_dict = detect_breakpoints(df, feature_cols)
    resultados = estimate_regimes(df, bkps_dict["final"], feature_cols)

    print("\n📊  Generando dashboard...")
    plot_dashboard(df, bkps_dict, resultados)

    export_results(resultados)

    print("\n✅  Análisis completo.\n")


if __name__ == "__main__":
    main()