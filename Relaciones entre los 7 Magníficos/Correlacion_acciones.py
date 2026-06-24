"""
correlacion_acciones.py
=======================
Análisis de 8 métodos de correlación para cualquier conjunto de acciones.

Uso rápido:
    python correlacion_acciones.py                  # 7 Magníficos por defecto
    python correlacion_acciones.py AAPL MSFT NVDA   # tickers custom

Importado como módulo:
    from correlacion_acciones import CorrelacionAcciones
    ca = CorrelacionAcciones(['AAPL','MSFT','NVDA'], start='2024-01-01')
    ca.calcular_todo()
    ca.graficar()
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import dcor

from itertools import combinations
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import coint, grangercausalitytests

# dtw-python expone la clase dtw; la función de bajo nivel es mucho más rápida
try:
    from dtaidistance import dtw as dtw_fast
    _DTW_BACKEND = "dtaidistance"
except ImportError:
    try:
        from dtw import dtw as dtw_obj
        _DTW_BACKEND = "dtw-python"
    except ImportError:
        _DTW_BACKEND = None

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _matriz_vacia(tickers: list[str], fill=np.nan) -> pd.DataFrame:
    """DataFrame cuadrado con diagonal en 1.0 si fill es np.nan."""
    n = len(tickers)
    arr = np.full((n, n), fill, dtype=float)   # array propio, nunca read-only
    if fill is np.nan:
        np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=tickers, columns=tickers)


def _heatmap(df: pd.DataFrame, titulo: str, ax: plt.Axes,
             fmt: str = ".2f", cmap: str = "coolwarm") -> None:
    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap,
                square=True, ax=ax, cbar=False,
                annot_kws={"size": 7})
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)


# ──────────────────────────────────────────────
# Clase principal
# ──────────────────────────────────────────────

class CorrelacionAcciones:
    """
    Calcula 8 métodos de correlación y genera heatmaps + ranking CSV.

    Parámetros
    ----------
    tickers : list[str]
        Símbolos de Yahoo Finance.
    start : str
        Fecha de inicio, formato 'YYYY-MM-DD'.
    end : str | None
        Fecha de fin (None = hoy).
    granger_maxlag : int
        Lags máximos para la prueba de Granger.
    vol_ventana : int
        Ventana en días de trading para la volatilidad rolling.
    """

    def __init__(
        self,
        tickers: list[str],
        start: str = "2024-06-01",
        end: str | None = None,
        granger_maxlag: int = 5,
        vol_ventana: int = 21,
    ):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.granger_maxlag = granger_maxlag
        self.vol_ventana = vol_ventana

        # Se llenan al llamar calcular_todo()
        self.precios: pd.DataFrame | None = None
        self.retornos: pd.DataFrame | None = None
        self.vol: pd.DataFrame | None = None
        self.resultados: dict[str, pd.DataFrame] = {}

    # ── Datos ────────────────────────────────

    def descargar_datos(self) -> None:
        log.info("Descargando precios para: %s", self.tickers)
        kwargs = dict(start=self.start, auto_adjust=True)
        if self.end:
            kwargs["end"] = self.end

        raw = yf.download(self.tickers, **kwargs, progress=False)

        # yfinance devuelve MultiIndex cuando hay varios tickers
        if isinstance(raw.columns, pd.MultiIndex):
            self.precios = raw["Close"].dropna()
        else:
            self.precios = raw[["Close"]].dropna()
            self.precios.columns = self.tickers

        tickers_faltantes = [t for t in self.tickers if t not in self.precios.columns]
        if tickers_faltantes:
            log.warning("Sin datos para: %s", tickers_faltantes)

        self.tickers = list(self.precios.columns)           # actualizar lista real
        self.retornos = np.log(self.precios / self.precios.shift(1)).dropna()
        self.vol = self.retornos.rolling(self.vol_ventana).std() * np.sqrt(252)
        log.info("%d filas · %d tickers cargados", len(self.precios), len(self.tickers))

    # ── Métodos de correlación ────────────────

    def _pearson(self) -> pd.DataFrame:
        return self.retornos.corr()

    def _spearman(self) -> pd.DataFrame:
        return self.retornos.corr("spearman")

    def _dcor(self) -> pd.DataFrame:
        """Distance correlation (detecta relaciones no lineales)."""
        tickers = self.tickers
        mat = _matriz_vacia(tickers, fill=np.nan)   # diagonal = 1.0 automático
        for a, b in combinations(tickers, 2):
            val = dcor.distance_correlation(
                self.retornos[a].values, self.retornos[b].values
            )
            mat.loc[a, b] = mat.loc[b, a] = val
        return mat

    def _vol_corr(self) -> pd.DataFrame:
        return self.vol.corr()

    def _cointegracion(self) -> pd.DataFrame:
        """P-valores de Engle-Granger. Valores bajos → cointegración."""
        tickers = self.tickers
        # fill=1.0 para off-diagonal; diagonal quedará en 0.0 (p trivial)
        mat = _matriz_vacia(tickers, fill=1.0)
        for i, t in enumerate(tickers):             # diagonal = 0 manualmente
            mat.iloc[i, i] = 0.0
        for a, b in combinations(tickers, 2):
            _, p, _ = coint(self.precios[a], self.precios[b])
            mat.loc[a, b] = mat.loc[b, a] = p
        return mat

    def _granger(self) -> pd.DataFrame:
        """P-valor mínimo entre lags 1..maxlag. mat[causa, efecto]."""
        tickers = self.tickers
        mat = _matriz_vacia(tickers, fill=np.nan)   # diagonal = 1.0, pero NaN es más correcto
        for i, t in enumerate(tickers):             # diagonal explícita = NaN
            mat.iloc[i, i] = np.nan
        for causa in tickers:
            for efecto in tickers:
                if causa == efecto:
                    continue
                try:
                    tests = grangercausalitytests(
                        self.retornos[[efecto, causa]].dropna(),
                        maxlag=self.granger_maxlag,
                        verbose=False,
                    )
                    p_min = min(
                        tests[lag][0]["ssr_ftest"][1]
                        for lag in range(1, self.granger_maxlag + 1)
                    )
                    mat.loc[causa, efecto] = p_min
                except Exception as exc:
                    log.debug("Granger %s→%s falló: %s", causa, efecto, exc)
        return mat

    def _mutual_info(self) -> pd.DataFrame:
        tickers = self.tickers
        mat = _matriz_vacia(tickers, fill=np.nan)   # diagonal = 1.0 automático
        for a, b in combinations(tickers, 2):
            val = mutual_info_regression(
                self.retornos[a].values.reshape(-1, 1),
                self.retornos[b].values,
                random_state=42,
            )[0]
            mat.loc[a, b] = mat.loc[b, a] = val
        return mat

    def _dtw(self) -> pd.DataFrame:
        """DTW sobre precios normalizados min-max."""
        tickers = self.tickers
        norm = self.precios.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        mat = _matriz_vacia(tickers, fill=0.0)

        for a, b in combinations(tickers, 2):
            sa = norm[a].values.astype(np.double)
            sb = norm[b].values.astype(np.double)

            if _DTW_BACKEND == "dtaidistance":
                d = dtw_fast.distance(sa, sb)
            elif _DTW_BACKEND == "dtw-python":
                d = dtw_obj(sa, sb).distance
            else:
                # Fallback: correlación cruzada como proxy (no DTW real)
                log.warning("dtaidistance/dtw-python no disponible; usando proxy de correlación.")
                d = 1 - abs(np.corrcoef(sa, sb)[0, 1])

            mat.loc[a, b] = mat.loc[b, a] = round(d, 2)

        return mat

    # ── Cálculo completo ──────────────────────

    def calcular_todo(self) -> None:
        """Descarga datos y ejecuta los 8 métodos."""
        if self.precios is None:
            self.descargar_datos()

        pasos = [
            ("pearson",      "1. Pearson",          self._pearson),
            ("spearman",     "2. Spearman",          self._spearman),
            ("dcor",         "3. dCor",              self._dcor),
            ("vol_corr",     "4. Vol Corr",          self._vol_corr),
            ("cointegracion","5. Cointeg. p-val",    self._cointegracion),
            ("granger",      "6. Granger p-val",     self._granger),
            ("mi",           "7. Mutual Info",       self._mutual_info),
            ("dtw",          "8. DTW Distance",      self._dtw),
        ]

        for clave, label, func in pasos:
            log.info("Calculando %s...", label)
            self.resultados[clave] = func()

        log.info("Cálculo completado.")

    # ── Visualización ─────────────────────────

    def graficar(self, guardar_como: str = "correlaciones.png") -> None:
        if not self.resultados:
            raise RuntimeError("Llamá primero a calcular_todo().")

        titulo = f"8 Métodos de Correlación — {', '.join(self.tickers)}"

        config_heatmaps = [
            ("pearson",       ".2f",  "coolwarm"),
            ("spearman",      ".2f",  "coolwarm"),
            ("dcor",          ".2f",  "viridis"),
            ("vol_corr",      ".2f",  "magma"),
            ("cointegracion", ".3f",  "RdYlGn_r"),
            ("granger",       ".3f",  "RdYlGn_r"),
            ("mi",            ".3f",  "YlOrRd"),
            ("dtw",           ".2f",  "viridis_r"),
        ]

        labels = [
            "1. Pearson", "2. Spearman", "3. dCor",
            "4. Vol Corr", "5. Cointeg. p", "6. Granger p",
            "7. Mutual Info", "8. DTW Dist",
        ]

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()

        for idx, ((clave, fmt, cmap), label) in enumerate(zip(config_heatmaps, labels)):
            _heatmap(self.resultados[clave], label, axes[idx], fmt=fmt, cmap=cmap)

        axes[8].axis("off")                          # celda libre
        plt.suptitle(titulo, fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(guardar_como, dpi=150, bbox_inches="tight")
        log.info("Gráfico guardado → %s", guardar_como)
        plt.show()

    # ── Ranking ───────────────────────────────

    def ranking(self, guardar_csv: str = "ranking_correlaciones.csv") -> pd.DataFrame:
        """
        Tabla de pares ordenada por MI y VolCorr descendente.
        Incluye columna 'Granger_p' tomando el mínimo de ambas direcciones.
        """
        if not self.resultados:
            raise RuntimeError("Llamá primero a calcular_todo().")

        r = self.resultados
        filas = []

        for a, b in combinations(self.tickers, 2):
            gp_ab = r["granger"].loc[a, b]
            gp_ba = r["granger"].loc[b, a]
            granger_min = (
                round(min(v for v in [gp_ab, gp_ba] if pd.notna(v)), 4)
                if any(pd.notna(v) for v in [gp_ab, gp_ba])
                else None
            )
            filas.append({
                "Par":       f"{a}-{b}",
                "Pearson":   round(r["pearson"].loc[a, b],   3),
                "Spearman":  round(r["spearman"].loc[a, b],  3),
                "dCor":      round(r["dcor"].loc[a, b],      3),
                "MI":        round(r["mi"].loc[a, b],        3),
                "VolCorr":   round(r["vol_corr"].loc[a, b],  3),
                "DTW":       r["dtw"].loc[a, b],
                "Coint_p":   round(r["cointegracion"].loc[a, b], 4),
                "Granger_p": granger_min,
            })

        df = (
            pd.DataFrame(filas)
            .sort_values(["MI", "VolCorr"], ascending=False)
            .reset_index(drop=True)
        )

        df.to_csv(guardar_csv, index=False)
        log.info("Ranking guardado → %s", guardar_csv)
        return df


# ──────────────────────────────────────────────
# Ejecución directa
# ──────────────────────────────────────────────

if __name__ == "__main__":
    TICKERS_DEFAULT = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS_DEFAULT

    ca = CorrelacionAcciones(tickers, start="2024-01-01")
    ca.calcular_todo()
    ca.graficar("8_metodos_magnificos.png")

    tabla = ca.ranking("ranking_8_metodos.csv")
    print("\nTop 5 pares más alineados:")
    print(tabla.head(5).to_string(index=False))