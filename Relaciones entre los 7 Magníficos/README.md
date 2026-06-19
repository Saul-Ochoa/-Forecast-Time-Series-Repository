# 🔬 Análisis de Relaciones entre los 7 Magníficos

## 📋 Descripción

Experimento sistemático que compara **8 métodos diferentes** para detectar relaciones entre acciones, usando los 7 Magníficos (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA) como caso de estudio.

## 🎯 Objetivos

1. Identificar qué método captura mejor diferentes tipos de relaciones
2. Encontrar pares de acciones con relaciones significativas
3. Descubrir patrones de lead-lag para estrategias de trading
4. Comparar DTW vs métodos tradicionales

## 📊 Métodos Implementados

| # | Método | Tipo de Relación | Aplicación |
|---|--------|-----------------|------------|
| 1 | **Pearson Correlation** | Lineal contemporánea | Diversificación |
| 2 | **Spearman Correlation** | Monótona | Relaciones no lineales |
| 3 | **Distance Correlation** | Cualquier dependencia | Detección general |
| 4 | **Cointegration** | Equilibrio largo plazo | Pairs trading |
| 5 | **Granger Causality** | Causalidad temporal | Lead-lag trading |
| 6 | **Cross-Correlation** | Correlación con rezago | Timing |
| 7 | **Mutual Information** | Dependencia general | Feature selection |
| 8 | **Dynamic Time Warping** | Forma trayectoria | Pattern matching |


## Resumen Final de los 8 Metodos

* Pearson/Spearman: META-MSFT parecen los más correlacionados (0.60)
* dCor + Gap: AMZN-META es el más lineal y estable
* Vol_corr: AMZN-META 0.80, comparten miedo
* Granger (precio): META → AMZN
* Cross-corr (vol): AMZN → META por 1 día
* Cointegración: ninguno cointegrado fuerte (todos p>0.1)
* MI: AMZN es el centro de información, AMZN-META 0.256