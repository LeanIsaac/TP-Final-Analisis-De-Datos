import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.float_format = '{:,.0f}'.format

# Configuración general
sns.set_style("whitegrid")
os.makedirs('graficos', exist_ok=True)

nombres_aglo = {32: 'CABA', 33: 'GBA'}
colores_aglo = {32: '#1f77b4', 33: '#ff7f0e'}  # azul / naranja, consistente en todos los gráficos


# =========================================================
# 1. CARGA Y UNIFICACIÓN DE LAS BASES EPH (2016-2025)
# =========================================================
ruta_carpeta = 'datos_eph/*.txt'
archivos = glob.glob(ruta_carpeta)

print(f"Se encontraron {len(archivos)} bases de la EPH para unir. Empezando la lectura... (puede tardar unos minutos)\n")

columnas_necesarias = [
    'ANO4', 'TRIMESTRE', 'AGLOMERADO', 'PONDERA', 'PONDIIO',
    'CH04', 'CH06', 'NIVEL_ED', 'ESTADO', 'CAT_OCUP',
    'PP04B_COD', 'PP04D_COD', 'P21', 'P47T', 'PP3E_TOT', 'PP07H', 'PP07A'
]

codigos_aglomerados = [32, 33]  # 32 = CABA, 33 = Partidos del GBA

lista_dataframes = []
for archivo in archivos:
    try:
        df_temporal = pd.read_csv(archivo, sep=';', usecols=lambda c: c in columnas_necesarias, low_memory=False)
        df_filtrado = df_temporal[df_temporal['AGLOMERADO'].isin(codigos_aglomerados)]
        lista_dataframes.append(df_filtrado)
    except Exception as e:
        print(f"Error leyendo el archivo {archivo}: {e}")

df_historico = pd.concat(lista_dataframes, ignore_index=True)
print(f"Base histórica (2016-2025) unificada: {df_historico.shape[0]:,} registros y {df_historico.shape[1]} columnas (CABA + GBA).\n")


# =========================================================
# 2. OBJETIVO 1.A - NO RESPUESTA AL INGRESO (P21) /////////
# =========================================================
print("=" * 60)
print("OBJETIVO 1.A - NO RESPUESTA A INGRESOS (P21)")
print("=" * 60)

# Nos quedamos con la población ocupada (ESTADO == 1)
df_ocupados = df_historico[df_historico['ESTADO'] == 1].copy()
df_ocupados['Sin_Respuesta'] = (df_ocupados['P21'] == -9).astype(int)

# Resumen global del período completo
total_ocupados = len(df_ocupados)
total_sin_respuesta = df_ocupados['Sin_Respuesta'].sum()
print(f"Total de ocupados (2016-2025, CABA+GBA): {total_ocupados:,}")
print(f"No declaran ingreso de la ocupación principal (-9): {total_sin_respuesta:,} "
      f"({total_sin_respuesta / total_ocupados * 100:.2f}%)\n")

# Evolución por año y aglomerado (para el gráfico)
no_respuesta_anual = df_ocupados.groupby(['ANO4', 'AGLOMERADO']).agg(
    Total_Ocupados=('Sin_Respuesta', 'size'),
    Sin_Respuesta=('Sin_Respuesta', 'sum')
).reset_index()
no_respuesta_anual['Tasa_NR (%)'] = (no_respuesta_anual['Sin_Respuesta'] / no_respuesta_anual['Total_Ocupados'] * 100).round(1)
no_respuesta_anual['Aglomerado'] = no_respuesta_anual['AGLOMERADO'].map(nombres_aglo)

print(no_respuesta_anual[['ANO4', 'Aglomerado', 'Total_Ocupados', 'Sin_Respuesta', 'Tasa_NR (%)']].to_string(index=False))

# --- Gráfico: evolución de la tasa de no respuesta ---
fig, ax = plt.subplots(figsize=(9, 5))
for aglo_cod, aglo_nombre in nombres_aglo.items():
    datos = no_respuesta_anual[no_respuesta_anual['AGLOMERADO'] == aglo_cod].sort_values('ANO4')
    ax.plot(datos['ANO4'], datos['Tasa_NR (%)'], marker='o', label=aglo_nombre, color=colores_aglo[aglo_cod])

ax.set_title('Evolución de la Tasa de No Respuesta al Ingreso (P21)\nPoblación Ocupada, CABA vs GBA (2016-2025)')
ax.set_xlabel('Año')
ax.set_ylabel('Tasa de No Respuesta (%)')
ax.legend(title='Aglomerado')
ax.set_xticks(range(2016, 2026))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/01_no_respuesta_evolucion.png', dpi=150)
plt.show()

# Reemplazamos los -9 por NaN para los cálculos numéricos posteriores
df_ocupados['P21'] = df_ocupados['P21'].replace(-9, np.nan)


# =========================================================
# 3. OBJETIVO 1.B - EXPLORACIÓN UNIVARIADA DEL INGRESO (P21)
#    Y DETECCIÓN DE OUTLIERS
# =========================================================
#
# NOTA METODOLÓGICA:
# P21 está en pesos CORRIENTES. Si calculamos los cuartiles/IQR
# mezclando 2016 a 2025, los valores nominales de los últimos años
# (varias veces más altos solo por inflación) van a inflar
# artificialmente el límite de outliers y distorsionar todo el análisis.
#
# Por eso, para esta exploración univariada usamos un CORTE TRANSVERSAL
# (el último trimestre disponible de la base). La comparación histórica
# !de ingresos en términos REALES (ajustados por IPC) la vamos a hacer
# !más adelante, en la sección de evolución de ingresos en el objetivo 2.

ultimo_ano = df_ocupados['ANO4'].max()
ultimo_trim = df_ocupados.loc[df_ocupados['ANO4'] == ultimo_ano, 'TRIMESTRE'].max()

print("\n" + "=" * 60)
print(f"OBJETIVO 1.B - DISTRIBUCIÓN DEL INGRESO (P21)")
print(f"Corte transversal: {ultimo_ano} - Trimestre {ultimo_trim}")
print("=" * 60)

df_reciente = df_ocupados[(df_ocupados['ANO4'] == ultimo_ano) & (df_ocupados['TRIMESTRE'] == ultimo_trim)]

estadisticos = {}
outliers_info = {}

for aglo_cod, aglo_nombre in nombres_aglo.items():
    ingresos = df_reciente[(df_reciente['AGLOMERADO'] == aglo_cod) & (df_reciente['P21'] > 0)]['P21'].dropna()

    Q1 = ingresos.quantile(0.25)
    Q2 = ingresos.quantile(0.50)
    Q3 = ingresos.quantile(0.75)
    IQR = Q3 - Q1
    limite_superior = Q3 + 1.5 * IQR
    outliers = ingresos[ingresos > limite_superior]

    estadisticos[aglo_cod] = {
        'n': len(ingresos),
        'media': ingresos.mean(),
        'mediana': Q2,
        'std': ingresos.std(),
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'min': ingresos.min(),
        'max': ingresos.max(),
    }
    outliers_info[aglo_cod] = {
        'limite_superior': limite_superior,
        'cantidad': len(outliers),
        'porcentaje': len(outliers) / len(ingresos) * 100,
    }

    print(f"\n{aglo_nombre} (n={len(ingresos)}):")
    print(f"  - Media:   ${ingresos.mean():>14,.0f}")
    print(f"  - Mediana: ${Q2:>14,.0f}")
    print(f"  - Q1 / Q3: ${Q1:>14,.0f}  /  ${Q3:>14,.0f}")
    print(f"  - IQR:     ${IQR:>14,.0f}")
    print(f"  - Límite atípico (Q3 + 1.5*IQR): ${limite_superior:,.0f}")
    print(f"  - Outliers detectados: {len(outliers)} casos ({outliers_info[aglo_cod]['porcentaje']:.2f}%)")

print("-" * 60)

# Tabla resumen (útil para pegar directo en el informe)
tabla_resumen = pd.DataFrame(estadisticos).T
tabla_resumen.index = tabla_resumen.index.map(nombres_aglo)
tabla_resumen = tabla_resumen.round(0)
print("\nTabla resumen - Estadísticos descriptivos de P21 (corte más reciente):")
print(tabla_resumen.to_string())


# --- Gráfico 1: Boxplot comparativo con umbral de outliers ---
fig, ax = plt.subplots(figsize=(8, 6))

data_to_plot = []
tick_labels_bp = []
for aglo_cod, aglo_nombre in nombres_aglo.items():
    ingresos = df_reciente[(df_reciente['AGLOMERADO'] == aglo_cod) & (df_reciente['P21'] > 0)]['P21'].dropna()
    data_to_plot.append(ingresos)
    tick_labels_bp.append(aglo_nombre)

bp = ax.boxplot(data_to_plot, tick_labels=tick_labels_bp, patch_artist=True, showfliers=False)

for patch, aglo_cod in zip(bp['boxes'], nombres_aglo.keys()):
    patch.set_facecolor(colores_aglo[aglo_cod])
    patch.set_alpha(0.6)

# Calculamos el Y máximo a mostrar: 1.3× el límite más alto (CABA = 3.8M → ~5M)
y_max = max(outliers_info[c]['limite_superior'] for c in nombres_aglo) * 1.3
ax.set_ylim(0, y_max)

# Outliers visibles: solo los que quedan dentro del rango graficado
for i, aglo_cod in enumerate(nombres_aglo.keys(), start=1):
    ingresos = df_reciente[(df_reciente['AGLOMERADO'] == aglo_cod) & (df_reciente['P21'] > 0)]['P21'].dropna()
    limite = outliers_info[aglo_cod]['limite_superior']
    outliers_visibles = ingresos[(ingresos > limite) & (ingresos <= y_max)]
    ax.scatter([i] * len(outliers_visibles), outliers_visibles,
               color=colores_aglo[aglo_cod], alpha=0.35, s=18, zorder=3)

# Anotaciones: mediana y línea de límite outlier
offsets = {32: (0.30, -0.30), 33: (0.30, -0.30)}  # ajuste horizontal del texto
for i, aglo_cod in enumerate(nombres_aglo.keys(), start=1):
    est = estadisticos[aglo_cod]
    out = outliers_info[aglo_cod]
    # Mediana
    ax.text(i + 0.28, est['mediana'],
            f"Mediana:\n${est['mediana']/1_000_000:.2f}M",
            fontsize=8, va='center', color=colores_aglo[aglo_cod])
    # Línea y etiqueta del límite outlier
    ax.axhline(out['limite_superior'], color=colores_aglo[aglo_cod],
               linestyle='--', alpha=0.7, linewidth=1.2)
    ax.text(i + 0.28, out['limite_superior'],
            f"Límite outlier:\n${out['limite_superior']/1_000_000:.2f}M",
            fontsize=8, va='bottom', color=colores_aglo[aglo_cod])

# Eje Y en millones de pesos
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1_000_000:.1f}M'))
ax.set_title(f"Análisis de Posición: Cuartiles, Mediana y Umbral de Valores Atípicos\nCABA vs GBA — {ultimo_ano} T{ultimo_trim}")
ax.set_ylabel("Ingreso de la Ocupación Principal ($ millones corrientes)")
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/02_boxplot_ingresos.png', dpi=150)
plt.show()

# --- Gráfico 2: Histograma + curva de densidad (KDE), sin outliers ---
fig, ax = plt.subplots(figsize=(9, 5))

for aglo_cod, aglo_nombre in nombres_aglo.items():
    ingresos = df_reciente[(df_reciente['AGLOMERADO'] == aglo_cod) & (df_reciente['P21'] > 0)]['P21'].dropna()
    limite_superior = outliers_info[aglo_cod]['limite_superior']
    ingresos_plot = ingresos[ingresos <= limite_superior]  # recorte visual para legibilidad

    sns.histplot(ingresos_plot, stat='density', kde=True, label=aglo_nombre,
                  color=colores_aglo[aglo_cod], alpha=0.4, ax=ax)
    ax.axvline(estadisticos[aglo_cod]['mediana'], color=colores_aglo[aglo_cod], linestyle='--',
               label=f"Mediana {aglo_nombre}: ${estadisticos[aglo_cod]['mediana']:,.0f}")

ax.set_title(f"Distribución del Ingreso de la Ocupación Principal (P21)\nCABA vs GBA - {ultimo_ano} T{ultimo_trim} (sin outliers)")
# ax.set_ylim(0, 8_000_000)
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1_000_000:.1f}M'))
ax.set_xlabel("Ingreso de la Ocupación Principal ($)")
ax.set_ylabel("Densidad")
ax.legend()
plt.tight_layout()
plt.savefig('graficos/03_histograma_ingresos.png', dpi=150)
plt.show()

print("\nGráficos guardados en la carpeta 'graficos/':")
print("  - 01_no_respuesta_evolucion.png")
print("  - 02_boxplot_ingresos.png")
print("  - 03_histograma_ingresos.png")


# =========================================================
# PARTE 1 - CARGA DEL IPC Y CONSTRUCCIÓN DEL DEFLACTOR
# =========================================================

import pandas as pd
import numpy as np

ipc_raw = pd.read_csv('ipc/ipc.csv', sep=';', encoding='latin1')

# Filtrar Nivel General GBA
ipc = ipc_raw[
    (ipc_raw['Codigo'].astype(str).str.strip() == '0') &
    (ipc_raw['Region'].str.strip() == 'GBA')
].copy()

# Convertir variación mensual a numérico
ipc['v_m_IPC'] = pd.to_numeric(
    ipc['v_m_IPC'].astype(str).str.replace(',', '.').str.strip(),
    errors='coerce'
)

# Ordenar por período
ipc['Periodo'] = ipc['Periodo'].astype(str)
ipc = ipc.sort_values('Periodo').reset_index(drop=True)

# Reconstruir índice acumulado desde base 100 (dic 2016)
ipc['indice_acum'] = 100.0
for i in range(1, len(ipc)):
    variacion = ipc.loc[i, 'v_m_IPC']
    if pd.isna(variacion):
        ipc.loc[i, 'indice_acum'] = ipc.loc[i-1, 'indice_acum']
    else:
        ipc.loc[i, 'indice_acum'] = ipc.loc[i-1, 'indice_acum'] * (1 + variacion / 100)

# Extraer año, mes y trimestre
ipc['anio'] = ipc['Periodo'].str[:4].astype(int)
ipc['mes']  = ipc['Periodo'].str[4:].astype(int)
ipc['trimestre'] = pd.cut(
    ipc['mes'],
    bins=[0, 3, 6, 9, 12],
    labels=[1, 2, 3, 4]
).astype(int)

# Promediar por trimestre
ipc_trim = ipc.groupby(['anio', 'trimestre'])['indice_acum'].mean().reset_index()
ipc_trim.columns = ['ANO4', 'TRIMESTRE', 'indice_ipc']

# Período base = T4 2025
indice_base = ipc_trim.loc[
    (ipc_trim['ANO4'] == 2025) & (ipc_trim['TRIMESTRE'] == 4),
    'indice_ipc'
].values[0]

print(f"Índice base (T4 2025): {indice_base:.2f}")

# Factor deflactor
ipc_trim['factor_deflactor'] = indice_base / ipc_trim['indice_ipc']

print("\nMuestra del deflactor:")
print(ipc_trim[ipc_trim['ANO4'] >= 2022].to_string(index=False))

# =========================================================
# PARTE 2 - DEFLACTAR P21 Y CALCULAR INGRESO REAL
# =========================================================

# Unir el deflactor con df_ocupados por año y trimestre
df_ocupados = df_ocupados.merge(
    ipc_trim[['ANO4', 'TRIMESTRE', 'factor_deflactor']],
    on=['ANO4', 'TRIMESTRE'],
    how='left'
)

# Calcular ingreso real (a pesos de T4 2025)
df_ocupados['P21'] = pd.to_numeric(df_ocupados['P21'], errors='coerce')
df_ocupados['P21_real'] = df_ocupados['P21'] * df_ocupados['factor_deflactor']

# Verificación
print("=" * 60)
print("PARTE 2 - DEFLACTADO DE P21")
print("=" * 60)
print(f"Registros con P21 real calculado: {df_ocupados['P21_real'].notna().sum():,}")
print(f"Registros sin factor (trimestres sin IPC): {df_ocupados['factor_deflactor'].isna().sum():,}")

# Muestra comparativa nominal vs real
muestra = df_ocupados[df_ocupados['P21'].notna()][['ANO4', 'TRIMESTRE', 'AGLOMERADO', 'P21', 'factor_deflactor', 'P21_real']].head(5)
print("\nMuestra comparativa nominal vs real:")
print(muestra.to_string(index=False))

# =========================================================
# PARTE 3 - EVOLUCIÓN HISTÓRICA DEL INGRESO REAL (P21)
# =========================================================
print("=" * 60)
print("PARTE 3 - EVOLUCIÓN HISTÓRICA DEL INGRESO REAL")
print("=" * 60)

# Solo ocupados con ingreso real válido y positivo
df_real = df_ocupados[
    (df_ocupados['P21_real'].notna()) &
    (df_ocupados['P21_real'] > 0)
].copy()

# Calcular media, mediana y percentiles ponderados por año y aglomerado
def estadisticos_ponderados(grupo):
    ingresos = grupo['P21_real'].values
    pesos    = grupo['PONDERA'].values
    orden    = np.argsort(ingresos)
    ingresos = ingresos[orden]
    pesos    = pesos[orden]
    pesos_acum = np.cumsum(pesos) / pesos.sum()
    media = np.average(ingresos, weights=pesos)
    p25  = ingresos[np.searchsorted(pesos_acum, 0.25)]
    p50  = ingresos[np.searchsorted(pesos_acum, 0.50)]
    p75  = ingresos[np.searchsorted(pesos_acum, 0.75)]
    p10  = ingresos[np.searchsorted(pesos_acum, 0.10)]
    p90  = ingresos[np.searchsorted(pesos_acum, 0.90)]
    return pd.Series({
        'media': media,
        'p10':   p10,
        'p25':   p25,
        'mediana': p50,
        'p75':   p75,
        'p90':   p90
    })

evolucion = df_real.groupby(['ANO4', 'AGLOMERADO']).apply(
    estadisticos_ponderados, include_groups=False
).reset_index()

evolucion['Aglomerado'] = evolucion['AGLOMERADO'].map(nombres_aglo)

print("\nEvolución del ingreso real (pesos T4 2025):")
print(evolucion[['ANO4', 'Aglomerado', 'media', 'mediana', 'p25', 'p75']].to_string(index=False))

# --- Gráfico: evolución de media y mediana ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

metricas = [('media', 'Media'), ('mediana', 'Mediana')]

for ax, (col, label) in zip(axes, metricas):
    for aglo_cod, aglo_nombre in nombres_aglo.items():
        datos = evolucion[evolucion['AGLOMERADO'] == aglo_cod].sort_values('ANO4')
        ax.plot(datos['ANO4'], datos[col] / 1_000_000,
                marker='o', label=aglo_nombre, color=colores_aglo[aglo_cod])
    ax.set_title(f'{label} del Ingreso Real\nCABA vs GBA (2016-2025)')
    ax.set_xlabel('Año')
    ax.set_ylabel('Ingreso real (millones $ T4 2025)')
    ax.legend(title='Aglomerado')
    ax.set_xticks(range(2016, 2026))
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/04_evolucion_ingreso_real.png', dpi=150)
plt.show()

# --- Gráfico: abanico de percentiles ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    datos = evolucion[evolucion['AGLOMERADO'] == aglo_cod].sort_values('ANO4')
    años  = datos['ANO4']
    color = colores_aglo[aglo_cod]
    ax.fill_between(años, datos['p10']/1e6, datos['p90']/1e6,
                    alpha=0.15, color=color, label='P10-P90')
    ax.fill_between(años, datos['p25']/1e6, datos['p75']/1e6,
                    alpha=0.30, color=color, label='P25-P75')
    ax.plot(años, datos['mediana']/1e6, color=color,
            marker='o', linewidth=2, label='Mediana')
    ax.set_title(f'Distribución del Ingreso Real — {aglo_nombre}\n(2016-2025, pesos T4 2025)')
    ax.set_xlabel('Año')
    ax.set_ylabel('Ingreso real (millones $ T4 2025)')
    ax.set_xticks(range(2016, 2026))
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/05_abanico_percentiles.png', dpi=150)
plt.show()

print("\nGráficos guardados:")
print("  - 04_evolucion_ingreso_real.png")
print("  - 05_abanico_percentiles.png")

# =========================================================
# PARTE 4 - ANÁLISIS MULTIVARIADO
# =========================================================

# --- 4.A Brecha de ingresos por SEXO ---
print("=" * 60)
print("PARTE 4.A - BRECHA DE INGRESOS POR SEXO (CH04)")
print("=" * 60)

df_real['CH04'] = pd.to_numeric(df_real['CH04'], errors='coerce')
df_sexo = df_real[df_real['CH04'].isin([1, 2])].copy()
df_sexo['Sexo'] = df_sexo['CH04'].map({1: 'Varón', 2: 'Mujer'})

evol_sexo = df_sexo.groupby(['ANO4', 'AGLOMERADO', 'Sexo']).apply(
    estadisticos_ponderados, include_groups=False
).reset_index()

print(evol_sexo[['ANO4', 'AGLOMERADO', 'Sexo', 'mediana']].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
estilos = {'Varón': '-', 'Mujer': '--'}

for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    for sexo in ['Varón', 'Mujer']:
        datos = evol_sexo[
            (evol_sexo['AGLOMERADO'] == aglo_cod) &
            (evol_sexo['Sexo'] == sexo)
        ].sort_values('ANO4')
        ax.plot(datos['ANO4'], datos['mediana'] / 1e6,
                marker='o', linestyle=estilos[sexo],
                color=colores_aglo[aglo_cod],
                alpha=0.6 if sexo == 'Mujer' else 1.0,
                label=sexo)
    ax.set_title(f'Mediana Ingreso Real por Sexo — {aglo_nombre}\n(2016-2025)')
    ax.set_xlabel('Año')
    ax.set_ylabel('Ingreso real (millones $ T4 2025)')
    ax.set_xticks(range(2016, 2026))
    ax.legend(title='Sexo')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/06_brecha_sexo.png', dpi=150)
plt.show()

# --- 4.B Ingreso por NIVEL EDUCATIVO ---
print("=" * 60)
print("PARTE 4.B - INGRESO REAL POR NIVEL EDUCATIVO (NIVEL_ED)")
print("=" * 60)

etiquetas_ed = {
    0: 'Sin instrucción',
    1: 'Primaria inc.',
    2: 'Primaria comp.',
    3: 'Secundaria inc.',
    4: 'Secundaria comp.',
    5: 'Superior inc.',
    6: 'Superior comp.',
    7: 'Posgrado'
}

df_real['NIVEL_ED'] = pd.to_numeric(df_real['NIVEL_ED'], errors='coerce')
df_ed = df_real[df_real['NIVEL_ED'].notna()].copy()
df_ed['Nivel'] = df_ed['NIVEL_ED'].map(etiquetas_ed)

evol_ed = df_ed.groupby(['ANO4', 'AGLOMERADO', 'NIVEL_ED']).apply(
    estadisticos_ponderados, include_groups=False
).reset_index()
evol_ed['Nivel'] = evol_ed['NIVEL_ED'].map(etiquetas_ed)

niveles_clave = [2, 4, 6]
colores_ed = {2: '#2ca02c', 4: '#ff7f0e', 6: '#d62728'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    for nivel in niveles_clave:
        datos = evol_ed[
            (evol_ed['AGLOMERADO'] == aglo_cod) &
            (evol_ed['NIVEL_ED'] == nivel)
        ].sort_values('ANO4')
        ax.plot(datos['ANO4'], datos['mediana'] / 1e6,
                marker='o', color=colores_ed[nivel],
                label=etiquetas_ed[nivel])
    ax.set_title(f'Mediana Ingreso Real por Nivel Educativo — {aglo_nombre}')
    ax.set_xlabel('Año')
    ax.set_ylabel('Ingreso real (millones $ T4 2025)')
    ax.set_xticks(range(2016, 2026))
    ax.legend(title='Nivel educativo', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/07_ingreso_nivel_educativo.png', dpi=150)
plt.show()

# --- 4.C Ingreso por GRUPO DE EDAD ---
print("=" * 60)
print("PARTE 4.C - INGRESO REAL POR GRUPO DE EDAD (CH06)")
print("=" * 60)

df_real['CH06'] = pd.to_numeric(df_real['CH06'], errors='coerce')
bins_edad   = [14, 24, 34, 44, 54, 64, 99]
labels_edad = ['15-24', '25-34', '35-44', '45-54', '55-64', '65+']
df_real['GrupoEdad'] = pd.cut(df_real['CH06'], bins=bins_edad, labels=labels_edad)

evol_edad = df_real[df_real['GrupoEdad'].notna()].groupby(
    ['ANO4', 'AGLOMERADO', 'GrupoEdad']
).apply(estadisticos_ponderados, include_groups=False).reset_index()

colores_edad = {
    '15-24': '#1f77b4', '25-34': '#ff7f0e', '35-44': '#2ca02c',
    '45-54': '#d62728', '55-64': '#9467bd', '65+':   '#8c564b'
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    for grupo in labels_edad:
        datos = evol_edad[
            (evol_edad['AGLOMERADO'] == aglo_cod) &
            (evol_edad['GrupoEdad'] == grupo)
        ].sort_values('ANO4')
        if len(datos) > 0:
            ax.plot(datos['ANO4'], datos['mediana'] / 1e6,
                    marker='o', color=colores_edad[grupo], label=grupo)
    ax.set_title(f'Mediana Ingreso Real por Grupo de Edad — {aglo_nombre}')
    ax.set_xlabel('Año')
    ax.set_ylabel('Ingreso real (millones $ T4 2025)')
    ax.set_xticks(range(2016, 2026))
    ax.legend(title='Grupo de edad', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/08_ingreso_grupo_edad.png', dpi=150)
plt.show()

print("\nGráficos guardados:")
print("  - 06_brecha_sexo.png")
print("  - 07_ingreso_nivel_educativo.png")
print("  - 08_ingreso_grupo_edad.png")

# =========================================================
# PARTE 5 - TASA DE DESOCUPACIÓN CRUZADA
# =========================================================
print("=" * 60)
print("PARTE 5 - TASA DE DESOCUPACIÓN POR VARIABLES SOCIODEMOGRÁFICAS")
print("=" * 60)

# Base: población activa (ESTADO 1=ocupado, 2=desocupado)
df_activos = df_historico[df_historico['ESTADO'].isin([1, 2])].copy()
df_activos['ESTADO'] = pd.to_numeric(df_activos['ESTADO'], errors='coerce')
df_activos['PONDERA'] = pd.to_numeric(df_activos['PONDERA'], errors='coerce')
df_activos['Desocupado'] = (df_activos['ESTADO'] == 2).astype(int)

def tasa_desoc_ponderada(grupo):
    desoc = (grupo['Desocupado'] * grupo['PONDERA']).sum()
    total = grupo['PONDERA'].sum()
    return pd.Series({'tasa_desoc': desoc / total * 100})

# --- 5.A Por SEXO ---
print("\n5.A - Tasa de desocupación por sexo")
df_activos['CH04'] = pd.to_numeric(df_activos['CH04'], errors='coerce')
df_activos['Sexo'] = df_activos['CH04'].map({1: 'Varón', 2: 'Mujer'})

desoc_sexo = df_activos[df_activos['Sexo'].notna()].groupby(
    ['ANO4', 'AGLOMERADO', 'Sexo']
).apply(tasa_desoc_ponderada, include_groups=False).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    for sexo in ['Varón', 'Mujer']:
        datos = desoc_sexo[
            (desoc_sexo['AGLOMERADO'] == aglo_cod) &
            (desoc_sexo['Sexo'] == sexo)
        ].sort_values('ANO4')
        ax.plot(datos['ANO4'], datos['tasa_desoc'],
                marker='o',
                linestyle='-' if sexo == 'Varón' else '--',
                color=colores_aglo[aglo_cod],
                alpha=1.0 if sexo == 'Varón' else 0.6,
                label=sexo)
    ax.set_title(f'Tasa de Desocupación por Sexo — {aglo_nombre}')
    ax.set_xlabel('Año')
    ax.set_ylabel('Tasa de desocupación (%)')
    ax.set_xticks(range(2016, 2026))
    ax.legend(title='Sexo')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/09_desoc_sexo.png', dpi=150)
plt.show()

# --- 5.B Por NIVEL EDUCATIVO ---
print("\n5.B - Tasa de desocupación por nivel educativo")
df_activos['NIVEL_ED'] = pd.to_numeric(df_activos['NIVEL_ED'], errors='coerce')
df_activos['Nivel'] = df_activos['NIVEL_ED'].map(etiquetas_ed)

desoc_ed = df_activos[df_activos['NIVEL_ED'].isin(niveles_clave)].groupby(
    ['ANO4', 'AGLOMERADO', 'NIVEL_ED']
).apply(tasa_desoc_ponderada, include_groups=False).reset_index()
desoc_ed['Nivel'] = desoc_ed['NIVEL_ED'].map(etiquetas_ed)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    for nivel in niveles_clave:
        datos = desoc_ed[
            (desoc_ed['AGLOMERADO'] == aglo_cod) &
            (desoc_ed['NIVEL_ED'] == nivel)
        ].sort_values('ANO4')
        ax.plot(datos['ANO4'], datos['tasa_desoc'],
                marker='o', color=colores_ed[nivel],
                label=etiquetas_ed[nivel])
    ax.set_title(f'Tasa de Desocupación por Nivel Educativo — {aglo_nombre}')
    ax.set_xlabel('Año')
    ax.set_ylabel('Tasa de desocupación (%)')
    ax.set_xticks(range(2016, 2026))
    ax.legend(title='Nivel educativo', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/10_desoc_nivel_educativo.png', dpi=150)
plt.show()

# --- 5.C Por GRUPO DE EDAD ---
print("\n5.C - Tasa de desocupación por grupo de edad")
df_activos['CH06'] = pd.to_numeric(df_activos['CH06'], errors='coerce')
df_activos['GrupoEdad'] = pd.cut(
    df_activos['CH06'],
    bins=bins_edad,
    labels=labels_edad
)

desoc_edad = df_activos[df_activos['GrupoEdad'].notna()].groupby(
    ['ANO4', 'AGLOMERADO', 'GrupoEdad']
).apply(tasa_desoc_ponderada, include_groups=False).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (aglo_cod, aglo_nombre) in zip(axes, nombres_aglo.items()):
    for grupo in labels_edad:
        datos = desoc_edad[
            (desoc_edad['AGLOMERADO'] == aglo_cod) &
            (desoc_edad['GrupoEdad'] == grupo)
        ].sort_values('ANO4')
        if len(datos) > 0:
            ax.plot(datos['ANO4'], datos['tasa_desoc'],
                    marker='o', color=colores_edad[grupo],
                    label=grupo)
    ax.set_title(f'Tasa de Desocupación por Grupo de Edad — {aglo_nombre}')
    ax.set_xlabel('Año')
    ax.set_ylabel('Tasa de desocupación (%)')
    ax.set_xticks(range(2016, 2026))
    ax.legend(title='Grupo de edad', fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/11_desoc_grupo_edad.png', dpi=150)
plt.show()

print("\nGráficos guardados:")
print("  - 09_desoc_sexo.png")
print("  - 10_desoc_nivel_educativo.png")
print("  - 11_desoc_grupo_edad.png")
