import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.title("📊 Control Automático de PM10 y PM2.5 - Nivel 1")

st.markdown("""
Esta aplicación procesa archivos de datos crudos (`.dat` o `.csv`) para evaluar:
- Rangos de funcionamiento (RF)
- Consistencia temporal (CT)
- Consistencia interna (CI)
y determinar el **estado final** de cada contaminante (`PM25`, `PM10`).
""")

# ======================================================
# CARGA DE ARCHIVO
# ======================================================
uploaded_file = st.file_uploader("📂 Sube el archivo de datos (formato .dat o .csv)", type=["dat", "csv"])

if uploaded_file is not None:
    # ======================================================
    # CARGA Y PREPROCESAMIENTO
    # ======================================================
    df = pd.read_csv(uploaded_file, delimiter=",", skiprows=[0, 2, 3])
    dff = df.copy()

    columnas_object = dff.select_dtypes(include=['object']).columns
    columnas_a_convertir = columnas_object.drop('TIMESTAMP', errors='ignore')
    dff[columnas_a_convertir] = dff[columnas_a_convertir].apply(pd.to_numeric, errors='coerce')
    dff['TIMESTAMP'] = pd.to_datetime(dff['TIMESTAMP'])
    dff = dff.set_index('TIMESTAMP')
    
    st.sidebar.subheader("🏭 Información y filtros")
    estacion = st.sidebar.text_input("Nombre de la estación:", value="Estacion_X")

    # ======================================================
    # RANGO DE FECHAS SELECCIONABLE
    # ======================================================
    min_fecha, max_fecha = dff.index.min(), dff.index.max()
    st.sidebar.subheader("🕒 Seleccionar rango de fechas a procesar")
    inicio = st.sidebar.date_input("Fecha inicio", min_fecha.date())
    fin = st.sidebar.date_input("Fecha fin", max_fecha.date())

    if inicio > fin:
        st.error("⚠️ La fecha de inicio no puede ser posterior a la fecha de fin.")
    else:
        inicio = pd.Timestamp(inicio)
        fin = pd.Timestamp(fin) + pd.Timedelta(hours=23)

        # Filtrar y limpiar duplicados
        df_filtrado = dff[(dff.index >= inicio) & (dff.index <= fin)]
        df_filtrado = df_filtrado[~df_filtrado.index.duplicated(keep='first')]

        # Rellenar fechas faltantes
        rango_fechas = pd.date_range(start=df_filtrado.index.min(), end=df_filtrado.index.max(), freq='H')
        df_completo = pd.DataFrame(index=rango_fechas)
        df_completo.index.name = 'date'
        dff_completo = df_completo.join(df_filtrado, how='left')

        # ======================================================
        # EXTRACCIÓN DE COLUMNAS PM25 y PM10
        # ======================================================
        prefijo = ("PM25", "PM10")
        columnas_prefijo = [col for col in dff_completo.columns if col.startswith(prefijo)]
        df_nuevo = dff_completo[columnas_prefijo].copy()

        # ======================================================
        # RANGOS DE FUNCIONAMIENTO
        # ======================================================
        limites = {
            'PM25_CONC_Avg': (0, 6000),
            'PM25_FLOW_Avg': (1.164, 1.236),
            'PM25_AMB_TEMP_Avg': (-20, 50),
            'PM25_AMB_RH_Avg': (None, 95),
            'PM25_BARO_PRES_Avg': (900, 1100),
            'PM10_CONC_Avg': (0, 10000)
        }

        vars_conc = ["PM10_CONC_Avg", "PM25_CONC_Avg"]
        
        def verificar_limites(row):
            tiene_fuera_rango = False
            tiene_dato_conc = False
            tiene_alguna_variable = False
            tiene_no_conc_vacia = False
            concentraciones_dentro_rango = True

            for col, (lim_inf, lim_sup) in limites.items():
                valor = row[col]

                # Si toda la columna está vacía
                if pd.isna(valor):
                    # Registrar si falta una variable no concentración
                    if col not in vars_conc:
                        tiene_no_conc_vacia = True
                    continue

                tiene_alguna_variable = True

                # Verificar si es concentración
                es_conc = col in vars_conc
                if es_conc:
                    tiene_dato_conc = True

                # Verificar fuera de rango
                if (lim_inf is not None and valor < lim_inf) or (lim_sup is not None and valor > lim_sup):
                    tiene_fuera_rango = True
                    if es_conc:
                        concentraciones_dentro_rango = False

            # --- Aplicación de reglas ---
            if not tiene_alguna_variable:                    # Toda la fila vacía
                return "ND"
            if not tiene_dato_conc:                          # No hay concentraciones
                return "ND"
            if tiene_fuera_rango:                            # Cualquier variable fuera de rango
                return "M"
            if tiene_no_conc_vacia and concentraciones_dentro_rango:  # Falta variable no concentración
                return "Dudoso"
            return "C"                                       # Todas dentro de rango

        df_nuevo["Bandera_RF"] = df_nuevo.apply(verificar_limites, axis=1)

        # ======================================================
        # CONSISTENCIA TEMPORAL
        # ======================================================
        def consistencia_temporal(serie):
            resultado = []
            for i in range(len(serie)):
                a = serie[i]
                # Verificar si el valor actual es válido (no NaN ni None)
                if a is None or (isinstance(a, float) and np.isnan(a)):
                    resultado.append("ND")  # Sin dato
                    continue

                # Evaluar solo si hay al menos dos anteriores válidos
                if i >= 2:
                    b, c = serie[i-1], serie[i-2]
                    # Comprobar que los tres sean válidos
                    if all(x is not None and not (isinstance(x, float) and np.isnan(x)) for x in [a, b, c]):
                        # Marcar solo si los tres son iguales y el siguiente (si existe) es distinto
                        if a == b == c and (i == len(serie)-1 or serie[i+1] != a):
                            resultado.append("D")
                            continue

                resultado.append("C")

            return resultado

        df_nuevo['Bandera_CT_PM25'] = consistencia_temporal(df_nuevo['PM25_CONC_Avg'])
        df_nuevo['Bandera_CT_PM10'] = consistencia_temporal(df_nuevo['PM10_CONC_Avg'])

        # ======================================================
        # CONSISTENCIA INTERNA
        # ======================================================
        df_nuevo['ratio'] = df_nuevo['PM25_CONC_Avg'] / df_nuevo['PM10_CONC_Avg']
        df_nuevo['Bandera_CI_PM25'] = df_nuevo['Bandera_CI_PM10'] = df_nuevo['ratio'].apply(
            lambda x: 'ND' if pd.isna(x) else ('D' if x > 1 else 'C')
        )
        
        # ======================================================
        # EXTRAER DATAFRAME FINAL REDUCIDO
        # ======================================================
        prefijo = ("Bandera",)
        columnas_extra = ["PM25_CONC_Avg", "PM10_CONC_Avg", "ratio"]
        columnas_prefijo = [col for col in df_nuevo.columns if col.startswith(prefijo)]
        columnas_seleccionadas = columnas_extra + columnas_prefijo
        df_final = df_nuevo[columnas_seleccionadas].copy()

        # ======================================================
        # ESTADO FINAL
        # ======================================================
        def estado_final(filas):
            if filas.isna().all():
                return 'ND'
            if 'M' in filas.values:
                return 'M'
            elif 'D' in filas.values:
                return 'D'
            elif all(f == 'C' for f in filas.values if pd.notna(f)):
                return 'C'
            else:
                return 'ND'

        df_final['Bandera_N1_PM25'] = df_final.apply(lambda x: estado_final(x[['Bandera_RF', 'Bandera_CT_PM25', 'Bandera_CI_PM25']]), axis=1)
        df_final['Bandera_N1_PM10'] = df_final.apply(lambda x: estado_final(x[['Bandera_RF', 'Bandera_CT_PM10', 'Bandera_CI_PM10']]), axis=1)

        

        # ======================================================
        # EXPORTAR Y MOSTRAR
        # ======================================================
        st.subheader("📋 Vista previa de resultados")
        st.dataframe(df_final.head(20))

        output = BytesIO()
        df_final.to_csv(output, index=True, encoding="utf-8")
        output.seek(0)

        nombre_archivo = f"{estacion}_PM_control_N1.csv".replace(" ", "_")

        st.download_button(
            label=f"💾 Descargar resultados ({nombre_archivo})",
            data=output,
            file_name=nombre_archivo,
            mime="text/csv")

        st.success("✅ Procesamiento completado correctamente.")


