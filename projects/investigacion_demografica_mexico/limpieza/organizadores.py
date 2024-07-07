import numpy as np
import pandas as pd
from unidecode import unidecode

def normalizar_columnas(df):
    df.columns = [unidecode(col).lower() for col in df.columns]
    return df

def organizador_datos(df, fecha):

    estados = np.array(df['desc_entidad'])
    indicadores = np.array(df['indicador'])
    datos_2020 = np.array(df[fecha])
    municipios = np.array(df['cve_municipio'])
    unidades_de_medida = np.array(df['unidad_medida'])

    j_organizado = {
        'estados': []
    }

    uniq_indicador = np.unique(indicadores)

    for indicador_unico in uniq_indicador:
        j_organizado[indicador_unico] = []

    #j_organizado['unidades'] = []

    for estado in np.unique(estados):
        j_organizado['estados'].append(unidecode(estado).lower())

        indices = np.where(estados == estado)

        indicadores_estado = indicadores[indices]
        datos_2020_estado = datos_2020[indices]
        municipios_estado = municipios[indices]

        indices_estatales = np.where(municipios_estado == 0)
        #unidades_estado = unidades_de_medida[indices]

        indicadores_estatales = indicadores_estado[indices_estatales]
        datos_estatales = datos_2020_estado[indices_estatales]
        

        temp_dict = {indicador: np.nan for indicador in uniq_indicador}
        dict_unidad = {indicador: None for indicador in uniq_indicador}

        for i in range(indicadores_estatales.size):
            indicador = indicadores_estatales[i]
            dato = datos_estatales[i]
            #unidad = unidades_estado[i]
            temp_dict[indicador] = dato
            #dict_unidad[indicador] = unidad

        for indicador_unico in uniq_indicador:
            j_organizado[indicador_unico].append(temp_dict[indicador_unico])

    # Normalizamos las unidades de medida
    """ 
    for indicador_unico in uniq_indicador:
        j_organizado['unidades'].append(unidecode(dict_unidad[indicador_unico]).lower() if dict_unidad[indicador_unico] else np.nan)
"""
    # Asegurarse de que todas las listas tengan la misma longitud
    max_length = len(j_organizado['estados'])
    for key in j_organizado.keys():
        if len(j_organizado[key]) < max_length:
            j_organizado[key].extend([np.nan] * (max_length - len(j_organizado[key])))

    df_organizado = pd.DataFrame(j_organizado)

    return df_organizado
