import pandas as pd
from projects.investigacion_demografica_mexico.limpieza.organizadores import organizador_datos, normalizar_columnas
import numpy as np
import json


#limpieza y organizacion de datos

#datos de educacion
df = pd.read_excel('projects/investigacion_demografica_mexico/recoleccion/educacion_00.xlsx')
df = df.replace({np.nan: None})


datos_j = organizador_datos(df, '2020')


datos_norm = normalizar_columnas(datos_j)
datos_limpios = datos_norm.dropna(axis=1, how='all')


datos_limpios.to_csv('projects/investigacion_demografica_mexico/datos_organizados/csv/datos_de_educacion.csv', mode='a', header=True, index=True)


#datos de viviendas
df_v = pd.read_excel('projects/investigacion_demografica_mexico/recoleccion/vivienda_00.xlsx')

datos_v = organizador_datos(df_v, '2020')

datos_v_normalizados = normalizar_columnas(datos_v)
datos_v_limpios = datos_v_normalizados.dropna(axis=1, how='all')

datos_v_limpios.to_csv('projects/investigacion_demografica_mexico/datos_organizados/csv/datos_vivienda.csv', mode='a', header=True, index=True)


#datos de poblacion
df_poblacion = pd.read_excel('projects/investigacion_demografica_mexico/recoleccion/estructura_00.xlsx')

j_poblacion = organizador_datos(df_poblacion, '2020')
j_poblacion_normalizada = normalizar_columnas(j_poblacion)
j_poblacion_limpios = j_poblacion_normalizada.dropna(axis=1, how='all')

j_poblacion_limpios.to_csv('projects/investigacion_demografica_mexico/datos_organizados/csv/datos_poblacion.csv', mode='a', header=True, index=True)


cantidad_de_poblacion = np.array(j_poblacion_limpios['poblacion total'])
estados_poblacion = np.array(j_poblacion_limpios['estados'])

json_poblacion = {}

for i in range(estados_poblacion.size):
    if np.isnan(cantidad_de_poblacion[i]):
        cantidad = None
    else:
        cantidad = cantidad_de_poblacion[i]

    json_poblacion[estados_poblacion[i]] = cantidad


with open('projects/investigacion_demografica_mexico/datos_organizados/json/poblacion.json', 'w') as f:
    json.dump(json_poblacion, f)


#analisis

educacion_df = pd.read_csv('projects/investigacion_demografica_mexico/datos_organizados/csv/datos_de_educacion.csv')
vivienda_df = pd.read_csv('projects/investigacion_demografica_mexico/datos_organizados/csv/datos_vivienda.csv')

columnas_educacion = educacion_df.columns
columnas_vivienda = vivienda_df.columns

print(columnas_educacion)
print(columnas_vivienda)

with open('projects/investigacion_demografica_mexico/datos_organizados/texto/columnas_vivienda.txt', 'w') as f:
    f.write(str(columnas_vivienda))

with open('projects/investigacion_demografica_mexico/datos_organizados/texto/columnas_educacion.txt', 'w') as f:
    f.write(str(columnas_educacion))



"""
1: identificar los estados con más problemas en la escolaridad.
2: identificar los estados con más problemas en sus viviendas.
3: investigar posibles relaciones entre el analfabetismo de la poblacion con otros
    indicadores de la educacion y vivienda
"""


# 1


import matplotlib.pyplot as plt
import numpy as np

estados = educacion_df['estados']
porcentaje_de_poblacion_de_15_y_mas_sin_escolaridad = educacion_df['porcentaje de poblacion de 15 anos y mas sin escolaridad']

max(porcentaje_de_poblacion_de_15_y_mas_sin_escolaridad)

np.where(porcentaje_de_poblacion_de_15_y_mas_sin_escolaridad == 13.2926115644057)

# plot
fig, ax = plt.subplots(figsize=(15,6))

ax.bar(estados, porcentaje_de_poblacion_de_15_y_mas_sin_escolaridad, color='#6bbc6b')

ax.grid()

plt.xticks(rotation=40, ha='right')
plt.subplots_adjust(bottom=0.3)
plt.title('porcentaje de la población de 15 años y más sin escolaridad por estado')
plt.show()

fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/porcentaje_de_poblacion_de_15_y_mas_sin_escolaridad.png')
plt.close(fig)



# plot 2



with open('projects/investigacion_demografica_mexico/datos_organizados/json/poblacion.json', 'r') as f:
    poblacion_por_estado = json.load(f)


poblacion_de_6_a_12_analfabetas = educacion_df['poblacion de 6 a 14 anos que no sabe leer y escribir']


porcentajes_poblacion_de_6_a_12_analfabetas = []

for i in range(estados.size):
    estado = estados[i]

    if estado in poblacion_por_estado:

        porcentaje = (poblacion_de_6_a_12_analfabetas[i] / poblacion_por_estado[estado]) *100
 
        print(estado)
        print('\n')
        print(f'{poblacion_por_estado[estado]}')


        print(porcentaje)
        porcentajes_poblacion_de_6_a_12_analfabetas.append(porcentaje)

    print('---------------------')



import seaborn as sns


sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(15, 6))

fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

bars = ax.bar(estados, porcentajes_poblacion_de_6_a_12_analfabetas, color='#4a90e2', edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

ax.grid(color='gray', linestyle='--', linewidth=0.5)

ax.set_title('Porcentaje de Población de 6 a 12 años Analfabetas por Estado', fontsize=16, color='#333333')
ax.set_xlabel('Estados', fontsize=14, color='#333333')
ax.set_ylabel('Porcentaje de Analfabetismo (%)', fontsize=14, color='#333333')

plt.xticks(rotation=40, ha='right', fontsize=12)

plt.subplots_adjust(bottom=0.3)

plt.show()

fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/porcentajes_poblacion_de_6_a_12_analfabetas.png')
plt.close(fig)


educacion_df.columns
############3

porcentaje_poblacion_3_5_que_asiste_a_la_escuela = educacion_df['porcentaje de la poblacion de 3 a 5 anos que asiste a la escuela']

cmap = plt.get_cmap('Pastel1')
colors = cmap(np.linspace(0, 1, len(estados)))

fig, ax = plt.subplots(figsize=(15, 6))

fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

bars = ax.bar(estados, porcentaje_poblacion_3_5_que_asiste_a_la_escuela, color=colors, edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

ax.grid(color='gray', linestyle='--', linewidth=0.5)

ax.set_title('Porcentaje de Población por estado de 3 a 5 años que asiste a la escuela', fontsize=16, color='#333333')
ax.set_xlabel('Estados', fontsize=14, color='#333333')
ax.set_ylabel('Porcentaje (%)', fontsize=14, color='#333333')

plt.xticks(rotation=40, ha='right', fontsize=12)

plt.subplots_adjust(bottom=0.3)

plt.show()

fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/porcentaje_poblacion_3_5_que_asiste_a_la_escuela.png')
plt.close(fig)

##############################3
porcentaje_de_la_poblacion_de_12_a_14_que_asiste_a_la_escuela = educacion_df['porcentaje de la poblacion de 12 a 14 anos que asiste a la escuela']

fig, ax = plt.subplots(figsize=(15, 6))

fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

bars = ax.bar(estados, porcentaje_poblacion_3_5_que_asiste_a_la_escuela, color='paleturquoise', edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

ax.grid(color='gray', linestyle='--', linewidth=0.5)

ax.set_title('Porcentaje de Población por estado de 3 a 5 años que asiste a la escuela', fontsize=16, color='#333333')
ax.set_xlabel('Estados', fontsize=14, color='#333333')
ax.set_ylabel('Porcentaje (%)', fontsize=14, color='#333333')

plt.xticks(rotation=40, ha='right', fontsize=12)

plt.subplots_adjust(bottom=0.3)

plt.show()
fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/porcentaje_poblacion_3_5_que_asiste_a_la_escuela.png')
plt.close(fig)




############################################################
porcentaje_de_la_poblacion_de_15_y_mas_con_instruccion_superior = educacion_df['porcentaje de la poblacion de 15 anos y mas con instruccion superior']



fig, ax = plt.subplots(figsize=(15, 6))

fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

bars = ax.bar(estados, porcentaje_de_la_poblacion_de_15_y_mas_con_instruccion_superior, color='skyblue', edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

ax.grid(color='gray', linestyle='--', linewidth=0.5)

ax.set_title('Porcentaje de Población por estado de 15 años y más con instrucción superior', fontsize=16, color='#333333')
ax.set_xlabel('Estados', fontsize=14, color='#333333')
ax.set_ylabel('Porcentaje (%)', fontsize=14, color='#333333')

plt.xticks(rotation=40, ha='right', fontsize=12)

plt.subplots_adjust(bottom=0.3)

plt.show()
#fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/porcentaje_de_la_poblacion_de_15_y_mas_con_instruccion_superior.png')
plt.close(fig)


########

porcentaje_de_personas_de_15_y_mas_alfabetas = educacion_df['porcentaje de personas de 15 anos y mas alfabetas']

fig, ax = plt.subplots(figsize=(15, 6))

fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

bars = ax.bar(estados, porcentaje_de_personas_de_15_y_mas_alfabetas, color='teal', edgecolor='darkslategray')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

ax.grid(color='gray', linestyle='-.', linewidth=0.5)

ax.set_title('Porcentaje de la población de 15 años y más alfabetas', fontsize=16, color='#333333')
ax.set_xlabel('Estados', fontsize=14, color='#333333')
ax.set_ylabel('Porcentaje (%)', fontsize=14, color='#333333')

plt.xticks(rotation=40, ha='right', fontsize=12)

plt.subplots_adjust(bottom=0.3)

plt.show()
fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/porcentaje_de_personas_de_15_y_mas_alfabetas.png')
plt.close(fig)




# 2

from matplotlib.colors import Normalize

estados = vivienda_df['estados']

viviendas_decad_a_cien_que_cuentan_con_electricidad_agua_y_drenaje = vivienda_df['viviendas, de cada cien que cuentan con electricidad, agua y drenaje']



cmap = plt.get_cmap('RdBu')

norm = Normalize(vmin=viviendas_decad_a_cien_que_cuentan_con_electricidad_agua_y_drenaje.min(), 
                 vmax=viviendas_decad_a_cien_que_cuentan_con_electricidad_agua_y_drenaje.max())

colores = np.array(viviendas_decad_a_cien_que_cuentan_con_electricidad_agua_y_drenaje)
colors = cmap(norm(colores))


fig, ax = plt.subplots(figsize=(15, 6))

fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

bars = ax.bar(estados, viviendas_decad_a_cien_que_cuentan_con_electricidad_agua_y_drenaje, color=colors, edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

ax.grid(color='gray', linestyle='--', linewidth=0.5)

ax.set_title('Viviendas, de cada cien que cuentan con electricidad, agua y drenaje', fontsize=16, color='#333333')
ax.set_xlabel('Estados', fontsize=14, color='#333333')
ax.set_ylabel('Porcentaje (%)', fontsize=14, color='#333333')

plt.xticks(rotation=40, ha='right', fontsize=12)

plt.subplots_adjust(bottom=0.3)

plt.show()

fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/viviendas_decad_a_cien_que_cuentan_con_electricidad_agua_y_drenaje.png')
plt.close(fig)
 


#####
tasa_de_crecimiento_promedio_anual_de_las_viviendas_particulares_habitadas = vivienda_df['tasa de crecimiento promedio anual de las viviendas particulares habitadas']



cmap = plt.get_cmap('RdYlGn')

norm = Normalize(vmin=tasa_de_crecimiento_promedio_anual_de_las_viviendas_particulares_habitadas.min(), 
                 vmax=tasa_de_crecimiento_promedio_anual_de_las_viviendas_particulares_habitadas.max())

colores = np.array(tasa_de_crecimiento_promedio_anual_de_las_viviendas_particulares_habitadas)
colors = cmap(norm(colores))


fig, ax = plt.subplots(figsize=(15, 6))

fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

bars = ax.bar(estados, tasa_de_crecimiento_promedio_anual_de_las_viviendas_particulares_habitadas, color=colors, edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')

ax.grid(color='gray', linestyle='--', linewidth=0.5)

ax.set_title('Tasa de crecimiento promedio anual de las viviendas particulares', fontsize=16, color='#333333')
ax.set_xlabel('Estados', fontsize=14, color='#333333')
ax.set_ylabel('Porcentaje (%)', fontsize=14, color='#333333')

plt.xticks(rotation=40, ha='right', fontsize=12)

plt.subplots_adjust(bottom=0.3)

plt.show()

fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/tasa_de_crecimiento_promedio_anual_de_las_viviendas_particulares_habitadas.png')
plt.close(fig)
 

#############################3

porcentaje_de_viviendas_particulares_habitadas_que_disponen_de_internet = vivienda_df['porcentaje de viviendas particulares habitadas que disponen de internet']



fig, ax = plt.subplots(figsize=(15, 6))

fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

ax.bar(estados, porcentaje_de_viviendas_particulares_habitadas_que_disponen_de_internet, edgecolor='black')

""" 
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom', fontsize=12, color='black')
"""
ax.grid(color='gray', linestyle='--', linewidth=0.5)

ax.set_title('Porcentaje de viviendas habitadas por estado que disponen de internet', fontsize=16, color='#333333')
ax.set_xlabel('Estados', fontsize=14, color='#333333')
ax.set_ylabel('Porcentaje (%)', fontsize=14, color='#333333')

plt.xticks(rotation=40, ha='right', fontsize=12)

plt.subplots_adjust(bottom=0.3)

plt.show()

fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/porcentaje_de_viviendas_particulares_habitadas_que_disponen_de_internet.png')
plt.close(fig)
 

####


poblacion_de_6_a_12_analfabetas = educacion_df['poblacion de 6 a 14 anos que no sabe leer y escribir']


porcentajes_poblacion_de_6_a_12_analfabetas = []

with open('projects/investigacion_demografica_mexico/datos_organizados/json/poblacion.json', 'r') as f:
    poblacion_por_estado = json.load(f)

for i in range(estados.size):
    estado = estados[i]

    if estado in poblacion_por_estado:

        porcentaje = (poblacion_de_6_a_12_analfabetas[i] / poblacion_por_estado[estado]) *100
 
        print(estado)
        print('\n')
        print(f'{poblacion_por_estado[estado]}')


        print(porcentaje)
        porcentajes_poblacion_de_6_a_12_analfabetas.append(porcentaje)

    print('---------------------')

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

norm_viviendas = normalize(porcentaje_de_viviendas_particulares_habitadas_que_disponen_de_internet)
norm_analfabetas = normalize(porcentajes_poblacion_de_6_a_12_analfabetas)

_x_ = [i for i in range(estados.size)]

fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(_x_, norm_viviendas, label='Porcentaje de viviendas con internet', color='blue')
ax.plot(_x_, norm_analfabetas, label='Porcentaje de población analfabeta (6-14 años)', color='red')

ax.set_title('Comparación de porcentajes normalizados', fontsize=16)
ax.set_xlabel('Estados', fontsize=14)
ax.set_ylabel('Porcentaje Normalizado', fontsize=14)

ax.legend()

ax.set_xticks(_x_)
ax.set_xticklabels(estados, rotation=40, ha='right', fontsize=12)

plt.show()

plt.close(fig)

_x_
porcentaje_de_viviendas_particulares_habitadas_que_disponen_de_internet.size



#3

"""
buscar posibles relaciones del analfabetismo de los estados
"""

#analfabetismo de las personas de los 6 a los 14 años:
# hipotesis 1: posible relación entre la asistencia a la escuela a edad temprana
    # de 3 a 5 años con la habilidad de escribir y de leer a la edad de 6 a 14
    #años

poblacion_por_estado
poblacion_de_6_a_14_que_sabe_leer_y_escribir = educacion_df['poblacion de 6 a 14 anos que sabe leer y escribir']
poblacion_de_6_a_14_que_no_sabe_leer_y_escribir = educacion_df['poblacion de 6 a 14 anos que no sabe leer y escribir']
poblacion_de_6_a_14_que_no_especifico_si_sabe_o_no_leer_y_escribir = educacion_df['poblacion de 6 a 14 anos que no especifico si sabe o no leer y escribir']
porcentaje_de_la_poblacion_de_3_a_5_que_asiste_a_la_escuela = educacion_df['porcentaje de la poblacion de 3 a 5 anos que asiste a la escuela']


porcentajes_analfabetas = []
porcentajes_alfabetas = []

for i in range(estados.size):
    estado = estados[i]
    poblacion_total = (poblacion_de_6_a_14_que_sabe_leer_y_escribir[i] + 
                       poblacion_de_6_a_14_que_no_sabe_leer_y_escribir[i] +
                       poblacion_de_6_a_14_que_no_especifico_si_sabe_o_no_leer_y_escribir[i])

    poblacion_alfabeta = poblacion_de_6_a_14_que_sabe_leer_y_escribir[i]
    
    print(estado)
    print(f'poblacion alfabeta: {poblacion_alfabeta}')
    print(f'poblacion total: {poblacion_total}')
    print(f'porsentaje alfabeta: {(poblacion_alfabeta / poblacion_total) * 100 if poblacion_alfabeta is not 0 else 0}')
    print('--------------------------')
    porcentaje_alfabeta = (poblacion_alfabeta / poblacion_total) * 100 if poblacion_alfabeta is not 0 else 0
    porcentajes_alfabetas.append(porcentaje_alfabeta)

    poblacion_analfabeta = poblacion_de_6_a_14_que_no_sabe_leer_y_escribir[i]
    porcentaje_analfabeta = (poblacion_analfabeta / poblacion_total) * 100 if poblacion_analfabeta is not 0 else 0
    porcentajes_analfabetas.append(porcentaje_analfabeta)

porcentajes_alfabetas = np.array(porcentajes_alfabetas)
porcentajes_analfabetas = np.array(porcentajes_analfabetas)
porcentaje_3_5_ = np.array(porcentaje_de_la_poblacion_de_3_a_5_que_asiste_a_la_escuela)


correlacion_alfb = np.corrcoef(porcentaje_3_5_, porcentaje_alfabeta)
correlacion_analrfb = np.corrcoef(porcentaje_3_5_, porcentajes_analfabetas)


#plot

fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(_x_, porcentajes_analfabetas, label='Porcentaje de la población de 6 a 14 años \n que no sabe leer y escribir', color='blue')
ax.plot(_x_, porcentajes_alfabetas, label='Porcentaje de la población de 6 a 14 años \n que sabe leer y escribir', color='red')
ax.plot(_x_, porcentaje_3_5_, label='Porcentaje de la población de 3 a 5 años que \n asiste a la escuela', color='pink')

ax.set_title('Relación entre la temprana asistencia a la escuela (3 a 5 años) y la habilidad de leer y escribir entre los 6 y 14 años', fontsize=16)
ax.set_xlabel('Estados', fontsize=14)
ax.set_ylabel('Porcentaje Normalizado', fontsize=14)

correlacion_alfb = np.corrcoef(porcentaje_3_5_, porcentajes_alfabetas)[0, 1]
correlacion_analrfb = np.corrcoef(porcentaje_3_5_, porcentajes_analfabetas)[0, 1]

ax.annotate(f'Correlación temprana asistencia \n a la escuela con alfabetismo: {correlacion_alfb:.2f}', xy=(1.2, 0.6), xycoords='axes fraction', fontsize=12, color='darkslategray', ha='center')
ax.annotate(f'Correlación temprana asistencia \n a la escuela con analfabetismo: {correlacion_analrfb:.2f}', xy=(1.2, 0.45), xycoords='axes fraction', fontsize=12, color='purple', ha='center')

ax.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

ax.set_xticks(_x_)
ax.set_xticklabels(estados, rotation=40, ha='right', fontsize=12)

plt.subplots_adjust(bottom=0.3)
plt.subplots_adjust(right=0.7)


plt.show()
fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/relacion_temprana_asistencia_a_la_escuela_con_alfabetismo.png')

plt.close(fig)

##########
