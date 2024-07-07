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
1: identificar los estados con más problemas en la escolaridad
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
#fig.savefig('projects/investigacion_demografica_mexico/datos_organizados/plots/porcentaje_de_personas_de_15_y_mas_alfabetas.png')
plt.close(fig)
