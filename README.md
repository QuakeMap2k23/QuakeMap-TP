# Análisis de Terremotos Globales 2023

## Descripción
Esta aplicación permite analizar los terremotos ocurridos en el año 2023 a nivel global. Utiliza diversas bibliotecas de Python para la visualización y análisis de datos geoespaciales y de redes.

## Integrantes
| Código      | Nombres y Apellidos                |
|-------------|------------------------------------|
| U202212648  | Joaquin Antonio Cortez Quezada     |
| U202212684  | Harold Miguel Elías Sanchez        |
| U202212338  | Rodrigo Adrián López Huamán        |


## Bibliotecas Utilizadas
- `streamlit`: Para la creación de la interfaz web interactiva.
- `pandas`: Para la manipulación y análisis de datos.
- `folium`: Para la visualización de mapas interactivos.
- `streamlit_folium`: Para integrar mapas de Folium en Streamlit.
- `networkx`: Para la creación y análisis de grafos.
- `scipy.spatial`: Para el cálculo de matrices de distancia.
- `matplotlib`: Para la visualización de gráficos.
- `plotly`: (Opcional) Para la visualización interactiva de gráficos.

## Funcionalidades
- **Visualización de Epicentros de Terremotos**: Muestra un mapa interactivo con los epicentros de los terremotos, donde cada círculo representa un terremoto con un radio proporcional a su magnitud.
- **Grafo de Distancias entre Terremotos**: Crea y visualiza un grafo no dirigido donde los nodos representan terremotos y las aristas representan las distancias euclidianas entre ellos.

## Instalación
Para ejecutar esta aplicación, asegúrese de tener instaladas las siguientes bibliotecas:
```bash
pip install streamlit pandas folium streamlit_folium networkx scipy matplotlib
```
Para una mejor visualización, se recomienda instalar `plotly`:
```bash
pip install plotly
```

## Ejecución
Para ejecutar la aplicación, utilice el siguiente comando:
```bash
streamlit run nombre_del_archivo.py
```

## Estructura del Código
1. **Importación de Bibliotecas**: Se importan todas las bibliotecas necesarias.
2. **Configuración de la Página**: Se configura el título, icono y diseño de la página, y se aplica un tema oscuro.
3. **Carga de Datos**: Se carga el dataset de terremotos del año 2023 y se limita a 1500 registros.
4. **Creación del Mapa**: Se crea un mapa interactivo con los epicentros de los terremotos.
5. **Creación del Grafo**: Se crea un grafo no dirigido con los terremotos como nodos y las distancias como aristas.
6. **Visualización del Grafo**: Se crea una visualización interactiva del grafo utilizando `plotly` o `matplotlib`.
7. **Función Principal**: Se muestra el título de la página, la cantidad de registros en el dataset, el mapa interactivo y el grafo de distancias.

## Datos
El dataset utilizado debe estar en formato CSV y contener información sobre los terremotos ocurridos en el año 2023. El archivo debe llamarse `earthquakes_2023_global.csv` y debe estar en el mismo directorio que el script de la aplicación.

## Créditos
Esta aplicación fue desarrollada utilizando `chatgpt` y `GitHub Copilot` como guías para la implementación de `plotly`.
