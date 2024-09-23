# Importar bibliotecas
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import networkx as nx
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# Configuración de la página
# Título, icono y diseño de la página
st.set_page_config(page_title="Análisis de Terremotos Globales 2023", layout="wide", page_icon=":earth_americas:")

# Aplicar un tema oscuro a toda la aplicación
st.markdown("""
    <style>
    .reportview-container {
        background: black;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Intenta importar plotly, si no está disponible, usará matplotlib
# Se mostrará una advertencia si plotly no está instalado en el entorno actual
# Para instalar plotly, ejecute: pip install plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False
    st.warning("Plotly no está instalado. Se usará matplotlib para las visualizaciones. Para mejores gráficos, instala plotly con: pip install plotly")

# Carga de datos de terremotos en 2023
# Se limita a 1500 registros para mejorar la velocidad de carga (500 registros por alumno)
@st.cache_data
def load_data():
    data = pd.read_csv("earthquakes_2023_global.csv")
    data = data.head(1500)
    data['time'] = pd.to_datetime(data['time'])
    return data

# Función para crear el mapa
# Se crea un círculo en el mapa para cada terremoto con un radio proporcional a la magnitud
# El círculo se rellena con un color rojo y se muestra la información del terremoto en un popup
def create_map(df):
    map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=2)
    
    for _, row in df.iterrows():
        popup_content = f"""
        <div style="font-size: 12px; width: 150px;">
            <strong>Magnitud:</strong> {row['mag']}<br>
            <strong>Profundidad:</strong> {row['depth']} km<br>
            <strong>Fecha:</strong> {row['time'].date()}<br>
            <strong>Lugar:</strong> {row['place']}
        </div>
        """
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=row['mag'] * 1.5,
            popup=folium.Popup(popup_content, max_width=300),
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.7
        ).add_to(map)
    
    return map

# Función para crear el grafo
# Se crea un grafo no dirigido con los terremotos como nodos (latitud y longitud) y las distancias como aristas (distancia euclidiana)
# Se conectan los 5 terremotos más cercanos a cada terremoto en el grafo
# Se almacenan los atributos de magnitud, profundidad y lugar en cada nodo
# Se almacena la distancia entre los nodos en cada arista
# Se devuelve el grafo creado
@st.cache_resource
def create_graph(df):
    coords = df[['latitude', 'longitude']].to_numpy()
    dist_matrix = distance_matrix(coords, coords)
    
    Grafo = nx.Graph()
    
    for i, row in df.iterrows():
        Grafo.add_node(i, pos=(row['longitude'], row['latitude']), mag=row['mag'], depth=row['depth'], place=row['place'])
    
    # Conectar solo los 5 terremotos más cercanos para cada terremoto
    for i in range(len(dist_matrix)):
        nearest = dist_matrix[i].argsort()[1:6]
        for j in nearest:
            Grafo.add_edge(i, j, weight=dist_matrix[i][j])
    
    return Grafo

# Función para crear la visualización interactiva del grafo
# Se crea un gráfico interactivo con plotly o matplotlib según la disponibilidad
# Se muestran las distancias entre los nodos en las aristas y la magnitud, profundidad y lugar en los nodos
# Se devuelve el gráfico creado
# Se usó como guía chatgpt y github copilot para usar plotly
def create_interactive_graph(G, title):
    if USE_PLOTLY:
        pos = nx.get_node_attributes(G, 'pos')
        edge_traces = []
        annotation_traces = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            distance = G[edge[0]][edge[1]]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            
            # Texto de la distancia en el medio de la arista
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            annotation_trace = go.Scatter(
                x=[mid_x],
                y=[mid_y],
                text=[f'{int(distance)} km'],
                mode='text',
                textposition='middle center',
                textfont=dict(size=8, color='white'),
                hoverinfo='none',
                showlegend=False
            )
            
            edge_traces.append(edge_trace)
            annotation_traces.append(annotation_trace)

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=8,
                colorbar=dict(thickness=15, title='Magnitud', xanchor='left', titleside='right')
            )
        )

        node_magnitudes = [G.nodes[node]['mag'] for node in G.nodes()]
        node_depths = [G.nodes[node]['depth'] for node in G.nodes()]
        node_places = [G.nodes[node]['place'] for node in G.nodes()]
        node_texts = [f"Mag: {mag}<br>Prof: {depth} km<br>Place: {place}" for mag, depth, place in zip(node_magnitudes, node_depths, node_places)]

        node_trace.marker.color = node_magnitudes
        node_trace.text = node_texts

        layout = go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            width=1000,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )

        fig = go.Figure(data=edge_traces + annotation_traces + [node_trace], layout=layout)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(15, 12), facecolor='black')
        ax.set_facecolor('black')
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, ax=ax, node_size=20, node_color='red', with_labels=False, edge_color='lightgray', width=0.5)
        
        # Dibujar etiquetas de aristas con distancias
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {edge: f'{int(weight)} km' for edge, weight in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_color='white')
        
        ax.set_title(title, fontsize=16, color='white')
        plt.tight_layout()
        return fig

# Función principal
# Se muestra el título de la página y la cantidad de registros en el dataset
def main():
    st.title("Análisis de Terremotos Globales 2023")

    data = load_data()

    st.write(f"El dataset contiene {len(data)} registros de terremotos en el año 2023.")

    # Mapa interactivo
    st.subheader("Visualización de Epicentros de Terremotos")
    earthquake_map = create_map(data)
    folium_static(earthquake_map)

    # Creación del grafo
    G = create_graph(data)
    
    st.subheader("Grafo de Distancias entre Terremotos")
    fig_complete = create_interactive_graph(G, "Grafo de Distancias")
    if USE_PLOTLY:
        st.plotly_chart(fig_complete, use_container_width=True)
    else:
        st.pyplot(fig_complete)

if __name__ == "__main__":
    main()
