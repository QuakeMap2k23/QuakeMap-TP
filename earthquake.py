import streamlit as st

# Configuración de la página (debe ser la primera llamada a Streamlit)
st.set_page_config(page_title="Análisis de Terremotos Globales 2023", layout="wide")

import pandas as pd
import folium as folium
from streamlit_folium import folium_static
import networkx as nx
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# Intenta importar plotly, si no está disponible, usará matplotlib
try:
    import plotly.express as px
    import plotly.graph_objects as go
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False
    st.warning("Plotly no está instalado. Se usará matplotlib para las visualizaciones. Para mejores gráficos, instala plotly con: pip install plotly")

# Carga de datos
@st.cache_data
def load_data():
    data = pd.read_csv("earthquakes_2023_global.csv")
    data = data.head(1500)
    data['time'] = pd.to_datetime(data['time'])
    return data

# Función para crear el mapa
def create_map(df):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=2)
    
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
            popup=folium.Popup(popup_content, max_width=300),  # Puedes ajustar el max_width aquí
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.7
        ).add_to(m)
    
    return m


# Función para crear el grafo
@st.cache_resource
def create_graph(df):
    coords = df[['latitude', 'longitude']].to_numpy()
    dist_matrix = distance_matrix(coords, coords)
    
    G = nx.Graph()
    
    for i, row in df.iterrows():
        G.add_node(i, pos=(row['longitude'], row['latitude']), mag=row['mag'], depth=row['depth'], place=row['place'])
    
    # Conectar solo los 5 terremotos más cercanos para cada terremoto
    for i in range(len(dist_matrix)):
        nearest = dist_matrix[i].argsort()[1:6]
        for j in nearest:
            G.add_edge(i, j, weight=dist_matrix[i][j])
    
    return G

# Función para crear la visualización interactiva del grafo
def create_interactive_graph(G, title):
    if USE_PLOTLY:
        pos = nx.get_node_attributes(G, 'pos')
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=10,
                colorbar=dict(thickness=15, title='Magnitud', xanchor='left', titleside='right')
            )
        )

        node_magnitudes = [G.nodes[node]['mag'] for node in G.nodes()]
        node_depths = [G.nodes[node]['depth'] for node in G.nodes()]
        node_places = [G.nodes[node]['place'] for node in G.nodes()]
        node_texts = [f"Mag: {mag}<br>Prof: {depth} km<br>Place: {place}" for mag, depth, place in zip(node_magnitudes, node_depths, node_places)]

        node_trace.marker.color = node_magnitudes
        node_trace.text = node_texts

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=title,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, ax=ax, node_size=20, node_color='red', with_labels=False)
        ax.set_title(title)
        return fig

# Función principal
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
    
    st.subheader("Grafo Completo de Distancias")
    fig_complete = create_interactive_graph(G, "Grafo Completo")
    if USE_PLOTLY:
        st.plotly_chart(fig_complete, use_container_width=True)
    else:
        st.pyplot(fig_complete)
   
if __name__ == "__main__":
    main()
