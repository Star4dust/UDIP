"""
UDIP v3 — modules/visualizations.py
Advanced visualization capabilities: 3D plots, interactive charts, maps, etc.
"""

import streamlit as st
import pandas as pd
import numpy as np
from modules.core import log

def create_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str = None):
    """Create 3D scatter plot."""
    try:
        import plotly.express as px
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                           title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        log.error(f"Failed to create 3D scatter: {e}")
        st.error("Failed to create 3D scatter plot")

def create_3d_surface(df: pd.DataFrame, x_col: str, y_col: str, z_col: str):
    """Create 3D surface plot."""
    try:
        import plotly.graph_objects as go
        
        # Check if we have enough data points
        if len(df) < 4:
            st.error("Need at least 4 data points for 3D surface")
            return
            
        # Check if data spans 3D space (not coplanar)
        points = df[[x_col, y_col, z_col]].dropna().values
        if len(points) < 4:
            st.error("Need at least 4 valid data points for 3D surface")
            return
            
        # Compute rank to check dimensionality
        rank = np.linalg.matrix_rank(points)
        if rank < 3:
            st.error("Data points are coplanar (lie in a plane) and cannot form a 3D surface. Try a 2D heatmap or 3D scatter plot instead.")
            return
            
        # Create meshgrid for surface - use unique values and interpolate
        x_unique = sorted(df[x_col].unique())
        y_unique = sorted(df[y_col].unique())
        
        if len(x_unique) < 2 or len(y_unique) < 2:
            st.error("Need at least 2 unique values in X and Y for surface plot")
            return
            
        # Create meshgrid
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Interpolate Z values
        from scipy.interpolate import griddata
        points_2d = df[[x_col, y_col]].values
        values = df[z_col].values
        Z = griddata(points_2d, values, (X, Y), method='linear', fill_value=np.nanmean(values))
        
        # Handle NaN values
        Z = np.nan_to_num(Z, nan=np.nanmean(values))
        
        fig = go.Figure(data=[go.Surface(z=Z, x=x_unique, y=y_unique, colorscale='Viridis')])
        fig.update_layout(
            title=f"3D Surface: {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            autosize=True
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        log.error(f"Failed to create 3D surface: {e}")
        st.error(f"Failed to create 3D surface plot: {str(e)}")

def create_interactive_dashboard(df: pd.DataFrame, columns: list):
    """Create interactive dashboard with multiple charts."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Scatter Matrix", "Parallel Coordinates", "Box Plot", "Violin Plot"),
            specs=[[{"type": "scatter"}, {"type": "parcoords"}],
                   [{"type": "box"}, {"type": "violin"}]]
        )
        
        # Scatter matrix
        if len(columns) >= 2:
            fig.add_trace(
                go.Scatter(x=df[columns[0]], y=df[columns[1]], mode='markers'),
                row=1, col=1
            )
        
        # Parallel coordinates
        if len(columns) >= 3:
            fig.add_trace(
                go.Parcoords(
                    dimensions=[dict(label=col, values=df[col]) for col in columns[:5]]
                ),
                row=1, col=2
            )
        
        # Box plot
        if len(columns) >= 1:
            fig.add_trace(
                go.Box(y=df[columns[0]], name=columns[0]),
                row=2, col=1
            )
        
        # Violin plot
        if len(columns) >= 1:
            fig.add_trace(
                go.Violin(y=df[columns[0]], name=columns[0]),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Interactive Dashboard")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        log.error(f"Failed to create dashboard: {e}")
        st.error("Failed to create interactive dashboard")

def create_geospatial_map(df: pd.DataFrame, lat_col: str, lon_col: str, 
                         popup_col: str = None, color_col: str = None):
    """Create geospatial map using Folium."""
    try:
        import folium
        from streamlit_folium import folium_static
        # Check if latitude and longitude columns exist
        if lat_col not in df.columns or lon_col not in df.columns:
            st.error("Latitude and Longitude columns not found")
            return
        
        # Create map centered on mean coordinates
        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add markers
        for idx, row in df.iterrows():
            popup_text = str(row[popup_col]) if popup_col else f"Point {idx}"
            color = 'blue' if not color_col else row[color_col]
            
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=popup_text,
                icon=folium.Icon(color=color if isinstance(color, str) else 'blue')
            ).add_to(m)
        
        # Display map
        folium_static(m)
    except Exception as e:
        log.error(f"Failed to create map: {e}")
        st.error("Failed to create geospatial map")

def create_network_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, 
                        source_col: str, target_col: str, weight_col: str = None):
    """Create network graph visualization."""
    try:
        import networkx as nx
        import plotly.graph_objects as go

        # Create graph
        G = nx.Graph()

        # Add nodes by identifier values
        if source_col in nodes_df.columns:
            for _, row in nodes_df.iterrows():
                identifier = row[source_col]
                if identifier is not None:
                    G.add_node(identifier, **row.to_dict())
        else:
            for idx, row in nodes_df.iterrows():
                G.add_node(idx, **row.to_dict())

        # Add edges
        for _, row in edges_df.iterrows():
            source = row[source_col]
            target = row[target_col]
            if source is None or target is None:
                continue
            weight = row[weight_col] if weight_col else 1
            G.add_edge(source, target, weight=weight)

        if G.number_of_nodes() == 0:
            st.error("No valid nodes were created for the network graph.")
            return

        # Create positions
        pos = nx.spring_layout(G, seed=42)

        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=[G.degree(n) for n in G.nodes()],
                colorbar=dict(
                    thickness=15,
                    title=dict(text='Node Degree', side='right'),
                    xanchor='left'
                )
            )
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Network Graph',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40)
                       ))
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.error("NetworkX not installed. Install with: pip install networkx")
    except Exception as e:
        log.error(f"Failed to create network graph: {e}")
        st.error("Failed to create network graph")

def create_advanced_chart(df: pd.DataFrame, chart_type: str, **kwargs):
    """Create advanced chart based on type."""
    try:
        if chart_type == '3d_scatter':
            create_3d_scatter(df, **kwargs)
        elif chart_type == '3d_surface':
            create_3d_surface(df, **kwargs)
        elif chart_type == 'interactive_dashboard':
            create_interactive_dashboard(df, **kwargs)
        elif chart_type == 'geospatial_map':
            create_geospatial_map(df, **kwargs)
        elif chart_type == 'network_graph':
            create_network_graph(df, **kwargs)
        else:
            st.error(f"Unknown chart type: {chart_type}")
    except Exception as e:
        log.error(f"Failed to create {chart_type}: {e}")
        st.error(f"Failed to create {chart_type}")