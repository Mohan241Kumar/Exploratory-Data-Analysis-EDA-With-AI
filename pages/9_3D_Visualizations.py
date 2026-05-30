import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="3D Visualizations", page_icon="📊", layout="wide")

st.title("📊 3D Visualizations")
st.markdown("Explore your data in three dimensions with various 3D plot types and interactive features.")

if st.session_state.get('data') is None:
    st.warning("⚠️ Please upload a dataset first from the main page.")
    st.stop()

data = st.session_state.data

# Sidebar controls
st.sidebar.header("🎛️ 3D Plot Controls")

# Get numeric columns for 3D plots
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

if len(numeric_cols) < 3:
    st.error("❌ This dataset needs at least 3 numeric columns for 3D visualizations.")
    st.info("💡 Try uploading a dataset with more numerical features.")
    st.stop()

# Plot type selection
plot_type = st.sidebar.selectbox(
    "Select 3D Plot Type",
    [
        "3D Scatter Plot",
        "3D Surface Plot", 
        "3D Line Plot",
        "3D Bar Chart",
        "3D Mesh Plot",
        "3D Contour Plot",
        "3D Volume Plot",
        "3D PCA Visualization",
        "3D Clustering",
        "3D Density Plot",
        "3D Heatmap",
        "3D Parallel Coordinates"
    ],
    help="Choose the type of 3D visualization to create"
)

# Initialize column variables
x_col = y_col = z_col = value_col = None

# Column selection based on plot type
if plot_type in ["3D Scatter Plot", "3D Line Plot", "3D Bar Chart", "3D Density Plot"]:
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols, index=0)
    with col2:
        y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    with col3:
        z_col = st.selectbox("Z-axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)

elif plot_type in ["3D Surface Plot", "3D Mesh Plot", "3D Contour Plot"]:
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols, index=0)
    with col2:
        y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    
    z_col = None  # Will be calculated for surface plots

elif plot_type in ["3D PCA Visualization", "3D Clustering"]:
    st.sidebar.info("Using all numeric columns for dimensionality reduction/clustering")
    x_col, y_col, z_col = None, None, None

elif plot_type == "3D Volume Plot":
    col1, col2, col3, col4 = st.sidebar.columns(4)
    
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols, index=0)
    with col2:
        y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    with col3:
        z_col = st.selectbox("Z-axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)
    with col4:
        value_col = st.selectbox("Value", numeric_cols, index=3 if len(numeric_cols) > 3 else 0)

elif plot_type == "3D Heatmap":
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols, index=0)
    with col2:
        y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    with col3:
        z_col = st.selectbox("Z-axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)

elif plot_type == "3D Parallel Coordinates":
    selected_cols = st.sidebar.multiselect(
        "Select columns for parallel coordinates",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )
# Initialize color_by and size_by variables
color_by = "None"
size_by = "None"

# Color and size options
if plot_type in ["3D Scatter Plot", "3D Line Plot", "3D Bar Chart", "3D Density Plot", "3D Volume Plot"]:
    st.sidebar.subheader("🎨 Styling Options")
    
    # Color by categorical column
    if categorical_cols:
        color_by = st.sidebar.selectbox(
            "Color by",
            ["None"] + categorical_cols,
            help="Color points by a categorical variable"
        )
    else:
        color_by = "None"
    
    # Size by numeric column
    if len(numeric_cols) > 3:
        size_by = st.sidebar.selectbox(
            "Size by",
            ["None"] + [col for col in numeric_cols if col not in [x_col, y_col, z_col]],
            help="Size points by a numeric variable"
        )
    else:
        size_by = "None"

# Additional options
st.sidebar.subheader("⚙️ Additional Options")

# Sample data if too large
if len(data) > 1000:
    sample_size = st.sidebar.slider("Sample size", 100, min(1000, len(data)), 500)
    use_sample = st.sidebar.checkbox("Use sample data", value=True)
else:
    use_sample = False
    sample_size = len(data)

# Create the plot
def create_3d_plot(plot_type, data, x_col, y_col, z_col, color_by, size_by, use_sample, sample_size):
    """Create various types of 3D plots"""
    
    # Use the data as-is (sampling is handled in the main execution block)
    plot_data = data
    
    if plot_type == "3D Scatter Plot":
        fig = go.Figure()
        
        if color_by != "None" and color_by in plot_data.columns:
            # Group by color
            for category in plot_data[color_by].unique():
                if pd.notna(category):
                    category_data = plot_data[plot_data[color_by] == category]
                    
                    # Size mapping
                    if size_by != "None" and size_by in plot_data.columns:
                        sizes = category_data[size_by] / category_data[size_by].max() * 20 + 5
                    else:
                        sizes = 8
                    
                    fig.add_trace(go.Scatter3d(
                        x=category_data[x_col],
                        y=category_data[y_col],
                        z=category_data[z_col],
                        mode='markers',
                        marker=dict(
                            size=sizes,
                            opacity=0.7
                        ),
                        name=str(category),
                        text=category_data.index,
                        hovertemplate=f'<b>{x_col}</b>: %{{x}}<br>' +
                                    f'<b>{y_col}</b>: %{{y}}<br>' +
                                    f'<b>{z_col}</b>: %{{z}}<br>' +
                                    '<extra></extra>'
                    ))
        else:
            # No color grouping
            if size_by != "None" and size_by in plot_data.columns:
                sizes = plot_data[size_by] / plot_data[size_by].max() * 20 + 5
            else:
                sizes = 8
                
            fig.add_trace(go.Scatter3d(
                x=plot_data[x_col],
                y=plot_data[y_col],
                z=plot_data[z_col],
                mode='markers',
                marker=dict(
                    size=sizes,
                    opacity=0.7,
                    color=plot_data[z_col],
                    colorscale='Viridis'
                ),
                text=plot_data.index,
                hovertemplate=f'<b>{x_col}</b>: %{{x}}<br>' +
                            f'<b>{y_col}</b>: %{{y}}<br>' +
                            f'<b>{z_col}</b>: %{{z}}<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
    elif plot_type == "3D Surface Plot":
        # Create grid for surface
        x_range = np.linspace(plot_data[x_col].min(), plot_data[x_col].max(), 20)
        y_range = np.linspace(plot_data[y_col].min(), plot_data[y_col].max(), 20)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Interpolate Z values
        from scipy.interpolate import griddata
        points = plot_data[[x_col, y_col]].values
        values = plot_data[y_col].values  # Using y_col as Z for demonstration
        
        Z = griddata(points, values, (X, Y), method='cubic', fill_value=0)
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            opacity=0.8
        )])
        
        fig.update_layout(
            title=f"3D Surface Plot: {x_col} vs {y_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title="Interpolated Values"
            )
        )
        
    elif plot_type == "3D Line Plot":
        fig = go.Figure()
        
        if color_by != "None" and color_by in plot_data.columns:
            for category in plot_data[color_by].unique():
                if pd.notna(category):
                    category_data = plot_data[plot_data[color_by] == category].sort_values(x_col)
                    fig.add_trace(go.Scatter3d(
                        x=category_data[x_col],
                        y=category_data[y_col],
                        z=category_data[z_col],
                        mode='lines+markers',
                        name=str(category),
                        line=dict(width=3)
                    ))
        else:
            sorted_data = plot_data.sort_values(x_col)
            fig.add_trace(go.Scatter3d(
                x=sorted_data[x_col],
                y=sorted_data[y_col],
                z=sorted_data[z_col],
                mode='lines+markers',
                line=dict(width=3, color='blue')
            ))
        
        fig.update_layout(
            title=f"3D Line Plot: {x_col} vs {y_col} vs {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
    elif plot_type == "3D Bar Chart":
        # Create 3D bar chart using Scatter3d with custom markers
        fig = go.Figure()
        
        # Create bars by adding multiple traces for each bar
        unique_x = plot_data[x_col].unique()[:20]  # Limit to first 20 for performance
        unique_y = plot_data[y_col].unique()[:20]
        
        for i, x_val in enumerate(unique_x):
            for j, y_val in enumerate(unique_y):
                subset = plot_data[(plot_data[x_col] == x_val) & (plot_data[y_col] == y_val)]
                if len(subset) > 0:
                    z_val = subset[z_col].mean()  # Average z value for this x,y combination
                    
                    # Create a bar by drawing lines from base to top
                    fig.add_trace(go.Scatter3d(
                        x=[x_val, x_val],
                        y=[y_val, y_val],
                        z=[0, z_val],
                        mode='lines',
                        line=dict(
                            color=z_val,
                            colorscale='Viridis',
                            width=10
                        ),
                        showlegend=False,
                        hovertemplate=f'<b>{x_col}</b>: {x_val}<br>' +
                                    f'<b>{y_col}</b>: {y_val}<br>' +
                                    f'<b>{z_col}</b>: {z_val:.2f}<br>' +
                                    '<extra></extra>'
                    ))
        
        # Add scatter points at the top of bars
        fig.add_trace(go.Scatter3d(
            x=plot_data[x_col],
            y=plot_data[y_col],
            z=plot_data[z_col],
            mode='markers',
            marker=dict(
                size=5,
                color=plot_data[z_col],
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Data Points',
            hovertemplate=f'<b>{x_col}</b>: %{{x}}<br>' +
                        f'<b>{y_col}</b>: %{{y}}<br>' +
                        f'<b>{z_col}</b>: %{{z}}<br>' +
                        '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"3D Bar Chart: {x_col} vs {y_col} vs {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
    elif plot_type == "3D Mesh Plot":
        # Create 3D mesh using Delaunay triangulation
        from scipy.spatial import Delaunay
        
        # Sample data for mesh (too many points can cause performance issues)
        sample_data = plot_data.sample(n=min(100, len(plot_data)), random_state=42)
        
        # Create triangulation
        points = sample_data[[x_col, y_col]].values
        tri = Delaunay(points)
        
        # Create mesh
        fig = go.Figure(data=[go.Mesh3d(
            x=sample_data[x_col],
            y=sample_data[y_col],
            z=sample_data[y_col],  # Using y_col as z for mesh
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            color='lightblue',
            opacity=0.5,
            name='Mesh'
        )])
        
        # Add scatter points
        fig.add_trace(go.Scatter3d(
            x=sample_data[x_col],
            y=sample_data[y_col],
            z=sample_data[y_col],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='Data Points'
        ))
        
        fig.update_layout(
            title=f"3D Mesh Plot: {x_col} vs {y_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=y_col
            )
        )
        
    elif plot_type == "3D Contour Plot":
        # Create 3D contour
        fig = go.Figure(data=[go.Surface(
            x=plot_data[x_col].values.reshape(-1, 1),
            y=plot_data[y_col].values.reshape(1, -1),
            z=plot_data[y_col].values.reshape(-1, 1),
            colorscale='Viridis',
            contours=dict(
                x=dict(show=True, start=plot_data[x_col].min(), end=plot_data[x_col].max(), size=0.5),
                y=dict(show=True, start=plot_data[y_col].min(), end=plot_data[y_col].max(), size=0.5),
                z=dict(show=True, start=plot_data[y_col].min(), end=plot_data[y_col].max(), size=0.5)
            )
        )])
        
        fig.update_layout(
            title=f"3D Contour Plot: {x_col} vs {y_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title="Contour Values"
            )
        )
        
    elif plot_type == "3D Volume Plot":
        # Create 3D volume plot using scatter3d with size and color mapping
        fig = go.Figure()
        
        # Create volume visualization using scatter points with varying size and color
        fig.add_trace(go.Scatter3d(
            x=plot_data[x_col],
            y=plot_data[y_col],
            z=plot_data[z_col],
            mode='markers',
            marker=dict(
                size=plot_data[value_col] / plot_data[value_col].max() * 20 + 5,
                color=plot_data[value_col],
                colorscale='Viridis',
                opacity=0.6,
                showscale=True,
                colorbar=dict(title=value_col)
            ),
            text=plot_data[value_col],
            hovertemplate=f'<b>{x_col}</b>: %{{x}}<br>' +
                        f'<b>{y_col}</b>: %{{y}}<br>' +
                        f'<b>{z_col}</b>: %{{z}}<br>' +
                        f'<b>{value_col}</b>: %{{text}}<br>' +
                        '<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"3D Volume Plot: {x_col} vs {y_col} vs {z_col} (Size/Color: {value_col})",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
    elif plot_type == "3D PCA Visualization":
        # Perform PCA
        numeric_data = plot_data[numeric_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_data)
        
        # Create PCA plot
        fig = go.Figure()
        
        if color_by != "None" and color_by in plot_data.columns:
            for category in plot_data[color_by].unique():
                if pd.notna(category):
                    mask = plot_data[color_by] == category
                    category_pca = pca_result[mask]
                    fig.add_trace(go.Scatter3d(
                        x=category_pca[:, 0],
                        y=category_pca[:, 1],
                        z=category_pca[:, 2],
                        mode='markers',
                        name=str(category),
                        marker=dict(size=5, opacity=0.7)
                    ))
        else:
            fig.add_trace(go.Scatter3d(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                z=pca_result[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    opacity=0.7,
                    color=pca_result[:, 2],
                    colorscale='Viridis'
                )
            ))
        
        # Add explained variance info
        explained_var = pca.explained_variance_ratio_
        
        fig.update_layout(
            title=f"3D PCA Visualization (Explained Variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}, PC3={explained_var[2]:.2%})",
            scene=dict(
                xaxis_title=f"PC1 ({explained_var[0]:.1%})",
                yaxis_title=f"PC2 ({explained_var[1]:.1%})",
                zaxis_title=f"PC3 ({explained_var[2]:.1%})"
            )
        )
        
    elif plot_type == "3D Clustering":
        # Perform K-means clustering
        numeric_data = plot_data[numeric_cols].dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Determine optimal number of clusters
        n_clusters = min(8, max(2, len(plot_data) // 50))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Use first 3 principal components for visualization
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled_data)
        
        fig = go.Figure()
        
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_pca = pca_result[cluster_mask]
            
            fig.add_trace(go.Scatter3d(
                x=cluster_pca[:, 0],
                y=cluster_pca[:, 1],
                z=cluster_pca[:, 2],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=5, opacity=0.7)
            ))
        
        fig.update_layout(
            title=f"3D Clustering Visualization ({n_clusters} clusters)",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            )
        )
        
    elif plot_type == "3D Density Plot":
        # Create 3D density plot
        fig = go.Figure(data=[go.Histogram2dContour(
            x=plot_data[x_col],
            y=plot_data[y_col],
            z=plot_data[z_col],
            colorscale='Viridis',
            opacity=0.8
        )])
        
        fig.update_layout(
            title=f"3D Density Plot: {x_col} vs {y_col} vs {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
    elif plot_type == "3D Heatmap":
        # Create 3D heatmap
        fig = go.Figure(data=[go.Heatmap(
            x=plot_data[x_col],
            y=plot_data[y_col],
            z=plot_data[z_col],
            colorscale='Viridis'
        )])
        
        fig.update_layout(
            title=f"3D Heatmap: {x_col} vs {y_col} vs {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
    elif plot_type == "3D Parallel Coordinates":
        if not selected_cols:
            st.error("Please select at least one column for parallel coordinates.")
            return None
            
        # Normalize data for parallel coordinates
        normalized_data = plot_data[selected_cols].copy()
        for col in selected_cols:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val != min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=normalized_data[selected_cols[0]] if len(selected_cols) > 0 else 'blue'),
            dimensions=list([dict(label=col, values=normalized_data[col]) for col in selected_cols])
        ))
        
        fig.update_layout(
            title=f"3D Parallel Coordinates: {', '.join(selected_cols)}"
        )
    
    return fig

# Generate the plot
if st.button("🚀 Generate 3D Plot", type="primary"):
    with st.spinner("Creating 3D visualization..."):
        try:
            # Prepare data for plotting
            if use_sample and len(data) > sample_size:
                plot_data = data.sample(n=sample_size, random_state=42)
            else:
                plot_data = data.copy()
            
            if plot_type == "3D Parallel Coordinates":
                fig = create_3d_plot(plot_type, data, None, None, None, None, None, use_sample, sample_size)
            else:
                fig = create_3d_plot(plot_type, data, x_col, y_col, z_col, color_by, size_by, use_sample, sample_size)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Add plot information
                st.subheader("📊 Plot Information")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Data Points", f"{len(plot_data):,}")
                
                with col2:
                    if plot_type in ["3D PCA Visualization", "3D Clustering"]:
                        st.metric("Features Used", len(numeric_cols))
                    else:
                        st.metric("X-axis", x_col)
                
                with col3:
                    if plot_type == "3D PCA Visualization":
                        pca = PCA(n_components=3)
                        numeric_data = plot_data[numeric_cols].dropna()
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(numeric_data)
                        pca.fit(scaled_data)
                        explained_var = pca.explained_variance_ratio_
                        st.metric("Total Variance Explained", f"{sum(explained_var):.1%}")
                    elif plot_type == "3D Clustering":
                        n_clusters = min(8, max(2, len(plot_data) // 50))
                        st.metric("Number of Clusters", n_clusters)
                    else:
                        st.metric("Z-axis", z_col)
                
                # Export options
                st.subheader("💾 Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📥 Download as HTML"):
                        html_string = fig.to_html(include_plotlyjs='cdn')
                        st.download_button(
                            label="💾 Download HTML",
                            data=html_string,
                            file_name=f"3d_plot_{plot_type.lower().replace(' ', '_')}.html",
                            mime="text/html"
                        )
                
                with col2:
                    if st.button("📊 Download as PNG"):
                        img_bytes = fig.to_image(format="png", width=1200, height=800)
                        st.download_button(
                            label="💾 Download PNG",
                            data=img_bytes,
                            file_name=f"3d_plot_{plot_type.lower().replace(' ', '_')}.png",
                            mime="image/png"
                        )
                
        except Exception as e:
            st.error(f"❌ Error creating 3D plot: {str(e)}")
            st.info("💡 Try selecting different columns or reducing the sample size.")

# Additional information
st.subheader("ℹ️ About 3D Visualizations")

with st.expander("📚 3D Plot Types Guide"):
    st.markdown("""
    **Available 3D Plot Types:**
    
    🔵 **3D Scatter Plot**: Shows relationships between three numeric variables
    - Best for: Identifying patterns and clusters in 3D space
    - Use when: You want to see how three variables relate to each other
    
    🌊 **3D Surface Plot**: Displays a surface over a 2D grid
    - Best for: Showing continuous relationships between two variables
    - Use when: You want to visualize mathematical functions or interpolated data
    
    📈 **3D Line Plot**: Connects points in 3D space with lines
    - Best for: Showing trends or paths in 3D space
    - Use when: You have sequential data or want to show trajectories
    
    📊 **3D Bar Chart**: Shows bars in 3D space
    - Best for: Comparing values across three dimensions
    - Use when: You want to emphasize individual data points
    
    🕸️ **3D Mesh Plot**: Creates a mesh surface connecting points
    - Best for: Showing surface topology
    - Use when: You want to visualize 3D shapes or surfaces
    
    🗺️ **3D Contour Plot**: Shows contour lines in 3D
    - Best for: Displaying level curves and gradients
    - Use when: You want to show how a value changes across a surface
    
    📦 **3D Volume Plot**: Shows volumetric data
    - Best for: Visualizing density or concentration
    - Use when: You have 4D data (x, y, z, value)
    
    🧠 **3D PCA Visualization**: Principal Component Analysis in 3D
    - Best for: Dimensionality reduction and pattern discovery
    - Use when: You have many features and want to see main patterns
    
    🎯 **3D Clustering**: Shows clusters in 3D space
    - Best for: Identifying groups in your data
    - Use when: You want to discover natural groupings
    
    📊 **3D Density Plot**: Shows data density in 3D
    - Best for: Visualizing data concentration
    - Use when: You want to see where most of your data lies
    
    🔥 **3D Heatmap**: Color-coded 3D representation
    - Best for: Showing intensity across three dimensions
    - Use when: You want to highlight high/low value regions
    
    📏 **3D Parallel Coordinates**: Shows multiple dimensions simultaneously
    - Best for: Comparing many variables at once
    - Use when: You have many features and want to see patterns across all of them
    """)

with st.expander("💡 Tips for Better 3D Visualizations"):
    st.markdown("""
    **Best Practices:**
    
    🎯 **Choose the Right Plot Type**: Match your data and analysis goals
    - Scatter plots for relationships
    - Surface plots for continuous functions
    - PCA for dimensionality reduction
    - Clustering for pattern discovery
    
    🎨 **Use Color and Size Wisely**: 
    - Color by categorical variables to show groups
    - Size by numeric variables to show importance
    - Use consistent color schemes
    
    📊 **Sample Large Datasets**: 
    - Use sampling for datasets with >1000 points
    - This improves performance and reduces visual clutter
    
    🔍 **Interactive Features**:
    - Rotate, zoom, and pan to explore different angles
    - Hover over points to see detailed information
    - Use the legend to toggle different groups
    
    📈 **Interpret Results Carefully**:
    - 3D plots can be misleading due to perspective
    - Always consider the actual data ranges
    - Use multiple plot types to confirm patterns
    """)

with st.expander("⚠️ Limitations and Considerations"):
    st.markdown("""
    **Important Notes:**
    
    🔍 **Visual Perception**: 3D plots can be harder to interpret than 2D plots
    - Some patterns might be hidden due to perspective
    - Consider using multiple 2D projections as well
    
    💻 **Performance**: 3D plots are computationally intensive
    - Large datasets may need sampling
    - Some plot types work better with smaller datasets
    
    📊 **Data Requirements**: 
    - Most 3D plots need at least 3 numeric columns
    - Some plots work better with specific data distributions
    - Missing data can affect plot quality
    
    🎯 **Interpretation**: 
    - Always consider the context of your data
    - 3D plots are great for exploration, but 2D plots might be better for presentation
    - Use statistical tests to confirm visual patterns
    """)

# Footer
st.markdown("---")
st.markdown("**3D Visualizations** - Explore your data in three dimensions with interactive plots")
