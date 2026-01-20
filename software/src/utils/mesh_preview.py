"""
Mesh preview utilities for Streamlit
"""

import streamlit as st
import plotly.graph_objects as go
import tempfile
import os
import numpy as np


def preview_meshes(mesh_files):
    """Display interactive 3D preview of uploaded meshes"""
    if not mesh_files:
        st.info("👆 Upload mesh files to preview")
        return
    
    st.subheader("Geometry Preview")
    fig = go.Figure()
    colors = ['lightgray', 'lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
    
    for i, mesh_file in enumerate(mesh_files):
        try:
            vertices, faces = load_mesh_pymeshlab(mesh_file)
            
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=colors[i % len(colors)],
                opacity=0.8,
                name=mesh_file.name,
                flatshading=True
            ))
            
            st.write(f"  ✓ {mesh_file.name}: {len(vertices)} vertices, {len(faces)} faces")
            
        except Exception as e:
            st.error(f"Error loading {mesh_file.name}: {str(e)}")
            import traceback
            with st.expander(f"Error details for {mesh_file.name}"):
                st.code(traceback.format_exc())
    
    if fig.data:
        fig.update_layout(
            scene=dict(aspectmode='data'),
            width=800,
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)


def preview_msh_file(mesh_path, color='lightgreen', name=None, show_surface=False):
    """Preview MSH file with statistics and optional surface visualization"""
    import meshio
    
    st.write(f"**{name or 'Tetrahedral Mesh'}**")
    st.info("📊 Tetrahedral mesh - showing statistics")
    
    # Read MSH file
    mesh = meshio.read(mesh_path)
    vertices = mesh.points
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Vertices", f"{len(vertices):,}")
    
    if 'tetra' in mesh.cells_dict:
        tetra = mesh.cells_dict['tetra']
        with col2:
            st.metric("Tetrahedra", f"{len(tetra):,}")
        
        # Calculate approximate surface faces
        approx_surface = int(len(tetra) * 0.4)  # Rough estimate
        with col3:
            st.metric("Est. Surface", f"~{approx_surface:,}")
        
        # Show surface visualization if requested
        if show_surface:
            st.write("**Surface Visualization:**")
            with st.spinner("Extracting surface mesh..."):
                try:
                    import plotly.graph_objects as go
                    from collections import Counter
                    import numpy as np
                    
                    # Extract surface triangles from tetrahedra
                    all_faces = []
                    for tet in tetra:
                        # Each tetrahedron has 4 triangular faces
                        all_faces.append(tuple(sorted([tet[0], tet[1], tet[2]])))
                        all_faces.append(tuple(sorted([tet[0], tet[1], tet[3]])))
                        all_faces.append(tuple(sorted([tet[0], tet[2], tet[3]])))
                        all_faces.append(tuple(sorted([tet[1], tet[2], tet[3]])))
                    
                    # Surface faces appear only once (internal faces appear twice)
                    face_counts = Counter(all_faces)
                    surface_faces = np.array([list(face) for face, count in face_counts.items() if count == 1])
                    
                    st.caption(f"Extracted {len(surface_faces):,} surface triangles")
                    
                    # Plot surface
                    if len(surface_faces) > 0:
                        fig = go.Figure(data=[
                            go.Mesh3d(
                                x=vertices[:, 0],
                                y=vertices[:, 1],
                                z=vertices[:, 2],
                                i=surface_faces[:, 0],
                                j=surface_faces[:, 1],
                                k=surface_faces[:, 2],
                                color=color,
                                opacity=0.8,
                                name=name or "Mesh Surface",
                                flatshading=True
                            )
                        ])
                        
                        fig.update_layout(
                            scene=dict(aspectmode='data'),
                            width=800,
                            height=600,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No surface faces found")
                        
                except Exception as e:
                    st.error(f"Error extracting surface: {str(e)}")
    
    elif 'triangle' in mesh.cells_dict:
        faces = mesh.cells_dict['triangle']
        with col2:
            st.metric("Triangles", f"{len(faces):,}")
        
        if show_surface:
            st.write("**Mesh Visualization:**")
            try:
                import plotly.graph_objects as go
                
                fig = go.Figure(data=[
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=color,
                        opacity=0.8,
                        name=name or "Mesh",
                        flatshading=True
                    )
                ])
                
                fig.update_layout(
                    scene=dict(aspectmode='data'),
                    width=800,
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error visualizing mesh: {str(e)}")
    
    # Show bounding box details
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    
    with st.expander("📏 Mesh Details"):
        st.write("**Bounding Box:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**X:** [{bbox_min[0]:.4f}, {bbox_max[0]:.4f}]")
            st.caption(f"Size: {bbox_size[0]:.4f} m")
        with col2:
            st.write(f"**Y:** [{bbox_min[1]:.4f}, {bbox_max[1]:.4f}]")
            st.caption(f"Size: {bbox_size[1]:.4f} m")
        with col3:
            st.write(f"**Z:** [{bbox_min[2]:.4f}, {bbox_max[2]:.4f}]")
            st.caption(f"Size: {bbox_size[2]:.4f} m")
        
        # Show volume
        volume = bbox_size[0] * bbox_size[1] * bbox_size[2]
        st.write(f"**Bounding Volume:** {volume:.6f} m³")
        
        # Show cell type information
        st.write("**Element Types:**")
        for cell_type, cells in mesh.cells_dict.items():
            st.write(f"- {cell_type}: {len(cells):,} elements")


def preview_single_mesh(mesh_path, color='lightgray', name=None, show_surface=True):
    """Display interactive 3D preview of a single mesh file from disk"""
    import plotly.graph_objects as go
    
    if name is None:
        name = os.path.basename(mesh_path)
    
    # Check file extension FIRST
    file_ext = mesh_path.split('.')[-1].lower()
    
    try:
        if file_ext == 'msh':
            # For MSH files (tetrahedral meshes), show statistics and optional surface
            preview_msh_file(mesh_path, color, name, show_surface=show_surface)
        else:
            # For surface meshes (PLY, OBJ, STL)
            preview_surface_mesh(mesh_path, color, name)
        
    except Exception as e:
        st.error(f"Error loading mesh: {str(e)}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

def preview_surface_mesh(mesh_path, color='lightgray', name=None):
    """Preview surface mesh files (PLY, OBJ, STL)"""
    import plotly.graph_objects as go
    import pymeshlab
    
    # Load with PyMeshLab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    
    # Get mesh data
    m = ms.current_mesh()
    vertices = m.vertex_matrix()
    faces = m.face_matrix()
    
    # Create figure
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=0.8,
            name=name,
            flatshading=True
        )
    ])
    
    fig.update_layout(
        scene=dict(aspectmode='data'),
        width=800,
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)



def load_mesh_pymeshlab(mesh_file):
    """Load mesh using PyMeshLab (handles PLY, OBJ, STL, etc.)"""
    import pymeshlab
    
    file_ext = mesh_file.name.split('.')[-1].lower()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
        tmp.write(mesh_file.read())
        tmp_path = tmp.name
    
    try:
        # Load with PyMeshLab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp_path)
        
        # Get current mesh
        m = ms.current_mesh()
        
        # Extract vertices and faces
        vertices = m.vertex_matrix()
        faces = m.face_matrix()
        
        return vertices, faces
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)