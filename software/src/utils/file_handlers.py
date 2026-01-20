import streamlit as st
import tempfile
import shutil
import os
from pathlib import Path

def save_uploaded_files(json_file, mesh_files):
    """
    Save uploaded files to a temporary working directory
    
    Returns:
        tuple: (work_dir, json_path, mesh_paths)
    """
    work_dir = tempfile.mkdtemp()
    
    # Save JSON
    json_path = os.path.join(work_dir, "config.json")
    with open(json_path, 'wb') as f:
        f.write(json_file.read())
    
    # Save meshes
    mesh_paths = []
    for mesh_file in mesh_files:
        mesh_path = os.path.join(work_dir, mesh_file.name)
        with open(mesh_path, 'wb') as f:
            f.write(mesh_file.read())
        mesh_paths.append(mesh_path)
    
    return work_dir, json_path, mesh_paths

def cleanup_working_directory():
    """Reset session state"""
    if st.session_state.work_dir:
        #shutil.rmtree(st.session_state.work_dir, ignore_errors=True)
        st.session_state.work_dir = None
        st.session_state.model_generated = False
        st.session_state.analysis_done = False
        st.session_state.visualization_done = False

def provide_file_download(file_path, label=None):
    """Create a download button for a file"""
    if label is None:
        label = f"📥 Download {Path(file_path).name}"
    
    with open(file_path, 'rb') as f:
        st.download_button(
            label=label,
            data=f,
            file_name=Path(file_path).name,
            mime="application/octet-stream"
        )