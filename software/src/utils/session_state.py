import streamlit as st
def initialize_session_state():
    """Initialize all session state variables"""
    if 'work_dir' not in st.session_state or st.session_state.work_dir is None:
        st.session_state.work_dir = None
    if 'model_generated' not in st.session_state:
        st.session_state.model_generated = False
    if 'meshing_done' not in st.session_state:
        st.session_state.meshing_done = False
    if 'contact_generation_done' not in st.session_state:
        st.session_state.contact_generation_done = False
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'visualization_done' not in st.session_state:
        st.session_state.visualization_done = False