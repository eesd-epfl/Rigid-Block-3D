import streamlit as st
import os
from ui import tab_model, tab_meshing, tab_analysis, tab_visualization, tab_contact
from utils.session_state import initialize_session_state

st.set_page_config(page_title="Stone Masonry Analyzer", layout="wide")
st.title("🏛️ Stone Masonry Wall Analyzer")

# Initialize session state
initialize_session_state()

# ========================================
# SIDEBAR - Settings
# ========================================

# st.sidebar.header("⚙️ Settings")

# with st.sidebar.expander("🔧 fTetWild Configuration", expanded=False):
#     st.write("**Specify fTetWild Binary Path**")
    
#     # Default path hint
#     st.caption("Example: /home/user/fTetWild/build/FloatTetwild_bin")
    
#     # Text input for path
#     ftetwild_path = st.text_input(
#         "fTetWild Binary Path",
#         value=st.session_state.get('ftetwild_path', ''),
#         key='ftetwild_path_input',
#         placeholder="/path/to/FloatTetwild_bin",
#         label_visibility="collapsed"
#     )
    
#     # Save button
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("💾 Save", use_container_width=True):
#             if ftetwild_path and os.path.exists(ftetwild_path):
#                 st.session_state.ftetwild_path = ftetwild_path
#                 os.environ['FTETWILD_BIN'] = ftetwild_path
#                 st.success("✓ Path saved!")
#                 st.rerun()
#             elif ftetwild_path:
#                 st.error("Path does not exist!")
#             else:
#                 st.warning("Please enter a path")
    
#     with col2:
#         if st.button("🔍 Test", use_container_width=True):
#             if ftetwild_path and os.path.exists(ftetwild_path):
#                 # Test if it's executable
#                 import subprocess
#                 try:
#                     result = subprocess.run(
#                         [ftetwild_path, '--help'],
#                         capture_output=True,
#                         timeout=5
#                     )
#                     st.success("✓ Binary works!")
#                 except Exception as e:
#                     st.error(f"Binary test failed: {e}")
#             else:
#                 st.error("Path does not exist!")
    
#     # Show current status
#     st.divider()
    
#     if st.session_state.get('ftetwild_path'):
#         st.success(f"✓ Configured: {st.session_state.ftetwild_path}")
#     else:
#         st.info("ℹ️ No path configured - auto-detection will be used")
    
#     # Clear button
#     if st.session_state.get('ftetwild_path'):
#         if st.button("🗑️ Clear Configuration", use_container_width=True):
#             st.session_state.ftetwild_path = ''
#             if 'FTETWILD_BIN' in os.environ:
#                 del os.environ['FTETWILD_BIN']
#             st.rerun()

# ========================================
# Main Tabs
# ========================================

# tabs = st.tabs([
#     "1️⃣ Model Generation", 
#     "2️⃣ Meshing",
#     "3️⃣ Contact Generation",
#     "4️⃣ Run Analysis", 
#     "5️⃣ Visualization"
# ])

# # Render each tab
# with tabs[0]:
#     tab_model.render()

# with tabs[1]:
#     tab_meshing.render()

# with tabs[2]:
#     tab_contact.render()

# with tabs[3]:
#     tab_analysis.render()

# with tabs[4]:
#     tab_visualization.render()


WORKSPACES_ROOT = os.path.join(os.getcwd(), "workspaces")
os.makedirs(WORKSPACES_ROOT, exist_ok=True)

#st.sidebar.divider()
st.sidebar.header("📁 Workspace")

# List existing workspaces (folders only)
existing = sorted([
    d for d in os.listdir(WORKSPACES_ROOT)
    if os.path.isdir(os.path.join(WORKSPACES_ROOT, d))
])

# UI: choose existing
selected = st.sidebar.selectbox(
    "Select workspace",
    options=["(new)"] + existing,
    index=0,
    key="workspace_select",
)

# UI: create new
new_name = None
if selected == "(new)":
    new_name = st.sidebar.text_input(
        "New workspace name",
        value="run_001",
        key="new_workspace_name",
    )

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("✅ OK", use_container_width=True):
        if selected == "(new)":
            name = (new_name or "").strip()
            if not name:
                st.sidebar.error("Please enter a workspace name.")
            else:
                work_dir = os.path.join(WORKSPACES_ROOT, name)
                os.makedirs(work_dir, exist_ok=True)
                st.session_state.work_dir = work_dir
                st.sidebar.success("Workspace selected.")
                st.rerun()
        else:
            work_dir = os.path.join(WORKSPACES_ROOT, selected)
            st.session_state.work_dir = work_dir
            st.sidebar.success("Workspace selected.")
            st.rerun()

with col2:
    if st.sidebar.button("➕ Refresh list", use_container_width=True):
        st.rerun()

# Show current workspace path
work_dir = st.session_state.get("work_dir")
if work_dir:
    st.sidebar.caption("Current workspace:")
    st.sidebar.code(work_dir)
else:
    st.sidebar.info("No workspace selected yet.")

if not st.session_state.get("work_dir"):
    st.info("👈 Please select or create a workspace in the sidebar to continue.")
    st.stop()


# # option to empty workspace
# work_dir = st.session_state.get("work_dir")
# import shutil
# if work_dir and os.path.isdir(work_dir):
#     st.sidebar.divider()
#     #st.sidebar.header("🧹 Workspace Maintenance")

#     confirm_clear = st.sidebar.checkbox("I want to delete all files in workspace")

#     if confirm_clear and st.sidebar.button("🗑️ Empty workspace", use_container_width=True):
#         for name in os.listdir(work_dir):
#             path = os.path.join(work_dir, name)
#             try:
#                 if os.path.isdir(path):
#                     shutil.rmtree(path)
#                 else:
#                     os.remove(path)
            
#             except Exception as e:
#                 st.sidebar.error(f"Failed to remove {name}: {e}")

#         st.sidebar.success("Workspace emptied.")
#         st.rerun()

# st.sidebar.divider()
# st.sidebar.header("📁 Workspace")

# work_dir = st.session_state.get("work_dir")

# if work_dir:
#     st.sidebar.success("Workspace is set")
#     st.sidebar.code(work_dir)
# else:
#     st.sidebar.warning("Workspace not set yet")
#     st.sidebar.caption("Generate a model (Tab 1) to create a workspace.")
tab_labels = [
    "1️⃣ Model Generation",
    "2️⃣ Meshing",
    "3️⃣ Contact Generation",
    "4️⃣ Run Analysis",
    "5️⃣ Visualization",
]
st.session_state.setdefault("active_tab", tab_labels[0])
st.session_state.setdefault("tab_refresh_tokens", {t: 0 for t in tab_labels})

st.sidebar.divider()
st.sidebar.header("🧭 Navigation")

active_tab = st.sidebar.radio(
    "Go to",
    tab_labels,
    key="active_tab",     # radio owns this key
)
st.session_state.setdefault("tab_refresh_tokens", {t: 0 for t in tab_labels})
if st.sidebar.button("🔄 Refresh current tab", use_container_width=True):
    # bump only the current tab token
    st.session_state.tab_refresh_tokens[active_tab] += 1
    st.rerun()

refresh_token = st.session_state.tab_refresh_tokens[active_tab]
# ----------------------------------------
# Main area: render ONLY active tab
# ----------------------------------------
st.header(active_tab)

if active_tab == "1️⃣ Model Generation":
    tab_model.render(refresh_token=refresh_token)

elif active_tab == "2️⃣ Meshing":
    tab_meshing.render(refresh_token=refresh_token)

elif active_tab == "3️⃣ Contact Generation":
    tab_contact.render(refresh_token=refresh_token)

elif active_tab == "4️⃣ Run Analysis":
    tab_analysis.render(refresh_token=refresh_token)

elif active_tab == "5️⃣ Visualization":
    tab_visualization.render(refresh_token=refresh_token)

# Sidebar cleanup
st.sidebar.divider()
st.sidebar.header("🧹 Restart")
if st.session_state.work_dir and st.sidebar.button("🗑️ Clear All Data"):
    from utils.file_handlers import cleanup_working_directory
    cleanup_working_directory()
    st.rerun()