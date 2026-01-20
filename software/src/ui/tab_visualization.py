import os
import re
import glob
import streamlit as st

from utils.file_handlers import provide_file_download

try:
    import pyvista as pv
    from stpyvista import stpyvista
    _VTU_PREVIEW_AVAILABLE = True
except Exception:
    _VTU_PREVIEW_AVAILABLE = False


@st.cache_data(show_spinner=False)
def _read_vtu_cached(vtu_path: str):
    return pv.read(vtu_path)

def _resolve_vtu_path(base_path_no_ext: str) -> str:
    candidates = [
        base_path_no_ext,
        base_path_no_ext + ".vtu",
        base_path_no_ext + ".vtu.gz",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    hits = glob.glob(base_path_no_ext + ".*")
    return hits[0] if hits else ""


def _list_iterations(result_dir: str):
    pattern = os.path.join(result_dir, "associative_elastic_iter*")
    files = glob.glob(pattern)

    iters = set()
    rx = re.compile(r"associative_elastic_iter(\d+)")
    for f in files:
        m = rx.search(os.path.basename(f))
        if m:
            iters.add(int(m.group(1)))
    return sorted(iters)


def add_vtu_preview(vtu_path: str, refresh_token: int):
    """
    refresh_token is used to force a re-render key change.
    """
    if not _VTU_PREVIEW_AVAILABLE:
        st.info("VTU preview requires: pyvista, vtk, stpyvista")
        return

    if not vtu_path or not os.path.exists(vtu_path):
        st.error(f"VTU file not found: {vtu_path}")
        return

    mesh = _read_vtu_cached(vtu_path)

    # Scalar selection
    point_keys = list(mesh.point_data.keys()) if hasattr(mesh, "point_data") else []
    cell_keys = list(mesh.cell_data.keys()) if hasattr(mesh, "cell_data") else []
    scalar_candidates = point_keys + cell_keys

    #st.subheader("VTU Preview")

    # Persist preview settings
    st.session_state.setdefault("viz_color_array", "(none)")
    st.session_state.setdefault("viz_color_mode", "Magnitude")
    #st.session_state.setdefault("viz_show_edges", True)
    st.session_state.setdefault("viz_opacity", 1.0)
    st.session_state.setdefault("viz_show_colorbar", True)
    st.session_state.setdefault("viz_show_axes", True)

    if scalar_candidates:
        st.selectbox(
            "Color by (array)",
            ["(none)"] + scalar_candidates,
            key="viz_color_array",
        )
        st.selectbox(
            "Color mode",
            ["Magnitude", "X", "Y", "Z"],
            key="viz_color_mode",
        )
    else:
        st.caption("No data arrays found in this VTU (nothing to color by).")

    colB, colC = st.columns(2)
    with colB:
        st.checkbox("Show color bar", key="viz_show_colorbar")
    with colC:
        st.checkbox("Show XYZ axes", key="viz_show_axes")

    st.slider("Opacity", 0.05, 1.0, key="viz_opacity", step=0.05)

    # Decide scalars to use (vector-aware)
    scalars_to_use = None
    array_name = st.session_state.viz_color_array

    if array_name != "(none)":
        arr = None
        if array_name in mesh.point_data:
            arr = mesh.point_data[array_name]
        elif array_name in mesh.cell_data:
            arr = mesh.cell_data[array_name]

        if arr is not None:
            mode = st.session_state.viz_color_mode
            if hasattr(arr, "ndim") and arr.ndim == 2 and arr.shape[1] == 3:
                suffix = {"Magnitude": "mag", "X": "x", "Y": "y", "Z": "z"}[mode]
                derived_name = f"{array_name}_{suffix}"

                # Avoid re-adding if already exists
                if derived_name not in mesh.point_data and derived_name not in mesh.cell_data:
                    if mode == "Magnitude":
                        mesh[derived_name] = (arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2) ** 0.5
                    elif mode == "X":
                        mesh[derived_name] = arr[:, 0]
                    elif mode == "Y":
                        mesh[derived_name] = arr[:, 1]
                    elif mode == "Z":
                        mesh[derived_name] = arr[:, 2]

                scalars_to_use = derived_name
            else:
                scalars_to_use = array_name

    plotter = pv.Plotter()

    actor = plotter.add_mesh(
    mesh,
    scalars=scalars_to_use,
    #show_edges=st.session_state.viz_show_edges,
    opacity=float(st.session_state.viz_opacity),
    scalar_bar_args={"title": scalars_to_use}
    if (st.session_state.viz_show_colorbar and scalars_to_use)
    else None,
    )

    # XYZ axes / orientation widget
    if st.session_state.viz_show_axes:
        plotter.show_axes()                 # in-scene axes
        plotter.add_orientation_widget(actor)  # orientation widget

    plotter.view_isometric()

    # Key includes refresh token so the viewer fully re-renders on refresh
    stpyvista(plotter, key=f"vtu_preview_{os.path.basename(vtu_path)}_{refresh_token}")


def render(refresh_token: int = 0):
    #st.header("Visualization")

    if not st.session_state.get("analysis_done", False):
        st.warning("⚠️ Please run the analysis first (Tab 2)")
        return

    work_dir = st.session_state.get("work_dir")
    if not work_dir:
        st.error("work_dir is not set.")
        return

    # ----- Initialize session defaults ONCE -----
    if "viz_result_dir" not in st.session_state:
        # prefer pushover_result_dir if set, else default under work_dir
        st.session_state.viz_result_dir = st.session_state.get(
            "pushover_result_dir",
            os.path.join(work_dir, "pushover_results"),
        )

    st.success("✅ Analysis results available")

    # Bind text_input to session state (no passing dynamic 'value' each run)
    st.text_input("Result directory", key="viz_result_dir")
    result_dir = st.session_state.viz_result_dir

    if not os.path.isdir(result_dir):
        st.warning("Result directory not found. Please enter the correct folder.")
        return

    iters = _list_iterations(result_dir)
    if not iters:
        st.warning("No files found matching associative_elastic_iter* in this folder.")
        return

    # iteration selection: persist and clamp if list changes
    if "viz_iter" not in st.session_state:
        st.session_state.viz_iter = iters[-1]

    # If previously selected iter is no longer available, snap to latest
    if st.session_state.viz_iter not in iters:
        st.session_state.viz_iter = iters[-1]

    idx = iters.index(st.session_state.viz_iter)

    st.subheader("VTU Preview")
    st.selectbox("Iteration i", iters, index=idx, key="viz_iter")

    i = st.session_state.viz_iter
    base_no_ext = os.path.join(result_dir, f"associative_elastic_iter{i}")
    vtu_path = _resolve_vtu_path(base_no_ext)

    if not vtu_path:
        st.error(f"Could not resolve VTU for iteration {i} (looked for .vtu/.vtu.gz).")
        return

    st.caption(f"VTU file: {vtu_path}")

    provide_file_download(vtu_path, "📥 Download VTU File")

    
    st.session_state.setdefault("viz_refresh_token", 0)
    add_vtu_preview(vtu_path, st.session_state.viz_refresh_token)
    # Refresh button: clear cache + bump token
    if st.button("🔄 Refresh Preview", key="viz_refresh_btn"):
        _read_vtu_cached.clear()  # clear cached VTU reads
        st.session_state.viz_refresh_token += 1

