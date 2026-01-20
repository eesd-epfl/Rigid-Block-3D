import json
from pathlib import Path
import streamlit as st
from analysis import analysis


DEFAULT_LOAD_CONFIG = {
    "control_dof": 2,
    "step_size": 0.02,
    "max_disp": 0.2,
    "unit_comment": "mm",
    "dead_load_dim": 0,
    "dead_load_value": -1.25,
    "dead_load_value_comment": "0 kN",
    "write_freq": 1,
    "beamID": 0,
}


def _write_uploaded_file(uploaded_file, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(uploaded_file.getbuffer())


def render(refresh_token: int = 0):
    #st.header("Run Analysis")

    st.subheader("Model input")

    contact_ready = st.session_state.get("contact_generation_done", False)

    # ----------------------------
    # Model selection
    # ----------------------------
    if contact_ready:
        st.success("✅ Contact points are ready for analysis")
        work_dir = Path(st.session_state.work_dir)
        model_dir = work_dir
        #model_dir = Path(st.text_input("Model directory (CSV folder)", value=str(default_model_dir)))

        # Light check
        if not (model_dir / "point.csv").exists() or not (model_dir / "element.csv").exists():
            st.warning("point.csv or element.csv not found in the selected model directory.")
    else:
        st.info("Contact points are not generated. You can upload a model manually (point.csv + element.csv).")

        up_point = st.file_uploader("Upload point.csv", type=["csv"], key="upload_point_csv")
        up_elem = st.file_uploader("Upload element.csv", type=["csv"], key="upload_element_csv")

        # Where we save uploaded model
        # Create working directory
        # import tempfile
        # work_dir = tempfile.mkdtemp()
        # st.session_state.work_dir = work_dir
        import os
        work_dir = st.session_state.get("work_dir")
        if not work_dir:
            work_dir = os.path.join(".", "workspaces", st.session_state.get("session_id", "default"))
            st.session_state.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        work_dir = Path(work_dir)
        model_dir = work_dir

        if up_point is not None and up_elem is not None:
            try:
                _write_uploaded_file(up_point, model_dir / "point.csv")
                _write_uploaded_file(up_elem, model_dir / "element.csv")
                st.success(f"✅ Saved uploaded model to: {model_dir}")
            except Exception as e:
                st.error(f"Failed to save uploaded model files: {e}")
                return
        else:
            st.warning("Please upload both point.csv and element.csv to enable analysis.")
            return  # stop here until files are provided

    # Final guard before run
    if not (model_dir / "point.csv").exists() or not (model_dir / "element.csv").exists():
        st.error("Model is incomplete: point.csv and element.csv must exist.")
        return

    # ----------------------------
    # Load config: upload OR UI edit
    # ----------------------------
    st.subheader("Load configuration")

    mode = st.radio("Provide load config", ["Edit in UI","Upload JSON"], horizontal=True)
    load_cfg = DEFAULT_LOAD_CONFIG.copy()

    if mode == "Upload JSON":
        up = st.file_uploader("Upload load_config.json", type=["json"], key="upload_load_config")
        if up is not None:
            try:
                load_cfg = json.loads(up.getvalue().decode("utf-8"))
                st.success("✅ JSON loaded")
            except Exception as e:
                st.error(f"Could not read JSON: {e}")
                return
        else:
            st.info("Upload a JSON file, or switch to 'Edit in UI'.")

    else:
        # col1 = st.columns(1)
        # with col1:
        #load_cfg["beamID"] = st.number_input("beamID", value=int(load_cfg["beamID"]), min_value=0, step=1)
        load_cfg["control_dof"] = st.number_input("Diplacement Control DOF", value=int(load_cfg["control_dof"]), min_value=0, step=1)
        load_cfg["step_size"] = st.number_input("Step size (mm)", value=float(load_cfg["step_size"]), min_value=0.0, format="%.6g")
        load_cfg["max_disp"] = st.number_input("Max displacement (mm)", value=float(load_cfg["max_disp"]), min_value=0.0, format="%.6g")
    #with col2:
        load_cfg["dead_load_dim"] = 0
        load_cfg["dead_load_value"] = st.number_input("Axial load (kN)", value=float(load_cfg["dead_load_value"]), format="%.6g")
        st.caption(f"Axial load should be negative for compression")
            

        #with st.expander("Optional fields"):
            #load_cfg["unit_comment"] = st.text_input("unit_comment", value=str(load_cfg.get("unit_comment", "mm")))
            # load_cfg["dead_load_value_comment"] = st.text_input(
            #     "dead_load_value_comment", value=str(load_cfg.get("dead_load_value_comment", "0 kN"))
            # )

    # Preview/edit as JSON text
    with st.expander("Preview / edit JSON directly"):
        edited_text = st.text_area("load_config.json", value=json.dumps(load_cfg, indent=2), height=220)
        try:
            load_cfg = json.loads(edited_text)
        except Exception as e:
            st.error(f"JSON is not valid: {e}")
            return

    

    # ----------------------------
    # Outputs + options
    # ----------------------------
    with st.expander("Options"):
        result_dir = Path(st.text_input("Result directory", value=str(work_dir / "pushover_results")))
        #save_results = st.checkbox("Save intermediate models / VTK plots", value=False)
        save_results = True
        load_cfg["write_freq"] = st.number_input("Save VTK frequency", value=int(load_cfg["write_freq"]), min_value=1, step=1)

    # with st.expander("Units (optional)"):
    length_scale = 1e3
    force_scale = 1e-3
    # Write config to disk for solver
    load_config_path = work_dir / "load_config.json"
    try:
        load_config_path.write_text(json.dumps(load_cfg, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Failed to write load config to {load_config_path}: {e}")
        return

    # ----------------------------
    # Run
    # ----------------------------
    if st.button("▶️ Run Pushover", type="primary"):
        progress = st.progress(0.0)
        status = st.empty()

        def progress_cb(frac: float, msg: str) -> None:
            progress.progress(max(0.0, min(1.0, float(frac))))
            status.write(msg)

        with st.spinner("Running pushover... This may take a while"):
            try:
                out = analysis.main(
                    work_dir=work_dir,
                    model_dir=model_dir,
                    load_config_path=load_config_path,
                    result_dir=result_dir,
                    save_results=save_results,
                    length_scale=length_scale,
                    force_scale=force_scale,
                    progress_cb=progress_cb,
                )

                st.session_state.analysis_done = True
                st.session_state.pushover_fd_path = out["fd_path"]
                st.session_state.pushover_forces = out["forces"]
                st.session_state.pushover_displacements = out["displacements"]

                st.success("✅ Pushover completed!")
                st.caption(f"Force–displacement file: {out['fd_path']}")

                import pandas as pd
                import numpy as np

                if out["forces"] and out["displacements"]:
                    disp_incr = np.array(out["displacements"], dtype=float)
                    disp_cum = np.cumsum(disp_incr)

                    df = pd.DataFrame({
                        "displacement (mm)": disp_cum,
                        "force (kN)": out["forces"],
                    })

                    st.line_chart(df, x="displacement (mm)", y="force (kN)")



            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())
