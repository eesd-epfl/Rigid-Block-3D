# analysis/analysis.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Tuple, List

from Kinematics import *
from .util_pushover import (
    convert_model_unit,
    initialize_contact_force,
    cal_gap_3d_elastic,
    adjust_ft_c,
    write_intermediate_model,
    _update_contp_force_3d,
    _update_elem_disp_3d,
    _displace_model_3d,
    compute_fracture_distance,
    recalculate_elasticity,
    compute_fracture_energy
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Load config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _req(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing '{key}' in load config JSON.")
    return cfg[key]

def _find_beam(elems):
    for key, value in elems.items():
        if value.type == "beam":
            return value.id
    return None

def main(
    work_dir: str | Path,
    model_dir: str | Path,
    load_config_path: str | Path,
    result_dir: str | Path,
    *,
    save_results: bool = False,
    length_scale: float = 1e3,   # m -> mm by default
    force_scale: float = 1e-3,   # N -> kN by default
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Runs 3D displacement-controlled pushover.

    Returns dict with:
      - fd_path: Path to force_displacement.txt
      - forces: list[float]
      - displacements: list[float]
      - steps_completed: int
    """
    work_dir = Path(work_dir)
    model_dir = Path(model_dir)
    load_config_path = Path(load_config_path)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Load config + model
    # -----------------------------
    load_protocol = _load_json(load_config_path)

    set_dimension(3)
    model = Model()
    model.from_csv(str(model_dir))

    # -----------------------------
    # Preprocessing
    # -----------------------------
    #model.contps = recalculate_elasticity(model)
    #model = compute_fracture_energy(model, GF2_equ = "2")
    model = compute_fracture_distance(model)

    # Units: (default) N,m -> N,mm -> kN,mm
    model = convert_model_unit(model, force_convert=1.0, length_convert=float(length_scale))
    model = convert_model_unit(model, force_convert=float(force_scale), length_convert=1.0)

    # Apply dead/axial load
    #beam_id = int(_req(load_protocol, "beamID"))
    beam_id = _find_beam(model.elems)
    dl_dir = int(_req(load_protocol, "dead_load_dim"))
    dl_val = float(_req(load_protocol, "dead_load_value"))
    model.elems[beam_id].dl[dl_dir] = dl_val

    # Initialize contact forces
    initialize_contact_force(model.contps)

    # Pushover parameters
    step_size = float(_req(load_protocol, "step_size"))
    max_disp = float(_req(load_protocol, "max_disp"))
    max_iteration = int(max_disp / step_size)

    write_freq = int(_req(load_protocol, "write_freq"))
    control_dof = int(_req(load_protocol, "control_dof"))

    # Output file
    fd_path = result_dir / "force_displacement.txt"
    with fd_path.open("w", encoding="utf-8") as f:
        f.write("force, displacement\n")

    forces: List[float] = [0]
    disps: List[float] = [0]

    def _progress(i: int, msg: str) -> None:
        if progress_cb is None:
            return
        total = max(1, max_iteration)
        progress_cb(min(1.0, i / total), msg)

    # -----------------------------
    # Step 0: initial equilibrium
    # -----------------------------
    _progress(0, "Solving initial equilibrium (step 0)...")
    A_matrix = cal_A_global_3d(model.elems, model.contps, sparse=True)
    sol0 = solve_elastic_finitefc_associative_3d(
        model.elems, model.contps, Aglobal=A_matrix, thickness_dict=None
    )
    # Update state at step 0
    if "contact_forces" in sol0:
        _update_contp_force_3d(model.contps, sol0["contact_forces"])
    if "displacements" in sol0:
        _update_elem_disp_3d(model.contps, model.elems, sol0["displacements"])

    if not sol0.get("convergence", False):
        if save_results:
            write_intermediate_model(model, str(result_dir), 0)
            viewer = VtkRenderer(model.elems, model.contps)
            viewer.plot_displaced_points(
                factor=0, filename=str(result_dir / "associative_elastic_iter0")
            )
        raise RuntimeError("Non-convergence at step 0 (initial equilibrium).")

    
    _displace_model_3d(model.elems, model.contps)

    # -----------------------------
    # Pushover loop
    # -----------------------------
    for i in range(1, max_iteration + 1):
        _progress(i, f"Running pushover step {i}/{max_iteration}...")

        cal_gap_3d_elastic(model.contps, model.elems)
        adjust_ft_c(model.contps)

        A_matrix = cal_A_global_3d(model.elems, model.contps, sparse=True)
        sol = solve_elastic_finitefc_associative_3d_disp_control(
            model.elems,
            model.contps,
            Aglobal=A_matrix,
            thickness_dict=None,
            control_element="beam",
            control_dof=control_dof,
            control_displacement=step_size,
        )

        if not sol.get("convergence", False):
            if save_results:
                write_intermediate_model(model, str(result_dir), i)
                viewer = VtkRenderer(model.elems, model.contps)
                viewer.plot_displaced_points(factor=0, filename=str(result_dir)+f'/associative_elastic_iter{i}')
            break

        live_load = -float(sol["imposed_force"])

        _update_contp_force_3d(model.contps, sol["contact_forces"])
        _update_elem_disp_3d(model.contps, model.elems, sol["displacements"])

        disp = float(model.elems[beam_id].displacement[control_dof])

        forces.append(live_load)
        disps.append(disp)

        with fd_path.open("a", encoding="utf-8") as f:
            f.write(f"{live_load}, {disp}\n")

        if save_results and (i % write_freq == 0):
            write_intermediate_model(model, str(result_dir), i)
            viewer = VtkRenderer(model.elems, model.contps)
            viewer.plot_displaced_points(factor=0, filename=str(result_dir)+f'/associative_elastic_iter{i}')

        _displace_model_3d(model.elems, model.contps)

    _progress(max_iteration, "Done.")

    return {
        "fd_path": str(fd_path),
        "forces": forces,
        "displacements": disps,
        "steps_completed": len(forces),
    }
