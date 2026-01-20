"""
Contact point generation for rigid block model
"""

import os
import sys
from pathlib import Path

def generate_contact_points(
    material_json_path: str,
    mortar_ply_path: str,
    stones_dir: str,
    mortar_msh_path: str,
    output_dir: str,
    boundary_string: str = "double_bending",
    stone_stone_contact: str = "false"
) -> dict:
    """
    Generate contact points and element data for rigid block model.
    """
    
    print("="*60)
    print("CONTACT POINT GENERATION")
    print("="*60)
    
    # Run in subprocess to avoid signal handling issues
    import subprocess
    import sys
    
    # Get path to util_meshing module
    from pathlib import Path
    script_path = Path(__file__).parent / "util_meshing.py"
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        material_json_path,
        mortar_ply_path,
        stones_dir,
        mortar_msh_path,
        output_dir,
        boundary_string,
        stone_stone_contact
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run in subprocess
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Warnings/Errors:", result.stderr)
    
    # Return paths to generated files
    output_files = {
        'point': os.path.join(output_dir, 'point.csv'),
        'element': os.path.join(output_dir, 'element.csv'),
        'properties': os.path.join(output_dir, 'properties.json'),
        'iteration_map': os.path.join(output_dir, 'iteration_id_to_element_id_map.json'),
        'parameters': os.path.join(output_dir, 'parameters.json'),
        'sample_points': os.path.join(output_dir, 'sample_points.ply'),
        'contact_with_ground': os.path.join(output_dir, 'contact_points_with_ground.ply'),
        'potential_ground': os.path.join(output_dir, 'potential_ground_contact_point.ply')
    }
    
    print("\n✓ Contact point generation complete")
    return output_files


def check_prerequisites(work_dir: str) -> tuple:
    """
    Check if all required files for contact generation exist.
    
    Returns:
        tuple: (all_exist: bool, missing_files: list, file_paths: dict)
    """
    required_files = {
        'material_json': 'material.json',
        'mortar_ply': 'mortar.ply',
        'mortar_msh': 'mortar_01.msh',
        'stones_dir': 'stones'
    }
    
    file_paths = {}
    missing = []
    
    for key, filename in required_files.items():
        if key == 'stones_dir':
            path = os.path.join(work_dir, filename)
            if os.path.exists(path) and os.path.isdir(path):
                file_paths[key] = path
            else:
                missing.append(filename + '/ (directory)')
        else:
            path = os.path.join(work_dir, filename)
            if os.path.exists(path):
                file_paths[key] = path
            else:
                missing.append(filename)
    
    return len(missing) == 0, missing, file_paths