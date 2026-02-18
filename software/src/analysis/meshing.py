"""
Meshing module using fTetWild for tetrahedral mesh generation
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def generate_mesh(
    input_mesh: str,
    output_dir: str,
    output_filename: str = "mortar_01.msh",
    max_element_size: float = 0.02,
    min_element_size: float = 0.002,
    mesh_algorithm: str = "delaunay",
    element_type: str = "tetrahedral",
    optimize: bool = True,
    optimization_iterations: int = 3,
    preserve_boundaries: bool = True,
    mesh_gradation: float = 1.3,
    angle_threshold: float = 30.0,
    output_format: str = "msh",
    stop_energy: int = 8,
    **kwargs
) -> str:
    """
    Generate a volumetric mesh using fTetWild (FloatTetwild).
    
    Args:
        input_mesh: Path to input surface mesh (should be PLY format)
        output_dir: Directory to save output mesh
        output_filename: Name of output mesh file (default: mortar_01.msh)
        max_element_size: Edge length parameter for fTetWild (default: 0.02)
        stop_energy: Stop energy parameter for fTetWild (default: 8)
        ... (other parameters for compatibility)
    
    Returns:
        Path to generated mesh file
    """
    
    print("="*60)
    print("MESH GENERATION USING fTetWild")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Full output path 
    output_basename = output_filename.replace('.msh', '.msh')
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\nInput mesh: {input_mesh}")
    print(f"Output directory: {output_dir}")
    print(f"Output file: {output_filename}")
    print(f"Edge length: {max_element_size}")
    print(f"Stop energy: {stop_energy}")
    print()
    
    # Run fTetWild
    try:
        result = process_mortar_tetrahedralization(
            input_ply_path=input_mesh,
            output_msh_path=output_path,
            edge_length=max_element_size,
            stop_energy=stop_energy
        )
        
        print("\n✓ Mesh generation complete")
        print(f"  Output: {output_path}")
        
        # Verify output exists
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"  File size: {file_size:.2f} MB")
        else:
            raise FileNotFoundError(f"Output mesh not created: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"\n✗ Meshing failed: {str(e)}")
        raise


def process_mortar_tetrahedralization(
    input_ply_path: str, 
    output_msh_path: str, 
    edge_length: float = 0.02, 
    stop_energy: int = 8,
    floattetwild_bin: Optional[str] = None
) -> subprocess.CompletedProcess:
    """
    Run FloatTetwild on a mortar.ply file to generate tetrahedral mesh.
    
    Args:
        input_ply_path: Path to the input mortar.ply file
        output_msh_path: Path where the output .msh file will be saved
        edge_length: Edge length parameter (default: 0.02)
        stop_energy: Stop energy parameter (default: 8)
        floattetwild_bin: Path to FloatTetwild binary (optional, will auto-detect)
    
    Returns:
        subprocess.CompletedProcess object with return code and output
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_msh_path) or '.', exist_ok=True)
    
    # Determine FloatTetwild binary path
    if floattetwild_bin is None:
        floattetwild_bin = find_floattetwild_binary()
    
    if not os.path.exists(floattetwild_bin):
        raise FileNotFoundError(
            f"FloatTetwild binary not found at: {floattetwild_bin}\n"
            f"Please build fTetWild or set FTETWILD_BIN environment variable"
        )
    
    print(f"Using FloatTetwild: {floattetwild_bin}")
    
    # Remove .msh extension from output path as fTetWild adds it automatically
    output_base = output_msh_path.replace('.msh', '.msh')
    
    # Build the command
    command = [
        floattetwild_bin,
        "-l", str(edge_length),
        "--stop-energy", str(stop_energy),
        "--input", input_ply_path,
        "-o", output_base
    ]
    
    print("Running command:")
    print(" ".join(command))
    print()
    
    # Run the command
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("FloatTetwild output:")
        print(result.stdout)
        
        if result.stderr:
            print("FloatTetwild stderr:")
            print(result.stderr)
        
        print(f"\n✓ FloatTetwild completed successfully!")
        
        return result
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running FloatTetwild:")
        print(f"  Return code: {e.returncode}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        raise RuntimeError(f"FloatTetwild failed: {e.stderr}")


def find_floattetwild_binary() -> str:
    """
    Auto-detect FloatTetwild binary location.
    
    Checks in order:
    1. FTETWILD_BIN environment variable
    2. Relative path from this script
    3. Common installation locations
    
    Returns:
        Path to FloatTetwild binary
    """
    
    # Check environment variable
    if 'FTETWILD_BIN' in os.environ:
        return os.environ['FTETWILD_BIN']
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Try relative paths (adjust these based on your project structure)
    possible_paths = [
        # Relative to analysis/ directory
        script_dir / "../../fTetWild/build/FloatTetwild_bin",
        script_dir / "../../../fTetWild/build/FloatTetwild_bin",
        
        # Relative to project root
        script_dir / "../../build/FloatTetwild_bin",
        
        # Absolute common locations
        Path.home() / "fTetWild/build/FloatTetwild_bin",
        Path("/usr/local/bin/FloatTetwild_bin"),
        Path("/opt/fTetWild/build/FloatTetwild_bin"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path.absolute())
    
    # Default fallback (will fail later if not found)
    return "../fTetWild/build/FloatTetwild_bin"


def check_ftetwild_available() -> bool:
    """
    Check if FloatTetwild is available.
    
    Returns:
        True if FloatTetwild binary is found, False otherwise
    """
    try:
        binary_path = find_floattetwild_binary()
        return os.path.exists(binary_path)
    except:
        return False


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python meshing.py <input_mesh.ply> [output_dir]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    
    print(f"FloatTetwild available: {check_ftetwild_available()}")
    print(f"FloatTetwild path: {find_floattetwild_binary()}")
    print()
    
    generate_mesh(
        input_mesh=input_file,
        output_dir=output_dir,
        output_filename="mortar_01.msh",
        max_element_size=0.02,
        stop_energy=8
    )