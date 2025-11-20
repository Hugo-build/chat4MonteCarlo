"""
MCP Server for Finite Element Analysis (Project-specific version)
==================================================================
This server provides tools for working with FE models within the project directory.
"""
from fastmcp import FastMCP
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base64
from io import BytesIO

mcp = FastMCP("FE-Analysis")

# Get project root directory (where this server is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

# ============================================================================
# Tool 1: Detect FE Files
# ============================================================================

@mcp.tool(name="detect_folders")
def detect_folders() -> Dict[str, Any]:
    """
    Detect all folders in the project directory.

    Args: 
        None
    
    Returns:
        Dictionary containing:
        - folders: list of folder names (relative to project root)
        - message: status message

    """
    folders = [str(f.relative_to(PROJECT_ROOT)) for f in PROJECT_ROOT.iterdir() if f.is_dir()]
    return {
        "folders": folders,
        "message": f"✓ Found {len(folders)} folders in the project"
    }

@mcp.tool(name="detect_json")
def detect_json(subdirectory: str = ".") -> Dict[str, Any]:
    """
    Detect JSON files in the project directory.
    
    Args:
        subdirectory: Subdirectory within the project to search (defaults to project root)
    
    Returns:
        Dictionary containing:
        - json_files: list of JSON file paths (relative to project root)
        - message: status message
    """
    # Resolve path relative to project root
    if subdirectory == ".":
        directory_path = PROJECT_ROOT
    else:
        directory_path = (PROJECT_ROOT / subdirectory).resolve()
    
    # Security: ensure we stay within project root
    try:
        directory_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return {
            "json_files": [],
            "message": "Access denied: path outside project directory"
        }
    
    if not directory_path.exists():
        return {
            "json_files": [],
            "message": f"Directory not found: {subdirectory}"
        }
    
    # Find all JSON files (using .suffix to check extension)
    json_files = [str(f.relative_to(PROJECT_ROOT)) for f in directory_path.iterdir() if f.is_file() and f.suffix == '.json']
    
    return {
        "json_files": json_files,
        "message": f"✓ Found {len(json_files)} JSON files in {subdirectory}"
    }
    
@mcp.tool(name="FE.detect_files")
def detect_fe_files(subdirectory: str = ".") -> Dict[str, Any]:
    """
    Detect FE model files (nodes.csv and elements.csv) in the project directory.
    
    Args:
        subdirectory: Subdirectory within the project to search (defaults to project root)
    
    Returns:
        Dictionary containing:
        - found: boolean indicating if both files were found
        - nodes_file: path to nodes.csv if found
        - elements_file: path to elements.csv if found
        - subdirectories: list of subdirectories that contain FE files
        - message: status message
    """
    # Resolve path relative to project root
    if subdirectory == ".":
        directory_path = PROJECT_ROOT
    else:
        directory_path = (PROJECT_ROOT / subdirectory).resolve()
    
    # Security: ensure we stay within project root
    try:
        directory_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return {
            "found": False,
            "nodes_file": None,
            "elements_file": None,
            "subdirectories": [],
            "message": f"Access denied: path outside project directory"
        }
    
    if not directory_path.exists():
        return {
            "found": False,
            "nodes_file": None,
            "elements_file": None,
            "subdirectories": [],
            "message": f"Directory not found: {subdirectory}"
        }
    
    # Check current directory
    nodes_file = directory_path / "nodes.csv"
    elements_file = directory_path / "elements.csv"
    
    current_dir_has_files = nodes_file.exists() and elements_file.exists()
    
    # Search subdirectories for FE files
    subdirs_with_fe = []
    for subdir in directory_path.iterdir():
        if subdir.is_dir():
            sub_nodes = subdir / "nodes.csv"
            sub_elements = subdir / "elements.csv"
            if sub_nodes.exists() and sub_elements.exists():
                # Return paths relative to project root
                rel_path = subdir.relative_to(PROJECT_ROOT)
                subdirs_with_fe.append({
                    "path": str(rel_path),
                    "name": subdir.name,
                    "nodes_file": str(sub_nodes.relative_to(PROJECT_ROOT)),
                    "elements_file": str(sub_elements.relative_to(PROJECT_ROOT))
                })
    
    result = {
        "found": current_dir_has_files,
        "nodes_file": str(nodes_file.relative_to(PROJECT_ROOT)) if nodes_file.exists() else None,
        "elements_file": str(elements_file.relative_to(PROJECT_ROOT)) if elements_file.exists() else None,
        "subdirectories": subdirs_with_fe,
        "project_root": str(PROJECT_ROOT),
        "message": ""
    }
    
    if current_dir_has_files:
        result["message"] = f"✓ FE files found in project {subdirectory}"
    elif subdirs_with_fe:
        result["message"] = f"✓ Found {len(subdirs_with_fe)} subdirectory(ies) with FE files"
    else:
        result["message"] = f"✗ No FE files found in {subdirectory} or its subdirectories"
    
    return result


# ============================================================================
# Tool 2: Load and Validate FE Files
# ============================================================================

@mcp.tool(name="FE.load_model")
def load_fe_model(subdirectory: str = ".") -> Dict[str, Any]:
    """
    Load FE model from nodes.csv and elements.csv files in the project.
    Automatically saves the model to fe_config.json in the same directory.
    
    Args:
        subdirectory: Subdirectory within the project containing the FE files (defaults to project root)
    
    Returns:
        Dictionary containing:
        - success: boolean indicating if loading was successful
        - nodes: list of node dictionaries with id and coordinates
        - elements: list of element dictionaries with connectivity and properties
        - stats: statistics about the model
        - config_saved: boolean indicating if config was saved to JSON
        - config_file: path to saved config file (relative to project root)
        - message: status message
    """
    # Resolve path relative to project root
    if subdirectory == ".":
        directory_path = PROJECT_ROOT
    else:
        directory_path = (PROJECT_ROOT / subdirectory).resolve()
    
    # Security: ensure we stay within project root
    try:
        directory_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return {
            "success": False,
            "nodes": [],
            "elements": [],
            "stats": {},
            "message": f"Access denied: path outside project directory"
        }
    
    nodes_file = directory_path / "nodes.csv"
    elements_file = directory_path / "elements.csv"
    
    # Validate files exist
    if not nodes_file.exists():
        return {
            "success": False,
            "nodes": [],
            "elements": [],
            "stats": {},
            "message": f"nodes.csv not found in {subdirectory}"
        }
    
    if not elements_file.exists():
        return {
            "success": False,
            "nodes": [],
            "elements": [],
            "stats": {},
            "message": f"elements.csv not found in {subdirectory}"
        }
    
    try:
        # Load nodes
        nodes_df = pd.read_csv(nodes_file)
        required_node_cols = ['node_id', 'x', 'y', 'z']
        if not all(col in nodes_df.columns for col in required_node_cols):
            return {
                "success": False,
                "nodes": [],
                "elements": [],
                "stats": {},
                "message": f"nodes.csv missing required columns: {required_node_cols}"
            }
        
        nodes = [
            {
                'id': int(row['node_id']),
                'coords': [float(row['x']), float(row['y']), float(row['z'])]
            }
            for _, row in nodes_df.iterrows()
        ]
        
        # Load elements
        elements_df = pd.read_csv(elements_file)
        required_elem_cols = ['element_id', 'node_i', 'node_j', 'E', 'A', 'rho']
        if not all(col in elements_df.columns for col in required_elem_cols):
            return {
                "success": False,
                "nodes": [],
                "elements": [],
                "stats": {},
                "message": f"elements.csv missing required columns: {required_elem_cols}"
            }
        
        elements = [
            {
                'id': int(row['element_id']),
                'node_i': int(row['node_i']),
                'node_j': int(row['node_j']),
                'E': float(row['E']),
                'A': float(row['A']),
                'rho': float(row['rho'])
            }
            for _, row in elements_df.iterrows()
        ]
        
        # Calculate statistics
        stats = {
            "num_nodes": len(nodes),
            "num_elements": len(elements),
            "node_id_range": [int(nodes_df['node_id'].min()), int(nodes_df['node_id'].max())],
            "element_id_range": [int(elements_df['element_id'].min()), int(elements_df['element_id'].max())],
            "E_range": [float(elements_df['E'].min()), float(elements_df['E'].max())],
            "A_range": [float(elements_df['A'].min()), float(elements_df['A'].max())],
            "bounds": {
                "x": [float(nodes_df['x'].min()), float(nodes_df['x'].max())],
                "y": [float(nodes_df['y'].min()), float(nodes_df['y'].max())],
                "z": [float(nodes_df['z'].min()), float(nodes_df['z'].max())]
            }
        }
        
        # Save FE config to JSON file (matching exp_40barTruss.py format)
        fe_config = {
            "nodes": nodes,
            "elements": elements
        }
        
        config_path = directory_path / "fe_config.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(fe_config, f, indent=2)
            config_saved = True
            config_file = str(config_path.relative_to(PROJECT_ROOT))
        except Exception as e:
            config_saved = False
            config_file = None
        
        return {
            "success": True,
            "nodes": nodes,
            "elements": elements,
            "stats": stats,
            "config_saved": config_saved,
            "config_file": config_file,
            "message": f"✓ Successfully loaded FE model: {len(nodes)} nodes, {len(elements)} elements" + 
                      (f" | Config saved to {config_file}" if config_saved else "")
        }
        
    except Exception as e:
        return {
            "success": False,
            "nodes": [],
            "elements": [],
            "stats": {},
            "message": f"Error loading FE model: {str(e)}"
        }


# ============================================================================
# Tool 3: Plot Undisplaced FE Model
# ============================================================================

@mcp.tool(name="FE.plot_model")
def plot_fe_model(subdirectory: str = ".", fe_config_file: str = "fe_config.json") -> Dict[str, Any]:
    """
    Plot the undisplaced FE model in 3D from a saved config file.
    
    Args:
        subdirectory: Subdirectory within the project containing the config file (defaults to project root)
        fe_config_file: Name of the JSON config file to load (defaults to "fe_config.json")
    
    Returns:
        Dictionary containing:
        - success: boolean indicating if plotting was successful
        - image_base64: base64 encoded PNG image
        - message: status message
    """
    # Resolve path relative to project root
    if subdirectory == ".":
        directory_path = PROJECT_ROOT
    else:
        directory_path = (PROJECT_ROOT / subdirectory).resolve()
    
    # Security: ensure we stay within project root
    try:
        directory_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return {
            "success": False,
            "image_base64": None,
            "message": "Access denied: path outside project directory"
        }
    
    config_path = directory_path / fe_config_file
    
    if not config_path.exists():
        return {
            "success": False,
            "image_base64": None,
            "message": f"Config file not found: {fe_config_file}"
        }
    
    try:
        # Load the model from JSON config
        with open(config_path, 'r') as f:
            model_data = json.load(f)
        
        nodes = model_data["nodes"]
        elements = model_data["elements"]
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot elements as lines
        for elem in elements:
            # Find node coordinates
            node_i = next(n for n in nodes if n['id'] == elem['node_i'])
            node_j = next(n for n in nodes if n['id'] == elem['node_j'])
            
            xs = [node_i['coords'][0], node_j['coords'][0]]
            ys = [node_i['coords'][1], node_j['coords'][1]]
            zs = [node_i['coords'][2], node_j['coords'][2]]
            
            ax.plot(xs, ys, zs, 'b-', linewidth=1, alpha=0.6)
        
        # Plot nodes
        node_coords = np.array([n['coords'] for n in nodes])
        ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], 
                  c='red', marker='o', s=50, alpha=0.8)
        
        # Labels and formatting
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'FE Model: {len(nodes)} nodes, {len(elements)} elements')
        
        # Calculate bounds for equal aspect ratio
        x_coords = [n['coords'][0] for n in nodes]
        y_coords = [n['coords'][1] for n in nodes]
        z_coords = [n['coords'][2] for n in nodes]
        
        max_range = max(
            max(x_coords) - min(x_coords),
            max(y_coords) - min(y_coords),
            max(z_coords) - min(z_coords)
        )
        mid_x = (max(x_coords) + min(x_coords)) / 2
        mid_y = (max(y_coords) + min(y_coords)) / 2
        mid_z = (max(z_coords) + min(z_coords)) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            "success": True,
            "image_base64": image_base64,
            "message": f"✓ Successfully plotted FE model ({len(nodes)} nodes, {len(elements)} elements)"
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "image_base64": None,
            "message": f"Error parsing config file: {str(e)}"
        }
    except KeyError as e:
        return {
            "success": False,
            "image_base64": None,
            "message": f"Invalid config format - missing key: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "image_base64": None,
            "message": f"Error plotting FE model: {str(e)}"
        }


# ============================================================================  # Tool 4: Add Fixed DOFs

@mcp.tool(name="FE.add_fixed_dofs")
def add_fixed_dofs(
    subdirectory: str = ".",
    config_filename: str = "fe_config.json",
    fixed_dofs: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Add fixed DOFs (boundary conditions) to an existing FE config file.
    
    Args:
        subdirectory: Subdirectory within the project containing the config file
        config_filename: Name of the config file to modify (defaults to "fe_config.json")
        fixed_dofs: List of DOF indices to fix (0-based). If None, fixes first 4 nodes (DOFs 0-11)
    
    Returns:
        Dictionary containing:
        - success: boolean indicating if fixed DOFs were added
        - num_fixed_dofs: number of fixed DOFs
        - message: status message
    """
    # Resolve path relative to project root
    if subdirectory == ".":
        directory_path = PROJECT_ROOT
    else:
        directory_path = (PROJECT_ROOT / subdirectory).resolve()
    
    # Security check
    try:
        directory_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return {
            "success": False,
            "message": "Access denied: path outside project directory"
        }
    
    config_path = directory_path / config_filename
    
    if not config_path.exists():
        return {
            "success": False,
            "message": f"Config file not found: {config_filename}. Run FE.load_model first."
        }
    
    try:
        # Load existing config
        with open(config_path, 'r') as f:
            fe_config = json.load(f)
        
        # Validate config has nodes and elements
        if "nodes" not in fe_config or "elements" not in fe_config:
            return {
                "success": False,
                "message": "Invalid config: missing 'nodes' or 'elements'"
            }
        
        num_nodes = len(fe_config["nodes"])
        
        # Set default fixed DOFs (first 4 nodes, all 3 DOFs each = DOFs 0-11)
        if fixed_dofs is None:
            fixed_dofs = list(range(12))  # Fix nodes 1-4 (DOFs 0-11)
        
        # Initialize loads array if not present
        if "loads" not in fe_config:
            fe_config["loads"] = [0.0] * (3 * num_nodes)
        
        # Update config with fixed DOFs
        fe_config["fixed_dofs"] = fixed_dofs
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(fe_config, f, indent=2)
        
        return {
            "success": True,
            "num_fixed_dofs": len(fixed_dofs),
            "fixed_dofs": fixed_dofs,
            "config_file": str(config_path.relative_to(PROJECT_ROOT)),
            "message": f"✓ Added {len(fixed_dofs)} fixed DOFs to config"
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "message": f"Error parsing config file: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error adding fixed DOFs: {str(e)}"
        }


@mcp.tool(name="FE.add_load")
def add_load(
    subdirectory: str = ".",
    config_filename: str = "fe_config.json",
    node_id: int = 10,
    direction: str = "y",
    magnitude: float = 1000.0) -> Dict[str, Any]:
    """
    Add a load to an existing FE config file at a specific node and direction.
    
    Args:
        subdirectory: Subdirectory within the project containing the config file
        config_filename: Name of the config file to modify (defaults to "fe_config.json")
        node_id: Node ID to apply load (1-based)
        direction: Direction of load ('x', 'y', or 'z')
        magnitude: Magnitude of the load in Newtons
    
    Returns:
        Dictionary containing:
        - success: boolean indicating if load was added
        - load_dof: DOF index where load was applied
        - message: status message
    """
    # Resolve path relative to project root
    if subdirectory == ".":
        directory_path = PROJECT_ROOT
    else:
        directory_path = (PROJECT_ROOT / subdirectory).resolve()
    
    # Security check
    try:
        directory_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return {
            "success": False,
            "message": "Access denied: path outside project directory"
        }
    
    config_path = directory_path / config_filename
    
    if not config_path.exists():
        return {
            "success": False,
            "message": f"Config file not found: {config_filename}. Run FE.load_model first."
        }
    
    try:
        # Load existing config
        with open(config_path, 'r') as f:
            fe_config = json.load(f)
        
        # Validate config has nodes and elements
        if "nodes" not in fe_config or "elements" not in fe_config:
            return {
                "success": False,
                "message": "Invalid config: missing 'nodes' or 'elements'"
            }
        
        num_nodes = len(fe_config["nodes"])
        
        # Validate node ID exists
        node_ids = [n["id"] for n in fe_config["nodes"]]
        if node_id not in node_ids:
            return {
                "success": False,
                "message": f"Invalid node_id: {node_id}. Node not found in model. Available nodes: {min(node_ids)} to {max(node_ids)}"
            }
        
        # Validate direction
        direction_map = {"x": 0, "y": 1, "z": 2}
        if direction.lower() not in direction_map:
            return {
                "success": False,
                "message": f"Invalid direction: {direction}. Must be 'x', 'y', or 'z'"
            }
        
        # Initialize or update load vector
        if "loads" not in fe_config:
            loads = [0.0] * (3 * num_nodes)
        else:
            loads = fe_config["loads"]
            # Ensure it has the right size
            if len(loads) != 3 * num_nodes:
                loads = [0.0] * (3 * num_nodes)
        
        # Calculate DOF index and apply load
        dof_offset = direction_map[direction.lower()]
        load_dof = 3 * (node_id - 1) + dof_offset
        loads[load_dof] = float(magnitude)
        
        # Initialize fixed_dofs if not present
        if "fixed_dofs" not in fe_config:
            fe_config["fixed_dofs"] = list(range(12))  # Default: fix first 4 nodes
        
        # Update config
        fe_config["loads"] = loads
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(fe_config, f, indent=2)
        
        num_loads = sum(1 for f in loads if f != 0)
        
        return {
            "success": True,
            "load_dof": load_dof,
            "node_id": node_id,
            "direction": direction.upper(),
            "magnitude": magnitude,
            "total_loads": num_loads,
            "config_file": str(config_path.relative_to(PROJECT_ROOT)),
            "message": f"✓ Added load: {magnitude}N at Node {node_id} ({direction.upper()}-direction, DOF {load_dof})"
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "message": f"Error parsing config file: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error adding load: {str(e)}"
        }




@mcp.tool(name="FE.solve_static")
def solve_fe_static(
    subdirectory: str = ".",
    config_filename: str = "fe_config.json",
    save_results: bool = True) -> Dict[str, Any]:
    """
    Solve a static FE model using the configuration file.
    Based on solve_FE_static from exp_40barTruss.py.
    
    Args:
        subdirectory: Subdirectory within the project containing the config file
        config_filename: Name of the config file to solve (defaults to "fe_config.json")
        save_results: Whether to save results back to config file (defaults to True)
    
    Returns:
        Dictionary containing:
        - success: boolean indicating if solving was successful
        - displacements: displacement vector
        - max_displacement: maximum displacement magnitude
        - message: status message
    """
    # Import FElib functions - they should be available since they're in the parent directory
    import sys
    from pathlib import Path as PathLib
    parent_dir = str(PathLib(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from FElib import elMatrixBar6DoF, rotateMat
    except ImportError as e:
        return {
            "success": False,
            "message": f"Failed to import FElib: {str(e)}. Ensure FElib.py is in the parent directory."
        }
    
    # Resolve path relative to project root
    if subdirectory == ".":
        directory_path = PROJECT_ROOT
    else:
        directory_path = (PROJECT_ROOT / subdirectory).resolve()
    
    # Security check
    try:
        directory_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return {
            "success": False,
            "message": "Access denied: path outside project directory"
        }
    
    config_path = directory_path / config_filename
    
    if not config_path.exists():
        return {
            "success": False,
            "message": f"Config file not found: {config_filename}. Run FE.load_model and add BCs first."
        }
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            fe_config = json.load(f)
        
        # Validate config
        required_keys = ['nodes', 'elements', 'fixed_dofs', 'loads']
        missing = [k for k in required_keys if k not in fe_config]
        if missing:
            return {
                "success": False,
                "message": f"Config missing required keys: {missing}. Add boundary conditions first."
            }
        
        nodes = fe_config['nodes']
        elements = fe_config['elements']
        fixed_dofs = fe_config['fixed_dofs']
        loads = fe_config['loads']
        
        # Create a lookup dictionary for nodes by id
        nodes_by_id = {node['id']: node for node in nodes}
        
        # Build global stiffness matrix
        num_nodes = len(nodes)
        K = np.zeros((3 * num_nodes, 3 * num_nodes))
        
        for elem in elements:
            ni = elem['node_i']
            nj = elem['node_j']
            node_i = nodes_by_id[ni]
            node_j = nodes_by_id[nj]
            
            xi = np.array(node_i['coords'])
            xj = np.array(node_j['coords'])
            L = np.linalg.norm(xj - xi)
            
            # Get rotation transformation matrix
            T = rotateMat(xi, xj)
            
            T_full = np.zeros((6, 6))
            T_full[0:3, 0:3] = T  # for node i
            T_full[3:6, 3:6] = T  # for node j
            
            # Get element stiffness matrix in local coordinates
            k_local, _ = elMatrixBar6DoF(E=elem['E'], A=elem['A'], rho=elem['rho'], L=L)
            
            # Transform element stiffness to global coordinates
            k = T_full.T @ k_local @ T_full
            
            # DOF indices (0-based)
            dof_i = [3*(ni-1), 3*(ni-1)+1, 3*(ni-1)+2]
            dof_j = [3*(nj-1), 3*(nj-1)+1, 3*(nj-1)+2]
            
            # Assemble local to global
            K[np.ix_(dof_i, dof_i)] += k[0:3, 0:3]
            K[np.ix_(dof_j, dof_j)] += k[3:6, 3:6]
            K[np.ix_(dof_i, dof_j)] += k[0:3, 3:6]
            K[np.ix_(dof_j, dof_i)] += k[3:6, 0:3]
        
        # Apply boundary conditions (eliminate fixed DOFs)
        all_dofs = np.arange(3 * num_nodes)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = np.array(loads)[free_dofs]
        
        # Solve for displacements
        U = np.zeros(3 * num_nodes)
        U[free_dofs] = np.linalg.solve(K_ff, F_f)
        
        # Calculate statistics
        max_disp = float(np.max(np.abs(U)))
        
        # Save results if requested
        if save_results:
            fe_config['displacements'] = U.tolist()
            fe_config['max_displacement'] = max_disp
            fe_config['solved'] = True
            
            with open(config_path, 'w') as f:
                json.dump(fe_config, f, indent=2)
        
        return {
            "success": True,
            "displacements": U.tolist(),
            "max_displacement": max_disp,
            "num_dofs": len(U),
            "num_free_dofs": len(free_dofs),
            "num_fixed_dofs": len(fixed_dofs),
            "config_file": str(config_path.relative_to(PROJECT_ROOT)),
            "message": f"✓ Successfully solved FE model: max displacement = {max_disp:.6e} mm"
        }
        
    except np.linalg.LinAlgError as e:
        return {
            "success": False,
            "message": f"Linear algebra error during solving: {str(e)}. Check if model is properly constrained."
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "message": f"Error parsing config file: {str(e)}"
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error solving FE model: {str(e)}",
            "traceback": traceback.format_exc()
        }


# ============================================================================
# Tool 4: Get Model Info
# ============================================================================

@mcp.tool(name="FE.get_info")
def get_fe_info(subdirectory: str = ".") -> str:
    """
    Get detailed information about an FE model in a human-readable format.
    
    Args:
        subdirectory: Subdirectory within the project containing the FE files (defaults to project root)
    
    Returns:
        Formatted string with model information
    """
    result = load_fe_model(subdirectory)
    
    if not result["success"]:
        return result["message"]
    
    stats = result["stats"]
    
    info = f"""
    FE Model Information
    {'='*60}
    Project: {PROJECT_ROOT.name}
    Subdirectory: {subdirectory}

    Model Statistics:
    • Nodes: {stats['num_nodes']}
    • Elements: {stats['num_elements']}
    • Node IDs: {stats['node_id_range'][0]} to {stats['node_id_range'][1]}
    • Element IDs: {stats['element_id_range'][0]} to {stats['element_id_range'][1]}

    Material Properties:
    • Young's Modulus (E): {stats['E_range'][0]:.2e} to {stats['E_range'][1]:.2e}
    • Cross-sectional Area (A): {stats['A_range'][0]:.2f} to {stats['A_range'][1]:.2f}

    Geometry Bounds:
    • X: [{stats['bounds']['x'][0]:.2f}, {stats['bounds']['x'][1]:.2f}]
    • Y: [{stats['bounds']['y'][0]:.2f}, {stats['bounds']['y'][1]:.2f}]
    • Z: [{stats['bounds']['z'][0]:.2f}, {stats['bounds']['z'][1]:.2f}]

    Status: ✓ Model loaded successfully
    {'='*60}
    """
    
    return info.strip()


@mcp.tool(name="FE.plot_deformed")
def plot_deformed(
    subdirectory: str = ".",
    config_filename: str = "fe_config.json",
    scale: float = 1000.0,
    cmap: str = None) -> Dict[str, Any]:
    """
    Plot the deformed FE model overlaid with the undeformed structure.
    Based on plot_displacements from exp_40barTruss.py.
    
    Args:
        subdirectory: Subdirectory within the project containing the config file
        config_filename: Name of the config file with solved displacements
        scale: Scale factor for displacements (to make them visible)
        cmap: Colormap name for coloring by displacement magnitude (e.g., 'plasma', 'viridis', 'jet').
              If None, uses solid orange color.
    
    Returns:
        Dictionary containing:
        - success: boolean indicating if plotting was successful
        - image_base64: base64 encoded PNG image
        - message: status message
    """
    # Resolve path relative to project root
    if subdirectory == ".":
        directory_path = PROJECT_ROOT
    else:
        directory_path = (PROJECT_ROOT / subdirectory).resolve()
    
    # Security check
    try:
        directory_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return {
            "success": False,
            "image_base64": None,
            "message": "Access denied: path outside project directory"
        }
    
    config_path = directory_path / config_filename
    
    if not config_path.exists():
        return {
            "success": False,
            "image_base64": None,
            "message": f"Config file not found: {config_filename}"
        }
    
    try:
        # Load config with displacements
        with open(config_path, 'r') as f:
            fe_config = json.load(f)
        
        # Validate config has displacements
        if "displacements" not in fe_config:
            return {
                "success": False,
                "image_base64": None,
                "message": "Config does not contain displacements. Run FE.solve_static first."
            }
        
        nodes = fe_config["nodes"]
        elements = fe_config["elements"]
        displacements = np.array(fe_config["displacements"])
        
        # Create a lookup dictionary for nodes by id
        nodes_by_id = {node['id']: node for node in nodes}
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate bounds for the structure
        x_coords = []
        y_coords = []
        z_coords = []
        
        for node in nodes:
            x_coords.append(node['coords'][0])
            y_coords.append(node['coords'][1])
            z_coords.append(node['coords'][2])
        
        # Calculate max displacement for display
        max_displacement = float(np.max(np.abs(displacements)))
        
        # Calculate element displacement magnitudes for colormap (if enabled)
        if cmap is not None:
            from matplotlib import cm
            from matplotlib.colors import Normalize
            
            element_displacements = []
            for elem in elements:
                ni = elem['node_i']
                nj = elem['node_j']
                
                # Get nodal displacements
                disp_i = displacements[3*(ni-1):3*(ni-1)+3]
                disp_j = displacements[3*(nj-1):3*(nj-1)+3]
                
                # Average displacement magnitude for this element
                avg_disp = (np.linalg.norm(disp_i) + np.linalg.norm(disp_j)) / 2
                element_displacements.append(avg_disp)
            
            element_displacements = np.array(element_displacements)
            norm = Normalize(vmin=element_displacements.min(), vmax=element_displacements.max())
            colormap = cm.get_cmap(cmap)
        
        # Plot elements
        for i, elem in enumerate(elements):
            ni = elem['node_i']
            nj = elem['node_j']
            node_i = nodes_by_id[ni]
            node_j = nodes_by_id[nj]
            
            p1 = np.array(node_i['coords'])
            p2 = np.array(node_j['coords'])
            
            # Plot original structure (undeformed) - blue
            if i == 0:
                ax.plot(*zip(p1, p2), color='b', linewidth=1.5, alpha=0.5, label='Undeformed')
            else:
                ax.plot(*zip(p1, p2), color='b', linewidth=1.5, alpha=0.5)
            
            # Plot deformed structure - orange (default) or colormap
            d1 = p1 + scale * displacements[3*(ni-1):3*(ni-1)+3]
            d2 = p2 + scale * displacements[3*(nj-1):3*(nj-1)+3]
            
            if cmap is not None:
                color = colormap(norm(element_displacements[i]))
                label = f'Deformed (colored by displacement)' if i == 0 else None
            else:
                color = 'orange'
                label = 'Deformed' if i == 0 else None
            
            ax.plot(*zip(d1, d2), color=color, linewidth=2.5, alpha=0.8, label=label)
        
        # Plot nodes (undeformed)
        node_coords = np.array([n['coords'] for n in nodes])
        ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], 
                  c='blue', marker='o', s=30, alpha=0.5)
        
        # Plot nodes (deformed)
        deformed_coords = np.array([
            [nodes[i]['coords'][j] + scale * displacements[3*i + j] 
             for j in range(3)] 
            for i in range(len(nodes))
        ])
        
        if cmap is not None:
            # Color nodes by their displacement magnitude
            node_displacements = np.array([
                np.linalg.norm(displacements[3*i:3*i+3]) 
                for i in range(len(nodes))
            ])
            ax.scatter(deformed_coords[:, 0], deformed_coords[:, 1], deformed_coords[:, 2], 
                      c=node_displacements, cmap=cmap, marker='o', s=50, alpha=0.8,
                      norm=norm)
            # Add colorbar
            mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
            mappable.set_array(element_displacements)
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Displacement Magnitude [mm]', rotation=270, labelpad=20)
        else:
            ax.scatter(deformed_coords[:, 0], deformed_coords[:, 1], deformed_coords[:, 2], 
                      c='orange', marker='o', s=50, alpha=0.8)
        
        # Labels and formatting
        title = f'Original vs Deformed Structure\n'
        title += f'Deformation scale: {scale}x (Max displacement: {max_displacement:.2e} mm)'
        if cmap is not None:
            title += f' [Colormap: {cmap}]'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        max_range = max(
            max(x_coords) - min(x_coords),
            max(y_coords) - min(y_coords),
            max(z_coords) - min(z_coords)
        )
        mid_x = (max(x_coords) + min(x_coords)) / 2
        mid_y = (max(y_coords) + min(y_coords)) / 2
        mid_z = (max(z_coords) + min(z_coords)) / 2
        
        # Expand range slightly to accommodate deformed shape
        range_factor = 1.2
        ax.set_xlim(mid_x - max_range*range_factor/2, mid_x + max_range*range_factor/2)
        ax.set_ylim(mid_y - max_range*range_factor/2, mid_y + max_range*range_factor/2)
        ax.set_zlim(mid_z - max_range*range_factor/2, mid_z + max_range*range_factor/2)
        
        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        colormap_status = f"with colormap '{cmap}'" if cmap else "in orange"
        return {
            "success": True,
            "image_base64": image_base64,
            "scale": scale,
            "max_displacement": max_displacement,
            "cmap": cmap,
            "message": f"✓ Successfully plotted deformed model {colormap_status} (scale: {scale}x, max disp: {max_displacement:.6e} mm)"
        }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "image_base64": None,
            "message": f"Error parsing config file: {str(e)}"
        }
    except KeyError as e:
        return {
            "success": False,
            "image_base64": None,
            "message": f"Invalid config format - missing key: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "image_base64": None,
            "message": f"Error plotting deformed model: {str(e)}"
        }
 

# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    mcp.run()

