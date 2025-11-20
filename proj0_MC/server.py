"""
MCP Server for Sensitivity Analysis
====================================
This server provides tools for sensitivity analysis workflows.

Important: All MCP tools must use JSON-serializable types only:
- Parameters: str, int, float, bool, Dict, List
- Returns: str, int, float, bool, Dict, List
- NO custom objects (Variable, GaussianProcess, np.ndarray, etc.)
"""
from fastmcp import FastMCP
from typing import Dict, List, Any
from pathlib import Path
import json
import sys
import numpy as np

# ----------------------------------------------------------------------------
# Get project root directory (parent of this server file)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Add parent directory to path for imports from core/
PARENT_DIR = str(PROJECT_ROOT.parent)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


# ------------------------------------------------------------------------

# Initialize MCP server
mcp = FastMCP("SA-Analysis")

# ============================================================================
# Helper Functions for Importing Core Modules
# ============================================================================

"""
Import core modules needed for SA operations.
Returns tuple: (Variable, VariableSet, success, message)
"""
try:
    from core.Variables import Variable, VariableSet
    from core.Samplers import sample_inputs
except ImportError:
    print("Failed to import core modules")

# ============================================================================
# Test Tools
# ============================================================================

# @mcp.tool
# def hello(name: str) -> str:
#     """Say hello to someone."""
#     return f"Hello, {name}!"

# @mcp.tool
# def goodbye(name: str) -> str:
#     """Say goodbye to someone."""
#     return f"Goodbye, {name}!"

# ============================================================================
# Project Management Tools
# ============================================================================

@mcp.tool
def get_project_info() -> Dict[str, Any]:
    """
    Get information about the SA project directory.
    
    Returns:
        Dictionary with project path and available files.
    """
    
    return {
        "project_root": str(PROJECT_ROOT),
        "parent_dir": PARENT_DIR,
        "project_name": "SA-Analysis",
        "message": "Sensitivity Analysis Project"
    }




# ============================================================================
# Tools for Variable Management
# ============================================================================

@mcp.tool(name="MC.create_variable")
def create_variable(
    name: str,
    kind: str = None,
    params: Dict[str, float] = None,
    targets: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Create a variable definition for sensitivity analysis and add it to the workflow.
    
    Args:
        name: Variable name (e.g., "E", "F_y")
        kind: Distribution type ("uniform", "normal", "lognormal", "fixed")
        params: Distribution parameters as dict
            - uniform: {"low": float, "high": float}
            - normal: {"mean": float, "std": float}
            - lognormal: {"mean": float, "sigma": float}
            - fixed: {"value": float}
        targets: Optional list of injection targets
            - Each target: {"doc": str, "path": str}
            - Example: [{"doc": "FE", "path": "elements[*].E"}]
    
    Returns:
        Dictionary with:
        - success: boolean
        - variable: created variable dict
        - message: status message
    """
    try:
        # 1. Validate parameters
        if params is None:
            params = {}
        
        if targets is None:
            targets = []
        
        # Validate distribution parameters
        valid_kinds = ["uniform", "normal", "lognormal", "fixed"]
        if kind not in valid_kinds:
            return {
                "success": False,
                "variable": None,
                "message": f"Invalid kind '{kind}'. Must be one of {valid_kinds}"
            }
        
        # Check required parameters for each distribution
        if kind == "uniform":
            if "low" not in params or "high" not in params:
                return {
                    "success": False,
                    "variable": None,
                    "message": "Uniform distribution requires 'low' and 'high' parameters"
                }
        elif kind == "normal":
            if "mean" not in params or "std" not in params:
                return {
                    "success": False,
                    "variable": None,
                    "message": "Normal distribution requires 'mean' and 'std' parameters"
                }
        elif kind == "lognormal":
            if "mean" not in params or "sigma" not in params:
                return {
                    "success": False,
                    "variable": None,
                    "message": "Lognormal distribution requires 'mean' and 'sigma' parameters"
                }
        elif kind == "fixed":
            if "value" not in params:
                return {
                    "success": False,
                    "variable": None,
                    "message": "Fixed distribution requires 'value' parameter"
                }
        
        # 2. Create variable dict
        variable = {
            "name": name,
            "kind": kind,
            "params": params,
            "targets": targets
        }
        
        # 3. Load or create workflow file
        workflow_path = PROJECT_ROOT / "sa_workflow.json"
        
        if workflow_path.exists():
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)
        else:
            # Create new workflow
            from datetime import datetime
            workflow = {
                "workflow_id": "sa_workflow_001",
                "status": "in_progress",
                "stages": {
                    "variables_defined": False,
                    "samples_generated": False,
                    "model_evaluated": False,
                    "surrogate_trained": False,
                    "sa_computed": False
                },
                "variables": [],
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "updated": datetime.now().isoformat()
                }
            }
        
        # Check if variable with same name exists
        existing_idx = None
        for idx, v in enumerate(workflow["variables"]):
            if v["name"] == name:
                existing_idx = idx
                break
        
        # 4. Add or update variable
        if existing_idx is not None:
            workflow["variables"][existing_idx] = variable
            message = f"Updated variable '{name}'"
        else:
            workflow["variables"].append(variable)
            message = f"Created variable '{name}'"
        
        # Update workflow status
        workflow["stages"]["variables_defined"] = len(workflow["variables"]) > 0
        workflow["metadata"]["num_vars"] = len(workflow["variables"])
        
        from datetime import datetime
        workflow["metadata"]["updated"] = datetime.now().isoformat()
        
        # 5. Save workflow
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        return {
            "success": True,
            "variable": variable,
            "message": message,
            "workflow_file": str(workflow_path),
            "total_variables": len(workflow["variables"])
        }
        
    except Exception as e:
        return {
            "success": False,
            "variable": None,
            "message": f"Error creating variable: {str(e)}"
        }


@mcp.tool(name="MC.list_variables")
def list_variables() -> Dict[str, Any]:
    """
    List all defined variables in the current workflow.
    
    Returns:
        Dictionary with:
        - success: boolean
        - variables: list of variable dicts
        - num_vars: number of variables
        - message: status message
    """
    try:
        workflow_path = PROJECT_ROOT / "sa_workflow.json"
        
        if not workflow_path.exists():
            return {
                "success": True,
                "variables": [],
                "num_vars": 0,
                "message": "No workflow file found. Create variables to start."
            }
        
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        variables = workflow.get("variables", [])
        
        return {
            "success": True,
            "variables": variables,
            "num_vars": len(variables),
            "message": f"Found {len(variables)} variable(s)",
            "workflow_file": str(workflow_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "variables": [],
            "num_vars": 0,
            "message": f"Error listing variables: {str(e)}"
        }


@mcp.tool(name="MC.delete_variable")
def delete_variable(name: str) -> Dict[str, Any]:
    """
    Delete a variable from the workflow.
    
    Args:
        name: Variable name to delete
    
    Returns:
        Dictionary with:
        - success: boolean
        - message: status message
    """
    try:
        workflow_path = PROJECT_ROOT / "sa_workflow.json"
        
        if not workflow_path.exists():
            return {
                "success": False,
                "message": "No workflow file found"
            }
        
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        # Find and remove variable
        found = False
        for idx, v in enumerate(workflow["variables"]):
            if v["name"] == name:
                workflow["variables"].pop(idx)
                found = True
                break
        
        if not found:
            return {
                "success": False,
                "message": f"Variable '{name}' not found"
            }
        
        # Update metadata
        workflow["metadata"]["num_vars"] = len(workflow["variables"])
        workflow["stages"]["variables_defined"] = len(workflow["variables"]) > 0
        
        from datetime import datetime
        workflow["metadata"]["updated"] = datetime.now().isoformat()
        
        # Save
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        return {
            "success": True,
            "message": f"Deleted variable '{name}'",
            "remaining_variables": len(workflow["variables"])
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error deleting variable: {str(e)}"
        }


@mcp.tool(name="MC.clear_workflow")
def clear_workflow() -> Dict[str, Any]:
    """
    Clear the current workflow (removes all variables and resets workflow state).
    
    Returns:
        Dictionary with:
        - success: boolean
        - message: status message
    """
    try:
        workflow_path = PROJECT_ROOT / "sa_workflow.json"
        
        if not workflow_path.exists():
            return {
                "success": True,
                "message": "No workflow file to clear"
            }
        
        # Create fresh workflow
        from datetime import datetime
        workflow = {
            "workflow_id": "sa_workflow_001",
            "status": "in_progress",
            "stages": {
                "variables_defined": False,
                "samples_generated": False,
                "model_evaluated": False,
                "surrogate_trained": False,
                "sa_computed": False
            },
            "variables": [],
            "metadata": {
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "num_vars": 0
            }
        }
        
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        return {
            "success": True,
            "message": "Workflow cleared successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error clearing workflow: {str(e)}"
        }


@mcp.tool(name="MC.get_workflow_status")
def get_workflow_status() -> Dict[str, Any]:
    """
    Get the current workflow status and progress.
    
    Returns:
        Dictionary with:
        - success: boolean
        - workflow_id: workflow identifier
        - status: workflow status
        - stages: completion status of each stage
        - num_vars: number of defined variables
        - message: status message
    """
    try:
        workflow_path = PROJECT_ROOT / "sa_workflow.json"
        
        if not workflow_path.exists():
            return {
                "success": True,
                "workflow_id": None,
                "status": "not_started",
                "stages": {},
                "num_vars": 0,
                "message": "No workflow found. Create variables to start."
            }
        
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        return {
            "success": True,
            "workflow_id": workflow.get("workflow_id"),
            "status": workflow.get("status"),
            "stages": workflow.get("stages", {}),
            "num_vars": workflow.get("metadata", {}).get("num_vars", 0),
            "metadata": workflow.get("metadata", {}),
            "message": "Workflow status retrieved successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting workflow status: {str(e)}"
        }


# ============================================================================
# Tools for Sampling
# ============================================================================

@mcp.tool(name="MC.generate_samples")
def generate_samples(
    n_samples: int,
    method: str = "random",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate input samples based on variable definitions.
    
    Args:
        n_samples: Number of samples to generate
        method: Sampling method ("random", "sobol", "lhs")
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with:
        - success: boolean
        - n_samples: number of samples generated
        - samples_file: path to saved samples
        - variable_names: list of variable names
        - message: status message
    """
    try:
        # 1. Load variables from workflow
        workflow_path = PROJECT_ROOT / "sa_workflow.json"
        
        if not workflow_path.exists():
            return {
                "success": False,
                "n_samples": 0,
                "samples_file": None,
                "message": "No workflow file found. Create variables first using SA.create_variable()"
            }
        
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        variables_data = workflow.get("variables", [])
        
        if len(variables_data) == 0:
            return {
                "success": False,
                "n_samples": 0,
                "samples_file": None,
                "message": "No variables defined. Create variables first using SA.create_variable()"
            }
        
        # Validate method
        valid_methods = ["random", "sobol", "lhs"]
        if method not in valid_methods:
            return {
                "success": False,
                "n_samples": 0,
                "samples_file": None,
                "message": f"Invalid method '{method}'. Must be one of {valid_methods}"
            }
        
        # 2. Convert JSON variables to Variable objects and create VariableSet
       
        
        variables = []
        for var_data in variables_data:
            var = Variable(
                name=var_data["name"],
                kind=var_data["kind"],
                params=var_data["params"],
                targets=var_data.get("targets", [])
            )
            variables.append(var)
        
        vset = VariableSet(variables=variables)
        
        # 3. Generate samples
        samples = sample_inputs(vset, n_samples, kind=method, seed=seed)
        
        # 4. Prepare samples data for JSON
        from datetime import datetime
        
        samples_data = {
            "samples": samples.tolist(),  # Convert numpy array to list
            "n_samples": int(samples.shape[0]),
            "n_vars": int(samples.shape[1]),
            "variable_names": [v.name for v in variables],
            "sampling_method": method,
            "seed": seed,
            "created": datetime.now().isoformat(),
            "bounds": {
                "lower": vset.bounds()[0].tolist(),
                "upper": vset.bounds()[1].tolist()
            }
        }
        
        # 5. Save samples to JSON file
        samples_file = PROJECT_ROOT / "sa_samples.json"
        with open(samples_file, 'w') as f:
            json.dump(samples_data, f, indent=2)
        
        # 6. Update workflow status
        workflow["stages"]["samples_generated"] = True
        workflow["metadata"]["n_samples"] = n_samples
        workflow["metadata"]["sampling_method"] = method
        workflow["metadata"]["updated"] = datetime.now().isoformat()
        
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        return {
            "success": True,
            "n_samples": n_samples,
            "n_vars": len(variables),
            "samples_file": str(samples_file),
            "variable_names": [v.name for v in variables],
            "method": method,
            "seed": seed,
            "message": f"Generated {n_samples} samples using {method} method"
        }
        
    except Exception as e:
        return {
            "success": False,
            "n_samples": 0,
            "samples_file": None,
            "message": f"Error generating samples: {str(e)}"
        }


@mcp.tool(name="MC.get_samples")
def get_samples(max_rows: int = None) -> Dict[str, Any]:
    """
    Retrieve generated samples from the workflow.
    
    Args:
        max_rows: Optional limit on number of samples to return (for large datasets)
    
    Returns:
        Dictionary with:
        - success: boolean
        - samples: list of sample arrays (limited by max_rows if specified)
        - n_samples: total number of samples
        - n_vars: number of variables
        - variable_names: list of variable names
        - message: status message
    """
    try:
        samples_file = PROJECT_ROOT / "sa_samples.json"
        
        if not samples_file.exists():
            return {
                "success": False,
                "samples": [],
                "n_samples": 0,
                "n_vars": 0,
                "message": "No samples file found. Generate samples first using SA.generate_samples()"
            }
        
        with open(samples_file, 'r') as f:
            samples_data = json.load(f)
        
        samples = samples_data.get("samples", [])
        
        # Limit rows if requested
        if max_rows is not None and max_rows < len(samples):
            samples_to_return = samples[:max_rows]
            message = f"Retrieved {max_rows} of {len(samples)} samples (limited)"
        else:
            samples_to_return = samples
            message = f"Retrieved all {len(samples)} samples"
        
        return {
            "success": True,
            "samples": samples_to_return,
            "n_samples": samples_data.get("n_samples", len(samples)),
            "n_vars": samples_data.get("n_vars", 0),
            "variable_names": samples_data.get("variable_names", []),
            "sampling_method": samples_data.get("sampling_method"),
            "seed": samples_data.get("seed"),
            "message": message
        }
        
    except Exception as e:
        return {
            "success": False,
            "samples": [],
            "n_samples": 0,
            "n_vars": 0,
            "message": f"Error retrieving samples: {str(e)}"
        }


@mcp.tool(name="MC.get_sample_statistics")
def get_sample_statistics() -> Dict[str, Any]:
    """
    Get statistical summary of generated samples.
    
    Returns:
        Dictionary with:
        - success: boolean
        - statistics: dict with mean, std, min, max per variable
        - message: status message
    """
    try:
        samples_file = PROJECT_ROOT / "sa_samples.json"
        
        if not samples_file.exists():
            return {
                "success": False,
                "statistics": {},
                "message": "No samples file found. Generate samples first using SA.generate_samples()"
            }
        
        with open(samples_file, 'r') as f:
            samples_data = json.load(f)
        
        import numpy as np
        
        samples = np.array(samples_data.get("samples", []))
        variable_names = samples_data.get("variable_names", [])
        
        if len(samples) == 0:
            return {
                "success": False,
                "statistics": {},
                "message": "No samples available"
            }
        
        # Compute statistics per variable
        statistics = {}
        for i, var_name in enumerate(variable_names):
            var_samples = samples[:, i]
            statistics[var_name] = {
                "mean": float(np.mean(var_samples)),
                "std": float(np.std(var_samples)),
                "min": float(np.min(var_samples)),
                "max": float(np.max(var_samples)),
                "median": float(np.median(var_samples)),
                "q25": float(np.percentile(var_samples, 25)),
                "q75": float(np.percentile(var_samples, 75))
            }
        
        return {
            "success": True,
            "statistics": statistics,
            "n_samples": len(samples),
            "message": f"Computed statistics for {len(variable_names)} variables"
        }
        
    except Exception as e:
        return {
            "success": False,
            "statistics": {},
            "message": f"Error computing statistics: {str(e)}"
        }


# ============================================================================
# Helper Functions for Evaluation
# ============================================================================

def _resolve_evaluator(evaluator: str):
    """
    Resolve evaluator from name or import path.
    
    Supports:
    1. Built-in evaluators from a registry (if evaluators module exists)
    2. Import paths like "module.function" or "package.module.function"
    3. Built-in FE_static evaluator (inline)
    
    Args:
        evaluator: Evaluator name or import path
    
    Returns:
        Callable evaluator function
    
    Raises:
        ValueError: If evaluator cannot be resolved
    """
    # Try importing from evaluators module first
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from evaluators import EVALUATORS
        if evaluator in EVALUATORS:
            return EVALUATORS[evaluator]
    except ImportError:
        pass  # evaluators module doesn't exist, continue
    
    # Handle built-in FE_static evaluator
    if evaluator == "FE_static":
        # Import FE solver components
        sys.path.insert(0, str(PARENT_DIR))
        from FElib import elMatrixBar6DoF, rotateMat
        
        def solve_FE_static(config: Dict[str, Any]) -> Dict[str, Any]:
            """
            Solve static finite element problem.
            
            Args:
                config: FE configuration with nodes, elements, fixed_dofs, loads
            
            Returns:
                Dictionary with max_displacement, displacements, and success flag
            """
            try:
                nodes = config['nodes']
                elements = config['elements']
                fixed_dofs = config['fixed_dofs']
                loads = config['loads']
                
                nodes_by_id = {node['id']: node for node in nodes}
                
                # Build stiffness matrix
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
                    
                    T = rotateMat(xi, xj)
                    T_full = np.zeros((6, 6))
                    T_full[0:3, 0:3] = T
                    T_full[3:6, 3:6] = T
                    
                    k_local, _ = elMatrixBar6DoF(E=elem['E'], A=elem['A'], rho=elem['rho'], L=L)
                    k = T_full.T @ k_local @ T_full
                    
                    dof_i = [3*(ni-1), 3*(ni-1)+1, 3*(ni-1)+2]
                    dof_j = [3*(nj-1), 3*(nj-1)+1, 3*(nj-1)+2]
                    
                    K[np.ix_(dof_i, dof_i)] += k[0:3, 0:3]
                    K[np.ix_(dof_j, dof_j)] += k[3:6, 3:6]
                    K[np.ix_(dof_i, dof_j)] += k[0:3, 3:6]
                    K[np.ix_(dof_j, dof_i)] += k[3:6, 0:3]
                
                # Solve
                all_dofs = np.arange(3 * num_nodes)
                free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
                
                K_ff = K[np.ix_(free_dofs, free_dofs)]
                F_f = np.array(loads)[free_dofs]
                
                U = np.zeros(3 * num_nodes)
                U[free_dofs] = np.linalg.solve(K_ff, F_f)
                
                return {
                    'max_displacement': float(np.max(np.abs(U))),
                    'displacements': U.tolist(),
                    'num_dofs': int(len(U)),
                    'num_free_dofs': int(len(free_dofs)),
                    'success': True
                }
                
            except KeyError as e:
                return {
                    'max_displacement': None,
                    'success': False,
                    'error': f"Missing required key: {str(e)}"
                }
            except np.linalg.LinAlgError as e:
                return {
                    'max_displacement': None,
                    'success': False,
                    'error': f"Linear algebra error: {str(e)}"
                }
            except Exception as e:
                return {
                    'max_displacement': None,
                    'success': False,
                    'error': f"Unexpected error: {str(e)}"
                }
        
        return solve_FE_static
    
    # Try import path
    try:
        if '.' in evaluator:
            module_path, func_name = evaluator.rsplit('.', 1)
            module = __import__(module_path, fromlist=[func_name])
            return getattr(module, func_name)
        else:
            raise ValueError(f"Invalid evaluator format: '{evaluator}'")
    except Exception as e:
        raise ValueError(
            f"Could not resolve evaluator '{evaluator}'. "
            f"Not in registry and import failed: {e}"
        )


def _reconstruct_variables(workflow: Dict[str, Any]) -> List:
    """
    Reconstruct Variable objects from workflow JSON.
    
    Args:
        workflow: Workflow dictionary containing variables data
    
    Returns:
        List of Variable objects
    """
    variables_data = workflow.get('variables', [])
    variables = []
    
    for var_data in variables_data:
        var = Variable(
            name=var_data['name'],
            kind=var_data['kind'],
            params=var_data['params'],
            targets=var_data.get('targets', [])
        )
        variables.append(var)
    
    return variables


# ============================================================================
# Tools for Model Evaluation
# ============================================================================

@mcp.tool(name="MC.evaluate")
def evaluate(
    evaluator: str = "FE_static",
    input_config: str = "fe_config.json",
    samples_file: str = "sa_samples.json",
    workflow_file: str = "sa_workflow.json",
    output_file: str = "sa_results.json",
    output_key: str = "max_displacement"
) -> Dict[str, Any]:
    """
    Evaluate model for all generated samples using any callable evaluator.
    
    The evaluator function must follow this interface:
        evaluator(config: Dict) -> Dict with 'success' and output keys
    
    Args:
        evaluator: Evaluator name ("FE_static") or import path ("module.function")
        input_config: Path to base model configuration file (relative to project root)
        samples_file: Path to samples JSON file
        workflow_file: Path to workflow JSON file
        output_file: Path to save results
        output_key: Key to extract from evaluator results (e.g., "max_displacement")
    
    Returns:
        Dictionary with:
        - success: boolean
        - n_evaluated: number of samples evaluated
        - results_file: path to saved results
        - statistics: summary statistics
        - message: status message
    
    Example:
        # Using built-in FE evaluator
        evaluate(evaluator="FE_static", input_config="fe_config.json")
        
        # Using custom evaluator
        evaluate(evaluator="my_module.my_evaluator", output_key="my_output")
    """
    try:
        # ------------------------------------------------------------
        # 1. Check if required files exist
        #
        samples_path = PROJECT_ROOT / samples_file
        if not samples_path.exists():
            return {
                "success": False,
                "n_evaluated": 0,
                "results_file": None,
                "statistics": {},
                "message": f"Samples file not found: {samples_file}. Run MC.generate_samples first."
            }
        
        workflow_path = PROJECT_ROOT / workflow_file
        if not workflow_path.exists():
            return {
                "success": False,
                "n_evaluated": 0,
                "results_file": None,
                "statistics": {},
                "message": f"Workflow file not found: {workflow_file}. Define variables first."
            }
        
        config_path = PROJECT_ROOT / input_config
        if not config_path.exists():
            return {
                "success": False,
                "n_evaluated": 0,
                "results_file": None,
                "statistics": {},
                "message": f"Model config file not found: {input_config}"
            }
        # ------------------------------------------------------------
        # 2. Resolve evaluator function
        
        eval_func = _resolve_evaluator(evaluator)
        
        # ------------------------------------------------------------
        # 3. Load all required data
        
        with open(samples_path, 'r') as f:
            samples_data = json.load(f)
        
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        with open(config_path, 'r') as f:
            base_config = json.load(f)
        
        # ------------------------------------------------------------
        # 4. Take out samples
        
        samples = np.array(samples_data['samples'])
        variable_names = samples_data['variable_names']
        n_samples = len(samples)
        
        # 4. Reconstruct Variable objects
        variables = _reconstruct_variables(workflow)
        
        # Create VariableSet for proper injection
        var_set = VariableSet(variables=variables)
        
        # 5. Evaluate all samples
        results = []
        outputs = []
        
        for i, sample in enumerate(samples):
            # Create value dict for injection
            values = {variable_names[j]: float(sample[j]) for j in range(len(variable_names))}
            
            # Inject sample values into config using VariableSet
            # Wrap base_config with document name (assuming "FE" as default)
            configs = {"FE": base_config}
            modified_configs = var_set.inject_values(configs, values)
            modified_config = modified_configs["FE"]
            
            # Call evaluator (always with config dict)
            try:
                result = eval_func(modified_config)
                
                # Check if evaluation succeeded
                if not result.get('success', True):
                    return {
                        "success": False,
                        "n_evaluated": i,
                        "results_file": None,
                        "statistics": {},
                        "message": f"Evaluator failed on sample {i}: {result.get('error', 'Unknown error')}"
                    }
                
                # Extract output value
                output_value = result.get(output_key, None)
                if output_value is None:
                    return {
                        "success": False,
                        "n_evaluated": i,
                        "results_file": None,
                        "statistics": {},
                        "message": f"Output key '{output_key}' not found in evaluation result. Available keys: {list(result.keys())}"
                    }
                
                results.append({
                    'sample_id': i,
                    'inputs': values,
                    'output': output_value,
                    'full_result': result
                })
                outputs.append(output_value)
                
            except Exception as e:
                import traceback
                return {
                    "success": False,
                    "n_evaluated": i,
                    "results_file": None,
                    "statistics": {},
                    "message": f"Error evaluating sample {i}: {str(e)}",
                    "traceback": traceback.format_exc()
                }
        
        # 6. Compute statistics
        outputs_array = np.array(outputs)
        statistics = {
            'mean': float(np.mean(outputs_array)),
            'std': float(np.std(outputs_array)),
            'min': float(np.min(outputs_array)),
            'max': float(np.max(outputs_array)),
            'median': float(np.median(outputs_array)),
            'q25': float(np.percentile(outputs_array, 25)),
            'q75': float(np.percentile(outputs_array, 75))
        }
        
        # 7. Save results
        from datetime import datetime
        results_data = {
            'n_samples': n_samples,
            'n_vars': len(variable_names),
            'variable_names': variable_names,
            'output_key': output_key,
            'evaluator': evaluator,
            'results': results,
            'statistics': statistics,
            'created': datetime.now().isoformat()
        }
        
        results_path = PROJECT_ROOT / output_file
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # 8. Update workflow status
        workflow["stages"]["model_evaluated"] = True
        workflow["metadata"]["n_evaluated"] = n_samples
        workflow["metadata"]["results_file"] = output_file
        workflow["metadata"]["updated"] = datetime.now().isoformat()
        
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        return {
            "success": True,
            "n_evaluated": n_samples,
            "results_file": str(results_path),
            "statistics": statistics,
            "evaluator": evaluator,
            "output_key": output_key,
            "message": f"✓ Successfully evaluated {n_samples} samples using '{evaluator}'"
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "n_evaluated": 0,
            "results_file": None,
            "statistics": {},
            "message": f"Error during evaluation: {str(e)}",
            "traceback": traceback.format_exc()
        }


@mcp.tool(name="MC.get_results")
def get_results(
    results_file: str = "sa_results.json",
    max_rows: int = None
) -> Dict[str, Any]:
    """
    Retrieve evaluation results.
    
    Args:
        results_file: Path to results file
        max_rows: Optional limit on number of results to return (for large datasets)
    
    Returns:
        Dictionary with:
        - success: boolean
        - results: list of result dicts (limited by max_rows if specified)
        - n_samples: total number of samples evaluated
        - statistics: summary statistics
        - message: status message
    """
    try:
        results_path = PROJECT_ROOT / results_file
        
        if not results_path.exists():
            return {
                "success": False,
                "results": [],
                "n_samples": 0,
                "statistics": {},
                "message": f"Results file not found: {results_file}. Run SA.evaluate first."
            }
        
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        results = results_data.get("results", [])
        
        # Limit rows if requested
        if max_rows is not None and max_rows < len(results):
            results_to_return = results[:max_rows]
            message = f"Retrieved {max_rows} of {len(results)} results (limited)"
        else:
            results_to_return = results
            message = f"Retrieved all {len(results)} results"
        
        return {
            "success": True,
            "results": results_to_return,
            "n_samples": results_data.get("n_samples", len(results)),
            "n_vars": results_data.get("n_vars", 0),
            "variable_names": results_data.get("variable_names", []),
            "output_key": results_data.get("output_key", ""),
            "statistics": results_data.get("statistics", {}),
            "evaluator": results_data.get("evaluator", ""),
            "model_type": results_data.get("model_type", ""),
            "message": message
        }
        
    except Exception as e:
        return {
            "success": False,
            "results": [],
            "n_samples": 0,
            "statistics": {},
            "message": f"Error retrieving results: {str(e)}"
        }


# ============================================================================
# Tools for Surrogate Modeling
# ============================================================================

@mcp.tool(name="SA.train_surrogate")
def train_surrogate(
    model_type: str = "GaussianProcess",
    kernel_type: str = "RBF",
    train_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Train a surrogate model on evaluation results.
    
    Args:
        model_type: Surrogate type ("GaussianProcess", "PolynomialChaos")
        kernel_type: Kernel for GP ("RBF", "Matern")
        train_config: Training configuration dict
    
    Returns:
        Dictionary with:
        - success: boolean
        - config_file: path to surrogate config
        - model_file: path to trained model
        - performance: performance metrics
        - message: status message
    """
    # TODO: Implement surrogate training
    # 1. Load results
    # 2. Split and scale data
    # 3. Train model
    # 4. Evaluate performance
    # 5. Save config and model
    # 6. Return summary
    
    return {
        "success": False,
        "config_file": None,
        "model_file": None,
        "performance": {},
        "message": "Not implemented yet"
    }


# ============================================================================
# Tools for Sensitivity Analysis
# ============================================================================

@mcp.tool(name="SA.compute_sobol")
def compute_sobol(
    n_samples: int = 10000,
    use_surrogate: bool = True,
    seed: int = 123
) -> Dict[str, Any]:
    """
    Compute Sobol sensitivity indices.
    
    Args:
        n_samples: Number of samples for SA
        use_surrogate: Whether to use surrogate or direct evaluation
        seed: Random seed
    
    Returns:
        Dictionary with:
        - success: boolean
        - indices_file: path to saved indices
        - indices: sensitivity indices dict
        - message: status message
    """
    # TODO: Implement Sobol analysis
    # 1. Load variables
    # 2. Generate SA samples
    # 3. Evaluate using surrogate or direct
    # 4. Compute indices
    # 5. Save results
    # 6. Return summary
    
    return {
        "success": False,
        "indices_file": None,
        "indices": {},
        "message": "Not implemented yet"
    }


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
