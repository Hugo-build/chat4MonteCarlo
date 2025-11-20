"""
Standalone Evaluator Functions
================================

This module contains model evaluator functions that can be executed independently.
Each evaluator follows the contract:
    - Input: config (Dict[str, Any])
    - Output: results (Dict[str, Any]) with at least one output key

These functions are model-specific and contain the actual execution logic.
They are called by the general evaluation framework in evaluate.py.

Usage:
    from evaluators import solve_FE_static
    
    result = solve_FE_static(config)
    print(result['max_displacement'])
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add parent directory to path for imports
PARENT_DIR = str(Path(__file__).parent.parent)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from FElib import elMatrixBar6DoF, rotateMat


# ============================================================================
# FE Static Solver
# ============================================================================

def solve_FE_static(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solve static finite element problem.
    
    This is a standalone evaluator that can be called independently
    or used with the evaluation framework.
    
    Args:
        config: FE configuration dictionary containing:
            - nodes: List of node dicts with 'id' and 'coords'
            - elements: List of element dicts with connectivity and properties
            - fixed_dofs: List of fixed DOF indices
            - loads: Load vector (list or array)
    
    Returns:
        Dictionary with:
            - max_displacement: Maximum displacement magnitude
            - displacements: Full displacement vector (list)
            - num_dofs: Total number of DOFs
            - num_free_dofs: Number of free DOFs
            - success: Boolean indicating success
            - error: Error message if failed (optional)
    
    Example:
        >>> config = {
        ...     'nodes': [...],
        ...     'elements': [...],
        ...     'fixed_dofs': [0, 1, 2, ...],
        ...     'loads': [0, 0, ..., 1000, ...]
        ... }
        >>> result = solve_FE_static(config)
        >>> print(result['max_displacement'])
        0.00245
    """
    try:
        # Extract configuration
        nodes = config['nodes']
        elements = config['elements']
        fixed_dofs = config['fixed_dofs']
        loads = config['loads']
        
        # Create node lookup
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
            'displacements': None,
            'num_dofs': 0,
            'num_free_dofs': 0,
            'success': False,
            'error': f"Missing required key in config: {str(e)}"
        }
    
    except np.linalg.LinAlgError as e:
        return {
            'max_displacement': None,
            'displacements': None,
            'num_dofs': 0,
            'num_free_dofs': 0,
            'success': False,
            'error': f"Linear algebra error (singular matrix?): {str(e)}"
        }
    
    except Exception as e:
        return {
            'max_displacement': None,
            'displacements': None,
            'num_dofs': 0,
            'num_free_dofs': 0,
            'success': False,
            'error': f"Unexpected error: {str(e)}"
        }


# ============================================================================
# Surrogate Predictor (Placeholder)
# ============================================================================

def predict_surrogate(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict using a trained surrogate model.
    
    Args:
        config: Configuration dictionary containing:
            - surrogate_path: Path to trained surrogate model
            - input_features: Features to predict from
            (or the config is used to extract features)
    
    Returns:
        Dictionary with:
            - max_displacement: Predicted value
            - uncertainty: Prediction uncertainty (if available)
            - success: Boolean indicating success
    
    Note:
        This is a placeholder. Implement based on your surrogate format.
    """
    # TODO: Implement surrogate prediction
    # Example structure:
    # surrogate = load_surrogate(config['surrogate_path'])
    # features = extract_features(config)
    # prediction = surrogate.predict(features)
    # return {'max_displacement': prediction, 'success': True}
    
    return {
        'max_displacement': None,
        'uncertainty': None,
        'success': False,
        'error': "Surrogate evaluation not yet implemented"
    }


# ============================================================================
# Custom Model Evaluator (Example Template)
# ============================================================================

def run_custom_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Template for custom model evaluator.
    
    Replace this with your own model-specific logic.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Dictionary with results (must include at least one output)
    
    Example:
        >>> def my_model(config):
        ...     result = my_solver(config['params'])
        ...     return {'output': result, 'success': True}
    """
    # TODO: Replace with your model logic
    return {
        'output': None,
        'success': False,
        'error': "Custom model not implemented"
    }


# ============================================================================
# Evaluator Registry
# ============================================================================

EVALUATORS = {
    "FE_static": solve_FE_static,
    "surrogate": predict_surrogate,
    "custom": run_custom_model
}


def get_evaluator(name: str):
    """
    Get evaluator function by name.
    
    Args:
        name: Evaluator name (from EVALUATORS registry)
    
    Returns:
        Evaluator function
    
    Raises:
        ValueError: If evaluator name not found
    
    Example:
        >>> evaluator = get_evaluator("FE_static")
        >>> result = evaluator(config)
    """
    if name not in EVALUATORS:
        available = ", ".join(EVALUATORS.keys())
        raise ValueError(
            f"Unknown evaluator: '{name}'. "
            f"Available evaluators: {available}"
        )
    return EVALUATORS[name]


def list_evaluators():
    """
    List all registered evaluators.
    
    Returns:
        List of evaluator names
    """
    return list(EVALUATORS.keys())


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_FE_config(config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate FE configuration has required fields.
    
    Args:
        config: Configuration to validate
    
    Returns:
        (is_valid, error_message)
    """
    required_keys = ['nodes', 'elements', 'fixed_dofs', 'loads']
    
    for key in required_keys:
        if key not in config:
            return False, f"Missing required key: '{key}'"
    
    if len(config['nodes']) == 0:
        return False, "No nodes defined"
    
    if len(config['elements']) == 0:
        return False, "No elements defined"
    
    if len(config['fixed_dofs']) == 0:
        return False, "No fixed DOFs defined (model may be under-constrained)"
    
    return True, None


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Evaluators Module - Standalone Test")
    print("=" * 70)
    
    # List available evaluators
    print("\nAvailable evaluators:")
    for name in list_evaluators():
        print(f"  • {name}")
    
    # Test with dummy config (will fail gracefully)
    print("\nTesting FE evaluator with minimal config:")
    test_config = {
        'nodes': [{'id': 1, 'coords': [0, 0, 0]}],
        'elements': [],
        'fixed_dofs': [0, 1, 2],
        'loads': [0, 0, 0]
    }
    
    result = solve_FE_static(test_config)
    print(f"  Success: {result['success']}")
    if not result['success']:
        print(f"  Error: {result['error']}")
    
    print("\n" + "=" * 70)
    print("To use: from evaluators import solve_FE_static")
    print("=" * 70)

