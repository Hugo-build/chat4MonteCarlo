"""
General Evaluation Framework
==============================

This module provides a model-agnostic evaluation framework.
It handles:
- Sample loading and management
- Variable injection into configurations
- Looping over samples
- Calling evaluator functions
- Results collection and statistics

The framework accepts any callable that follows the evaluator contract:
    evaluator(config: Dict) -> Dict

Usage:
    from evaluate import evaluate_samples
    from evaluators import solve_FE_static
    
    results = evaluate_samples(
        evaluator_callable=solve_FE_static,
        base_config=config,
        samples=samples_array,
        variables=variables_list,
        output_key="max_displacement"
    )
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
import numpy as np
import json
from datetime import datetime

# Add parent directory to path for imports
PARENT_DIR = str(Path(__file__).parent.parent)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from core.Variables import Variable, inject_single_config


# ============================================================================
# Core Evaluation Function
# ============================================================================

def evaluate_samples(
    evaluator_callable: Callable[[Dict[str, Any]], Dict[str, Any]],
    base_config: Dict[str, Any],
    samples: np.ndarray,
    variables: List[Variable],
    output_key: str = "output",
    variable_names: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Evaluate multiple samples using a provided callable function.
    
    This is the core evaluation framework that:
    1. Loops over samples
    2. Injects variable values into base config
    3. Calls evaluator for each modified config
    4. Collects results and computes statistics
    
    Args:
        evaluator_callable: Function that takes config and returns results
            Must follow signature: f(config: Dict) -> Dict
        base_config: Base configuration dictionary to modify
        samples: Array of shape (n_samples, n_vars) with sampled values
        variables: List of Variable objects with targets for injection
        output_key: Key to extract from evaluator results
        variable_names: Optional list of variable names (inferred if None)
        progress_callback: Optional callback(current, total) for progress
    
    Returns:
        Dictionary containing:
            - success: Boolean indicating if evaluation completed
            - n_evaluated: Number of samples successfully evaluated
            - n_failed: Number of failed evaluations
            - results: List of result dicts
            - outputs: List of output values (for statistics)
            - statistics: Dict with mean, std, min, max, etc.
            - failed_samples: List of failed sample indices
            - message: Status message
    
    Example:
        >>> from evaluators import solve_FE_static
        >>> results = evaluate_samples(
        ...     evaluator_callable=solve_FE_static,
        ...     base_config=fe_config,
        ...     samples=sample_array,
        ...     variables=variables,
        ...     output_key="max_displacement"
        ... )
        >>> print(results["statistics"]["mean"])
    """
    n_samples = len(samples)
    n_vars = samples.shape[1] if len(samples.shape) > 1 else 1
    
    # Infer variable names if not provided
    if variable_names is None:
        variable_names = [v.name for v in variables]
    
    # Validate inputs
    if len(variable_names) != n_vars:
        return {
            'success': False,
            'n_evaluated': 0,
            'n_failed': 0,
            'message': f"Mismatch: {len(variable_names)} variable names but {n_vars} columns in samples"
        }
    
    if len(variables) != n_vars:
        return {
            'success': False,
            'n_evaluated': 0,
            'n_failed': 0,
            'message': f"Mismatch: {len(variables)} variables but {n_vars} columns in samples"
        }
    
    # Evaluation loop
    results = []
    outputs = []
    failed_samples = []
    
    for i, sample in enumerate(samples):
        # Progress callback
        if progress_callback is not None:
            progress_callback(i + 1, n_samples)
        
        # Create value dict for injection
        values = {variable_names[j]: float(sample[j]) for j in range(n_vars)}
        
        # Inject values into config
        try:
            modified_config = inject_single_config(base_config, variables, values)
        except Exception as e:
            failed_samples.append(i)
            results.append({
                'sample_id': i,
                'inputs': values,
                'output': None,
                'success': False,
                'error': f"Injection failed: {str(e)}"
            })
            continue
        
        # Evaluate
        try:
            result = evaluator_callable(modified_config)
            
            # Check if evaluation was successful
            if isinstance(result, dict) and result.get('success', True) is False:
                # Evaluator reported failure
                failed_samples.append(i)
                results.append({
                    'sample_id': i,
                    'inputs': values,
                    'output': None,
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'full_result': result
                })
                continue
            
            # Extract output value
            output_value = result.get(output_key, None)
            
            if output_value is None:
                failed_samples.append(i)
                results.append({
                    'sample_id': i,
                    'inputs': values,
                    'output': None,
                    'success': False,
                    'error': f"Output key '{output_key}' not found or is None in result",
                    'full_result': result
                })
                continue
            
            # Success
            results.append({
                'sample_id': i,
                'inputs': values,
                'output': output_value,
                'success': True,
                'full_result': result
            })
            outputs.append(output_value)
            
        except Exception as e:
            failed_samples.append(i)
            results.append({
                'sample_id': i,
                'inputs': values,
                'output': None,
                'success': False,
                'error': f"Evaluation error: {str(e)}"
            })
    
    # Compute statistics on successful outputs
    n_evaluated = len(outputs)
    n_failed = len(failed_samples)
    
    if n_evaluated == 0:
        return {
            'success': False,
            'n_evaluated': 0,
            'n_failed': n_failed,
            'results': results,
            'outputs': [],
            'statistics': {},
            'failed_samples': failed_samples,
            'message': f"All {n_samples} evaluations failed"
        }
    
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
    
    return {
        'success': True,
        'n_evaluated': n_evaluated,
        'n_failed': n_failed,
        'results': results,
        'outputs': outputs,
        'statistics': statistics,
        'failed_samples': failed_samples,
        'message': f"Successfully evaluated {n_evaluated}/{n_samples} samples" +
                   (f" ({n_failed} failed)" if n_failed > 0 else "")
    }


# ============================================================================
# File-Based Evaluation (Convenience Function)
# ============================================================================

def evaluate_from_files(
    evaluator_callable: Callable[[Dict[str, Any]], Dict[str, Any]],
    samples_file: str,
    workflow_file: str,
    config_file: str,
    output_file: str,
    output_key: str = "output"
) -> Dict[str, Any]:
    """
    Evaluate samples from files (convenience wrapper).
    
    This function:
    1. Loads samples, workflow, and config from files
    2. Reconstructs Variable objects
    3. Calls evaluate_samples
    4. Saves results to output file
    
    Args:
        evaluator_callable: Function that evaluates a config
        samples_file: Path to samples JSON
        workflow_file: Path to workflow JSON (with variables)
        config_file: Path to model config JSON
        output_file: Path to save results JSON
        output_key: Key to extract from evaluator results
    
    Returns:
        Dictionary with evaluation results (same as evaluate_samples)
    
    Example:
        >>> from evaluators import solve_FE_static
        >>> results = evaluate_from_files(
        ...     evaluator_callable=solve_FE_static,
        ...     samples_file="sa_samples.json",
        ...     workflow_file="sa_workflow.json",
        ...     config_file="fe_config.json",
        ...     output_file="sa_results.json",
        ...     output_key="max_displacement"
        ... )
    """
    # Load files
    try:
        with open(samples_file, 'r') as f:
            samples_data = json.load(f)
        
        with open(workflow_file, 'r') as f:
            workflow = json.load(f)
        
        with open(config_file, 'r') as f:
            base_config = json.load(f)
            
    except FileNotFoundError as e:
        return {
            'success': False,
            'n_evaluated': 0,
            'n_failed': 0,
            'message': f"File not found: {str(e)}"
        }
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'n_evaluated': 0,
            'n_failed': 0,
            'message': f"JSON decode error: {str(e)}"
        }
    
    # Extract data
    samples = np.array(samples_data['samples'])
    variable_names = samples_data['variable_names']
    
    # Reconstruct Variable objects
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
    
    # Evaluate
    eval_results = evaluate_samples(
        evaluator_callable=evaluator_callable,
        base_config=base_config,
        samples=samples,
        variables=variables,
        output_key=output_key,
        variable_names=variable_names
    )
    
    if not eval_results['success']:
        return eval_results
    
    # Prepare output data
    results_data = {
        'n_samples': int(len(samples)),
        'n_evaluated': eval_results['n_evaluated'],
        'n_failed': eval_results['n_failed'],
        'n_vars': int(len(variable_names)),
        'variable_names': variable_names,
        'output_key': output_key,
        'results': eval_results['results'],
        'statistics': eval_results['statistics'],
        'failed_samples': eval_results['failed_samples'],
        'created': datetime.now().isoformat()
    }
    
    # Save results
    try:
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        eval_results['results_file'] = output_file
    except Exception as e:
        eval_results['message'] += f" | Warning: Could not save results: {str(e)}"
    
    return eval_results


# ============================================================================
# Progress Helpers
# ============================================================================

def simple_progress(current: int, total: int):
    """Simple progress printer."""
    if current % 10 == 0 or current == total:
        print(f"  Progress: {current}/{total} ({100*current/total:.1f}%)")


def tqdm_progress(current: int, total: int):
    """Progress using tqdm (if available)."""
    try:
        from tqdm import tqdm
        # This would need to be integrated differently
        # Just a placeholder
        pass
    except ImportError:
        simple_progress(current, total)


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Evaluation Framework - Standalone Test")
    print("=" * 70)
    
    # Create dummy data for testing
    print("\nCreating test data...")
    
    # Dummy evaluator
    def dummy_evaluator(config):
        """Dummy evaluator for testing."""
        x = config.get('x', 0)
        y = config.get('y', 0)
        return {
            'output': x**2 + y**2,
            'success': True
        }
    
    # Dummy variables
    var_x = Variable(name="x", kind="uniform", params={"low": 0, "high": 1})
    var_x.add_target(doc="test", path="x")
    
    var_y = Variable(name="y", kind="uniform", params={"low": 0, "high": 1})
    var_y.add_target(doc="test", path="y")
    
    variables = [var_x, var_y]
    
    # Generate samples
    samples = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.9, 1.0]
    ])
    
    # Base config
    base_config = {'x': 0, 'y': 0}
    
    # Evaluate
    print("\nEvaluating 5 samples...")
    results = evaluate_samples(
        evaluator_callable=dummy_evaluator,
        base_config=base_config,
        samples=samples,
        variables=variables,
        output_key="output",
        progress_callback=simple_progress
    )
    
    print(f"\nResults:")
    print(f"  Success: {results['success']}")
    print(f"  Evaluated: {results['n_evaluated']}/{len(samples)}")
    print(f"  Statistics:")
    for key, value in results['statistics'].items():
        print(f"    {key}: {value:.4f}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)

