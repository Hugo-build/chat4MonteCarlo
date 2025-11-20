"""
Demo script for testing the evaluation workflow.

This script demonstrates the end-to-end evaluation process:
1. Create variables with targets
2. Generate samples
3. Evaluate using FE simulation
4. Retrieve and analyze results

Usage:
    python demo_evaluation.py
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.Variables import Variable, VariableSet
from core.Samplers import sample_inputs

def main():
    print("="*70)
    print("EVALUATION WORKFLOW DEMO")
    print("="*70)
    
    # Check if FE config exists
    fe_config_path = Path("fe_config.json")
    if not fe_config_path.exists():
        print("\n❌ ERROR: fe_config.json not found!")
        print("Please run this from proj0_SA directory with a valid FE config.")
        print("You may need to copy fe_config.json from proj0_FE or examples.")
        return
    
    # ========================================================================
    # Step 1: Define Variables with Targets
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Define Variables")
    print("-"*70)
    
    # Variable 1: Young's modulus - apply to all elements
    E_var = Variable(
        name="E",
        kind="uniform",
        params={"low": 200e3, "high": 220e3}
    )
    E_var.add_target(doc="FE", path="elements[*].E")
    
    # Variable 2: Applied load - single load component
    load_var = Variable(
        name="F_y",
        kind="uniform",
        params={"low": 800, "high": 1200}
    )
    load_var.add_target(doc="FE", path="loads[29]")
    
    variables = [E_var, load_var]
    var_set = VariableSet(variables=variables)
    
    print(f"✓ Defined {len(variables)} variables:")
    for var in variables:
        print(f"  • {var.name}: {var.kind} {var.params}")
        print(f"    Targets: {len(var.targets)} location(s)")
    
    # ========================================================================
    # Step 2: Save Variables to Workflow
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Save Variables to Workflow")
    print("-"*70)
    
    from datetime import datetime
    workflow = {
        "workflow_id": "demo_evaluation_001",
        "status": "in_progress",
        "stages": {
            "variables_defined": True,
            "samples_generated": False,
            "model_evaluated": False,
            "surrogate_trained": False,
            "sa_computed": False
        },
        "variables": [
            {
                "name": var.name,
                "kind": var.kind,
                "params": var.params,
                "targets": var.targets
            } for var in variables
        ],
        "metadata": {
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "num_vars": len(variables)
        }
    }
    
    workflow_path = Path("sa_workflow.json")
    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"✓ Saved workflow to {workflow_path}")
    
    # ========================================================================
    # Step 3: Generate Samples
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 3: Generate Samples")
    print("-"*70)
    
    n_samples = 50
    samples = sample_inputs(var_set, n_samples, kind="sobol", seed=42)
    
    samples_data = {
        "samples": samples.tolist(),
        "n_samples": int(samples.shape[0]),
        "n_vars": int(samples.shape[1]),
        "variable_names": [v.name for v in variables],
        "sampling_method": "sobol",
        "seed": 42,
        "created": datetime.now().isoformat(),
        "bounds": {
            "lower": var_set.bounds()[0].tolist(),
            "upper": var_set.bounds()[1].tolist()
        }
    }
    
    samples_path = Path("sa_samples.json")
    with open(samples_path, 'w') as f:
        json.dump(samples_data, f, indent=2)
    
    print(f"✓ Generated {n_samples} samples using Sobol sequence")
    print(f"  Saved to {samples_path}")
    print(f"  Sample range:")
    print(f"    {variables[0].name}: [{samples[:,0].min():.1f}, {samples[:,0].max():.1f}]")
    print(f"    {variables[1].name}: [{samples[:,1].min():.1f}, {samples[:,1].max():.1f}]")
    
    # Update workflow
    workflow["stages"]["samples_generated"] = True
    workflow["metadata"]["n_samples"] = n_samples
    workflow["metadata"]["sampling_method"] = "sobol"
    workflow["metadata"]["updated"] = datetime.now().isoformat()
    
    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    # ========================================================================
    # Step 4: Evaluate Samples
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 4: Evaluate Samples (FE Simulation)")
    print("-"*70)
    print(f"Evaluating {n_samples} samples...")
    print("(This may take a moment for FE simulations)")
    
    # Load FE config
    with open(fe_config_path, 'r') as f:
        base_config = json.load(f)
    
    # Import FE solver
    from FElib import elMatrixBar6DoF, rotateMat
    from core.Variables import inject_single_config
    
    # Evaluation function
    def evaluate_fe(config):
        """Evaluate FE config and return max displacement"""
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
        
        return float(np.max(np.abs(U)))
    
    # Evaluate all samples
    results = []
    outputs = []
    
    for i, sample in enumerate(samples):
        # Create value dict
        values = {variables[j].name: float(sample[j]) for j in range(len(variables))}
        
        # Inject values into config
        modified_config = inject_single_config(base_config, variables, values)
        
        # Evaluate
        try:
            output = evaluate_fe(modified_config)
            results.append({
                'sample_id': i,
                'inputs': values,
                'output': output
            })
            outputs.append(output)
        except Exception as e:
            print(f"  ❌ Error evaluating sample {i}: {e}")
            break
    
    print(f"✓ Successfully evaluated {len(results)} samples")
    
    # ========================================================================
    # Step 5: Save Results and Compute Statistics
    # ========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Save Results and Statistics")
    print("-"*70)
    
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
    
    results_data = {
        'n_samples': len(results),
        'n_vars': len(variables),
        'variable_names': [v.name for v in variables],
        'output_key': 'max_displacement',
        'evaluator': 'simulation',
        'model_type': 'FE_static',
        'results': results,
        'statistics': statistics,
        'created': datetime.now().isoformat()
    }
    
    results_path = Path("sa_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"✓ Saved results to {results_path}")
    print(f"\n  Statistics (max_displacement):")
    print(f"    Mean:   {statistics['mean']:.6e} mm")
    print(f"    Std:    {statistics['std']:.6e} mm")
    print(f"    Min:    {statistics['min']:.6e} mm")
    print(f"    Max:    {statistics['max']:.6e} mm")
    print(f"    Median: {statistics['median']:.6e} mm")
    
    # Update workflow
    workflow["stages"]["model_evaluated"] = True
    workflow["metadata"]["n_evaluated"] = len(results)
    workflow["metadata"]["results_file"] = "sa_results.json"
    workflow["metadata"]["updated"] = datetime.now().isoformat()
    
    with open(workflow_path, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print("\n" + "="*70)
    print("✓ EVALUATION WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  • {workflow_path} - Workflow state")
    print(f"  • {samples_path} - Input samples")
    print(f"  • {results_path} - Evaluation results")
    print(f"\nNext steps:")
    print(f"  1. Train surrogate: SA.train_surrogate()")
    print(f"  2. Compute Sobol indices: SA.compute_sobol()")
    print(f"  3. Visualize results with plotting tools")

if __name__ == "__main__":
    main()

