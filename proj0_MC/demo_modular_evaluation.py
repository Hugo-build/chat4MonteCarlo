"""
Demonstration of Modular Evaluation Framework
==============================================

This script demonstrates the new modular architecture where:
1. Evaluators are standalone callable functions (evaluators.py)
2. Evaluation framework is general and model-agnostic (evaluate.py)
3. Clean separation between model logic and orchestration

Usage:
    python demo_modular_evaluation.py
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from new modular structure
from evaluators import solve_FE_static, get_evaluator, list_evaluators
from evaluate import evaluate_samples, evaluate_from_files
from core.Variables import Variable
from core.Samplers import sample_inputs, VariableSet


def demo_direct_usage():
    """Demo 1: Direct usage of evaluator and framework"""
    print("\n" + "="*70)
    print("DEMO 1: Direct Usage (No Files)")
    print("="*70)
    
    # Check if FE config exists
    fe_config_path = Path("fe_config.json")
    if not fe_config_path.exists():
        print("\n⚠ Warning: fe_config.json not found. Skipping this demo.")
        print("Copy fe_config.json from proj0_FE or examples/ to run this demo.")
        return
    
    # Load base config
    with open(fe_config_path, 'r') as f:
        base_config = json.load(f)
    
    print("\n1. Define variables with targets:")
    E_var = Variable(name="E", kind="uniform", params={"low": 200e3, "high": 220e3})
    E_var.add_target(doc="FE", path="elements[*].E")
    
    F_var = Variable(name="F_y", kind="uniform", params={"low": 800, "high": 1200})
    F_var.add_target(doc="FE", path="loads[29]")
    
    variables = [E_var, F_var]
    var_set = VariableSet(variables=variables)
    
    print(f"   • E: uniform({E_var.params['low']}, {E_var.params['high']})")
    print(f"   • F_y: uniform({F_var.params['low']}, {F_var.params['high']})")
    
    # Generate samples
    print("\n2. Generate samples:")
    n_samples = 20
    samples = sample_inputs(var_set, n_samples, kind="sobol", seed=42)
    print(f"   Generated {n_samples} Sobol samples")
    
    # Get evaluator
    print("\n3. Get evaluator function:")
    evaluator = get_evaluator("FE_static")
    print(f"   Using: {evaluator.__name__}")
    
    # Evaluate
    print("\n4. Evaluate samples:")
    results = evaluate_samples(
        evaluator_callable=evaluator,
        base_config=base_config,
        samples=samples,
        variables=variables,
        output_key="max_displacement"
    )
    
    print(f"\n✓ Results:")
    print(f"   Success: {results['success']}")
    print(f"   Evaluated: {results['n_evaluated']}/{n_samples}")
    if results['n_failed'] > 0:
        print(f"   Failed: {results['n_failed']}")
    
    print(f"\n   Statistics (max_displacement):")
    for key, value in results['statistics'].items():
        print(f"     {key}: {value:.6e}")


def demo_file_based():
    """Demo 2: File-based evaluation (convenience wrapper)"""
    print("\n" + "="*70)
    print("DEMO 2: File-Based Evaluation")
    print("="*70)
    
    # Check if required files exist
    required_files = ["sa_samples.json", "sa_workflow.json", "fe_config.json"]
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print(f"\n⚠ Warning: Missing files: {missing}")
        print("Generate them first with SA.generate_samples() or demo_evaluation.py")
        return
    
    print("\n1. List available evaluators:")
    evaluators = list_evaluators()
    print(f"   {', '.join(evaluators)}")
    
    print("\n2. Select evaluator:")
    evaluator = get_evaluator("FE_static")
    print(f"   Using: {evaluator.__name__}")
    
    print("\n3. Evaluate from files:")
    results = evaluate_from_files(
        evaluator_callable=evaluator,
        samples_file="sa_samples.json",
        workflow_file="sa_workflow.json",
        config_file="fe_config.json",
        output_file="sa_results_modular.json",
        output_key="max_displacement"
    )
    
    print(f"\n✓ Results:")
    print(f"   Success: {results['success']}")
    print(f"   Evaluated: {results['n_evaluated']}")
    print(f"   Results saved to: {results.get('results_file', 'N/A')}")
    
    if results['success']:
        print(f"\n   Statistics:")
        for key, value in results['statistics'].items():
            print(f"     {key}: {value:.6e}")


def demo_custom_evaluator():
    """Demo 3: Custom evaluator function"""
    print("\n" + "="*70)
    print("DEMO 3: Custom Evaluator")
    print("="*70)
    
    print("\n1. Define custom evaluator:")
    def my_custom_model(config):
        """Custom model: y = a*x^2 + b*x + c"""
        a = config.get('a', 1.0)
        b = config.get('b', 0.0)
        c = config.get('c', 0.0)
        x = config.get('x', 0.0)
        
        y = a * x**2 + b * x + c
        
        return {
            'output': y,
            'coefficients': {'a': a, 'b': b, 'c': c},
            'input': x,
            'success': True
        }
    
    print("   def my_custom_model(config):")
    print("       return y = a*x^2 + b*x + c")
    
    print("\n2. Define variables:")
    var_a = Variable(name="a", kind="uniform", params={"low": 0.5, "high": 1.5})
    var_a.add_target(doc="custom", path="a")
    
    var_b = Variable(name="b", kind="uniform", params={"low": -1.0, "high": 1.0})
    var_b.add_target(doc="custom", path="b")
    
    var_c = Variable(name="c", kind="fixed", params={"value": 2.0})
    var_c.add_target(doc="custom", path="c")
    
    var_x = Variable(name="x", kind="uniform", params={"low": 0.0, "high": 10.0})
    var_x.add_target(doc="custom", path="x")
    
    variables = [var_a, var_b, var_c, var_x]
    print("   4 variables: a, b, c, x")
    
    print("\n3. Generate samples:")
    var_set = VariableSet(variables=variables)
    samples = sample_inputs(var_set, 30, kind="sobol", seed=42)
    print(f"   Generated {len(samples)} samples")
    
    print("\n4. Evaluate with custom model:")
    base_config = {'a': 1.0, 'b': 0.0, 'c': 0.0, 'x': 0.0}
    
    results = evaluate_samples(
        evaluator_callable=my_custom_model,
        base_config=base_config,
        samples=samples,
        variables=variables,
        output_key="output"
    )
    
    print(f"\n✓ Results:")
    print(f"   Success: {results['success']}")
    print(f"   Evaluated: {results['n_evaluated']}/{len(samples)}")
    
    print(f"\n   Statistics (y = a*x^2 + b*x + c):")
    for key, value in results['statistics'].items():
        print(f"     {key}: {value:.4f}")


def demo_evaluator_features():
    """Demo 4: Evaluator features (testing, validation, etc.)"""
    print("\n" + "="*70)
    print("DEMO 4: Evaluator Features")
    print("="*70)
    
    print("\n1. Test evaluator directly:")
    # Create minimal test config
    test_config = {
        'nodes': [
            {'id': 1, 'coords': [0, 0, 0]},
            {'id': 2, 'coords': [1000, 0, 0]}
        ],
        'elements': [
            {'id': 1, 'node_i': 1, 'node_j': 2, 'E': 210e3, 'A': 100, 'rho': 7.85e-6}
        ],
        'fixed_dofs': [0, 1, 2],
        'loads': [0, 0, 0, 1000, 0, 0]
    }
    
    print("   Testing with 2-node truss...")
    result = solve_FE_static(test_config)
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Max displacement: {result['max_displacement']:.6e}")
    else:
        print(f"   Error: {result['error']}")
    
    print("\n2. Test with invalid config:")
    invalid_config = {'nodes': []}  # Missing required fields
    result = solve_FE_static(invalid_config)
    print(f"   Success: {result['success']}")
    print(f"   Error message: {result['error']}")
    
    print("\n3. Available evaluators:")
    evaluators = list_evaluators()
    for name in evaluators:
        evaluator = get_evaluator(name)
        print(f"   • {name}: {evaluator.__name__}")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("MODULAR EVALUATION FRAMEWORK DEMOS")
    print("="*70)
    print("\nThis demo shows the new modular architecture:")
    print("  • evaluators.py - Standalone model execution functions")
    print("  • evaluate.py - General evaluation framework")
    print("  • Clean separation of concerns")
    
    # Run demos
    try:
        demo_direct_usage()
    except Exception as e:
        print(f"\n❌ Demo 1 error: {e}")
    
    try:
        demo_file_based()
    except Exception as e:
        print(f"\n❌ Demo 2 error: {e}")
    
    try:
        demo_custom_evaluator()
    except Exception as e:
        print(f"\n❌ Demo 3 error: {e}")
    
    try:
        demo_evaluator_features()
    except Exception as e:
        print(f"\n❌ Demo 4 error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nKey advantages of modular design:")
    print("  ✓ Evaluators are standalone and testable")
    print("  ✓ Framework is model-agnostic")
    print("  ✓ Easy to add custom evaluators")
    print("  ✓ Clean separation of concerns")
    print("  ✓ Reusable across projects")
    
    print("\nUsage patterns:")
    print("  1. Direct: evaluate_samples(callable, config, samples, ...)")
    print("  2. File-based: evaluate_from_files(callable, files, ...)")
    print("  3. Custom: Define your own evaluator function")
    
    print("\nFiles created:")
    print("  • evaluators.py - Model execution functions")
    print("  • evaluate.py - Evaluation framework")
    print("  • demo_modular_evaluation.py - This demo")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

