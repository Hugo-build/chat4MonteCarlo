"""
Test script for variable management in SA server
=================================================

This script demonstrates how the MCP tools work for creating and managing
variables for sensitivity analysis.

Note: This is for local testing. In production, these functions are called
via the MCP protocol from an LLM assistant.
"""
import sys
from pathlib import Path

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.resolve()
PARENT_DIR = str(PROJECT_ROOT.parent)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import the server functions (simulating MCP calls)
from proj0_SA.server import (
    create_variable,
    list_variables,
    delete_variable,
    clear_workflow,
    get_workflow_status
)


def print_result(name: str, result: dict):
    """Pretty print test results"""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print('='*60)
    print(f"Success: {result.get('success')}")
    print(f"Message: {result.get('message')}")
    for key, value in result.items():
        if key not in ['success', 'message']:
            if key == 'variables' and len(str(value)) > 200:
                print(f"{key}: [... {len(value)} items ...]")
            else:
                print(f"{key}: {value}")


def test_variable_management():
    """Test the complete variable management workflow"""
    
    print("\n" + "="*60)
    print("SA SERVER VARIABLE MANAGEMENT TEST")
    print("="*60)
    
    # 1. Clear workflow to start fresh
    result = clear_workflow()
    print_result("Clear Workflow", result)
    
    # 2. Check initial status
    result = get_workflow_status()
    print_result("Initial Status", result)
    
    # 3. Create first variable (Young's modulus)
    result = create_variable(
        name="E",
        kind="uniform",
        params={"low": 200e9, "high": 220e9},
        targets=[{"doc": "FE", "path": "elements[*].E"}]
    )
    print_result("Create Variable: E (Young's Modulus)", result)
    
    # 4. Create second variable (Load)
    result = create_variable(
        name="F_y",
        kind="uniform",
        params={"low": 500.0, "high": 1500.0},
        targets=[{"doc": "FE", "path": "loads[29]"}]
    )
    print_result("Create Variable: F_y (Load)", result)
    
    # 5. Create third variable (Density)
    result = create_variable(
        name="rho",
        kind="uniform",
        params={"low": 7500.0, "high": 8000.0},
        targets=[{"doc": "FE", "path": "elements[*].rho"}]
    )
    print_result("Create Variable: rho (Density)", result)
    
    # 6. Create fourth variable (Cross-sectional area - normal distribution)
    result = create_variable(
        name="A_bottom",
        kind="normal",
        params={"mean": 2.5e-4, "std": 5e-5},
        targets=[
            {"doc": "FE", "path": "elements[0].A"},
            {"doc": "FE", "path": "elements[1].A"},
            {"doc": "FE", "path": "elements[2].A"}
        ]
    )
    print_result("Create Variable: A_bottom (Cross-section)", result)
    
    # 7. Create fixed variable (Poisson's ratio)
    result = create_variable(
        name="nu",
        kind="fixed",
        params={"value": 0.3}
    )
    print_result("Create Variable: nu (Poisson's ratio - fixed)", result)
    
    # 8. List all variables
    result = list_variables()
    print_result("List All Variables", result)
    
    # 9. Update existing variable (change E bounds)
    result = create_variable(
        name="E",
        kind="uniform",
        params={"low": 195e9, "high": 225e9},  # Changed bounds
        targets=[{"doc": "FE", "path": "elements[*].E"}]
    )
    print_result("Update Variable: E (changed bounds)", result)
    
    # 10. Get workflow status
    result = get_workflow_status()
    print_result("Workflow Status", result)
    
    # 11. Test error handling - invalid distribution
    result = create_variable(
        name="invalid",
        kind="unknown_distribution",
        params={"low": 0, "high": 1}
    )
    print_result("Test Error: Invalid Distribution", result)
    
    # 12. Test error handling - missing parameters
    result = create_variable(
        name="incomplete",
        kind="uniform",
        params={"low": 0}  # Missing 'high'
    )
    print_result("Test Error: Missing Parameter", result)
    
    # 13. Delete a variable
    result = delete_variable("nu")
    print_result("Delete Variable: nu", result)
    
    # 14. List variables after deletion
    result = list_variables()
    print_result("List Variables After Deletion", result)
    
    # 15. Final workflow status
    result = get_workflow_status()
    print_result("Final Workflow Status", result)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nWorkflow file location:")
    print(f"  {PROJECT_ROOT / 'sa_workflow.json'}")
    print("\nNext steps:")
    print("  1. Review the generated sa_workflow.json file")
    print("  2. Use SA.generate_samples() to create input samples")
    print("  3. Use SA.evaluate_model() to run simulations")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_variable_management()

