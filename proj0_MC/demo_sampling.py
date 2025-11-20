"""
Demo: Variable Definition and Sample Generation
================================================

This demo shows how to use the SA server to:
1. Define variables for sensitivity analysis
2. Generate samples using different methods
3. Review sample statistics

Note: Run the server with: fastmcp run proj0_SA/server.py
Then interact with it via MCP protocol or use this demo for testing.
"""

import json
from pathlib import Path

# This is just for demonstration - in practice, these would be MCP tool calls
# from an LLM conversation

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_result(result: dict):
    """Pretty print a result dictionary"""
    print(f"Success: {result.get('success')}")
    print(f"Message: {result.get('message')}")
    for key, value in result.items():
        if key not in ['success', 'message']:
            if isinstance(value, (list, dict)) and len(str(value)) > 200:
                print(f"{key}: [{type(value).__name__} with {len(value)} items]")
            else:
                print(f"{key}: {value}")


def demo_workflow():
    """
    Demonstrate the complete variable definition and sampling workflow.
    This simulates what would happen in an LLM conversation using MCP tools.
    """
    
    print_section("SA SERVER - SAMPLING DEMO")
    print("\nThis demo shows the MCP tools for variable definition and sampling.")
    print("In production, an LLM would call these tools via the MCP protocol.\n")
    
    # Define the workflow
    workflow_steps = [
        {
            "step": "1. Define Variables",
            "description": "Create 4 variables for a structural analysis",
            "tools": [
                {
                    "tool": "SA.create_variable",
                    "params": {
                        "name": "E",
                        "kind": "uniform",
                        "params": {"low": 200e9, "high": 220e9},
                        "targets": [{"doc": "FE", "path": "elements[*].E"}]
                    },
                    "explanation": "Young's modulus (Pa)"
                },
                {
                    "tool": "SA.create_variable",
                    "params": {
                        "name": "F_y",
                        "kind": "uniform",
                        "params": {"low": 500, "high": 1500},
                        "targets": [{"doc": "FE", "path": "loads[29]"}]
                    },
                    "explanation": "Applied load (N)"
                },
                {
                    "tool": "SA.create_variable",
                    "params": {
                        "name": "rho",
                        "kind": "uniform",
                        "params": {"low": 7500, "high": 8000},
                        "targets": [{"doc": "FE", "path": "elements[*].rho"}]
                    },
                    "explanation": "Density (kg/m³)"
                },
                {
                    "tool": "SA.create_variable",
                    "params": {
                        "name": "A_bottom",
                        "kind": "normal",
                        "params": {"mean": 2.5e-4, "std": 5e-5}
                    },
                    "explanation": "Cross-sectional area (m²)"
                }
            ]
        },
        {
            "step": "2. Check Workflow Status",
            "description": "Verify variables are defined",
            "tools": [
                {
                    "tool": "SA.get_workflow_status",
                    "params": {},
                    "explanation": "Check that variables_defined = True"
                },
                {
                    "tool": "SA.list_variables",
                    "params": {},
                    "explanation": "List all defined variables"
                }
            ]
        },
        {
            "step": "3. Generate Samples (Random)",
            "description": "Generate 50 samples using random sampling",
            "tools": [
                {
                    "tool": "SA.generate_samples",
                    "params": {"n_samples": 50, "method": "random", "seed": 42},
                    "explanation": "Quick random sampling for testing"
                }
            ]
        },
        {
            "step": "4. Review Sample Statistics",
            "description": "Check sample distribution",
            "tools": [
                {
                    "tool": "SA.get_sample_statistics",
                    "params": {},
                    "explanation": "Compute mean, std, min, max for each variable"
                }
            ]
        },
        {
            "step": "5. Generate Samples (LHS)",
            "description": "Generate 100 samples using Latin Hypercube",
            "tools": [
                {
                    "tool": "SA.generate_samples",
                    "params": {"n_samples": 100, "method": "lhs", "seed": 42},
                    "explanation": "Better space coverage for surrogate training"
                }
            ]
        },
        {
            "step": "6. Generate Samples (Sobol)",
            "description": "Generate 128 samples using Sobol sequence",
            "tools": [
                {
                    "tool": "SA.generate_samples",
                    "params": {"n_samples": 128, "method": "sobol", "seed": 42},
                    "explanation": "Optimal for sensitivity analysis"
                }
            ]
        },
        {
            "step": "7. Preview Samples",
            "description": "Look at first few samples",
            "tools": [
                {
                    "tool": "SA.get_samples",
                    "params": {"max_rows": 5},
                    "explanation": "Preview first 5 samples"
                }
            ]
        },
        {
            "step": "8. Final Status",
            "description": "Check workflow completion",
            "tools": [
                {
                    "tool": "SA.get_workflow_status",
                    "params": {},
                    "explanation": "Verify samples_generated = True"
                }
            ]
        }
    ]
    
    # Print the workflow
    for workflow_step in workflow_steps:
        print_section(workflow_step["step"])
        print(f"\n{workflow_step['description']}\n")
        
        for tool_call in workflow_step["tools"]:
            print(f"\n→ Tool: {tool_call['tool']}")
            print(f"  Purpose: {tool_call['explanation']}")
            print(f"  Parameters:")
            for param_name, param_value in tool_call["params"].items():
                if isinstance(param_value, dict):
                    print(f"    {param_name}:")
                    for k, v in param_value.items():
                        print(f"      {k}: {v}")
                elif isinstance(param_value, list):
                    print(f"    {param_name}: [{len(param_value)} items]")
                else:
                    print(f"    {param_name}: {param_value}")
    
    print_section("DEMO COMPLETE")
    print("\nTo run this workflow in practice:")
    print("1. Start the MCP server:")
    print("   fastmcp run proj0_SA/server.py")
    print("\n2. Connect via MCP client (e.g., Claude Desktop)")
    print("\n3. Ask the LLM to execute these steps:")
    print("   'Create 4 variables for structural analysis and generate 100 samples using LHS'")
    print("\nThe LLM will call the MCP tools shown above to complete the workflow.")
    
    print_section("GENERATED FILES")
    
    project_root = Path(__file__).parent
    workflow_file = project_root / "sa_workflow.json"
    samples_file = project_root / "sa_samples.json"
    
    print(f"\nWorkflow config: {workflow_file}")
    if workflow_file.exists():
        print("  ✓ File exists")
        with open(workflow_file) as f:
            data = json.load(f)
        print(f"  Variables: {data.get('metadata', {}).get('num_vars', 0)}")
        print(f"  Status: {data.get('status')}")
    else:
        print("  ✗ Not yet created (run the server)")
    
    print(f"\nSamples data: {samples_file}")
    if samples_file.exists():
        print("  ✓ File exists")
        with open(samples_file) as f:
            data = json.load(f)
        print(f"  Samples: {data.get('n_samples')}")
        print(f"  Variables: {data.get('n_vars')}")
        print(f"  Method: {data.get('sampling_method')}")
    else:
        print("  ✗ Not yet created (run SA.generate_samples)")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    demo_workflow()

