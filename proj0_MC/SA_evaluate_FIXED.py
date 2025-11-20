"""
FIXED VERSION OF SA.evaluate FUNCTION
======================================

Replace the broken SA.evaluate function in server.py with this clean version.

Instructions:
1. Find the @mcp.tool(name="SA.evaluate") function in server.py (around line 709)
2. Delete the entire function (from @mcp.tool to the end of the function)
3. Copy and paste this function in its place
"""

@mcp.tool(name="SA.evaluate")
def evaluate(
    evaluator: str = "FE_static",
    model_config_path: str = "fe_config.json",
    samples_file: str = "sa_samples.json",
    workflow_file: str = "sa_workflow.json",
    output_file: str = "sa_results.json",
    output_key: str = "max_displacement"
) -> Dict[str, Any]:
    """
    Evaluate model for all generated samples using modular evaluator framework.
    
    This tool uses the new modular architecture:
    - Evaluators are standalone functions in evaluators.py
    - General evaluation framework in evaluate.py
    - Clean separation of model logic and orchestration
    
    Args:
        evaluator: Name of evaluator to use (e.g., "FE_static", "surrogate", "custom")
                  Available evaluators can be listed with SA.list_evaluators()
        model_config_path: Path to model configuration file (relative to project root)
        samples_file: Path to samples JSON file (from SA.generate_samples)
        workflow_file: Path to workflow JSON file (contains variable definitions)
        output_file: Path to save results
        output_key: Key to extract from evaluator results (e.g., "max_displacement")
    
    Returns:
        Dictionary with:
        - success: boolean
        - n_evaluated: number of samples evaluated
        - n_failed: number of failed evaluations
        - results_file: path to saved results
        - statistics: summary statistics
        - evaluator: evaluator name used
        - message: status message
    
    Example:
        >>> SA.evaluate(
        ...     evaluator="FE_static",
        ...     model_config_path="fe_config.json",
        ...     output_key="max_displacement"
        ... )
    """
    try:
        # Import modular evaluation framework
        sys.path.insert(0, str(PROJECT_ROOT))
        from evaluators import get_evaluator, list_evaluators
        from evaluate import evaluate_from_files
        
        # Get evaluator callable from registry
        try:
            evaluator_callable = get_evaluator(evaluator)
        except ValueError as e:
            available = ", ".join(list_evaluators())
            return {
                "success": False,
                "n_evaluated": 0,
                "n_failed": 0,
                "results_file": None,
                "statistics": {},
                "message": f"{str(e)}. Available: {available}"
            }
        
        # Validate file paths
        samples_path = PROJECT_ROOT / samples_file
        workflow_path = PROJECT_ROOT / workflow_file
        config_path = PROJECT_ROOT / model_config_path
        
        if not samples_path.exists():
            return {
                "success": False,
                "n_evaluated": 0,
                "n_failed": 0,
                "results_file": None,
                "statistics": {},
                "message": f"Samples file not found: {samples_file}. Run SA.generate_samples first."
            }
        
        if not workflow_path.exists():
            return {
                "success": False,
                "n_evaluated": 0,
                "n_failed": 0,
                "results_file": None,
                "statistics": {},
                "message": f"Workflow file not found: {workflow_file}. Define variables first."
            }
        
        if not config_path.exists():
            return {
                "success": False,
                "n_evaluated": 0,
                "n_failed": 0,
                "results_file": None,
                "statistics": {},
                "message": f"Model config file not found: {model_config_path}"
            }
        
        # Use modular evaluation framework
        eval_results = evaluate_from_files(
            evaluator_callable=evaluator_callable,
            samples_file=str(samples_path),
            workflow_file=str(workflow_path),
            config_file=str(config_path),
            output_file=str(PROJECT_ROOT / output_file),
            output_key=output_key
        )
        
        if not eval_results['success']:
            return {
                "success": False,
                "n_evaluated": eval_results.get('n_evaluated', 0),
                "n_failed": eval_results.get('n_failed', 0),
                "results_file": None,
                "statistics": {},
                "evaluator": evaluator,
                "message": eval_results.get('message', 'Evaluation failed')
            }
        
        # Update workflow status
        from datetime import datetime
        with open(workflow_path, 'r') as f:
            workflow = json.load(f)
        
        workflow["stages"]["model_evaluated"] = True
        workflow["metadata"]["n_evaluated"] = eval_results['n_evaluated']
        workflow["metadata"]["n_failed"] = eval_results.get('n_failed', 0)
        workflow["metadata"]["results_file"] = output_file
        workflow["metadata"]["evaluator"] = evaluator
        workflow["metadata"]["updated"] = datetime.now().isoformat()
        
        with open(workflow_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        return {
            "success": True,
            "n_evaluated": eval_results['n_evaluated'],
            "n_failed": eval_results.get('n_failed', 0),
            "results_file": eval_results.get('results_file', output_file),
            "statistics": eval_results['statistics'],
            "evaluator": evaluator,
            "output_key": output_key,
            "message": eval_results['message']
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "n_evaluated": 0,
            "n_failed": 0,
            "results_file": None,
            "statistics": {},
            "message": f"Error during evaluation: {str(e)}",
            "traceback": traceback.format_exc()
        }

