"""
MCP Server for Surrogate Modeling
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
import os
import numpy as np
import joblib

# ----------------------------------------------------------------------------
# Get project root directory (parent of this server file)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Add parent directory to path for imports from core/
PARENT_DIR = str(PROJECT_ROOT.parent)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ------------------------------------------------------------------------
# Initialize MCP server
mcp = FastMCP[Any]("Surrogate-Modeling")

# Global storage for surrogate pipes (since MCP can't return custom objects)
_PIPE_STORAGE: Dict[str, Any] = {}

# ============================================================================
# Helper Functions for Importing Core Modules
# ============================================================================

"""
Import core modules needed for surrogate modeling operations.
Returns tuple: (Variable, VariableSet, success, message)
"""
try:
    from pySMC import Variable, VariableSet
    from pySMC import sample_inputs
    from pySMC import SurrogatePipe, SurrogatePool, StandardScaler
    from pySMC import GaussianProcess, RBF
    from pySMC.core.GPax import Matern32, Matern52
    from pySMC import optSetup
    import jax.numpy as jnp
except ImportError:
    print("Failed to import core modules")

# Global pool storage 
_POOL: SurrogatePool = None  # Initialized lazily


# ============================================================================
# Helper Functions
# ============================================================================

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate prediction metrics: R², MAE, RMSE.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary with R2, MAE, RMSE
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    return {
        "R2": float(r2),
        "MAE": mae,
        "RMSE": rmse
    }

# ============================================================================
#   Helper Functions
# ============================================================================

def _calc_metrics(y_true, y_pred, y_std=None):
    """Print prediction metrics."""
    r2 = 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    metrics = {
        'R²': float(r2),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE (%)': float(mape),
        'Max Error': float(max_error),
    }
    return metrics





"""
------------------------------------------------------------------------------------------------
Overview of the surrogate modeling workflow:

MC Data → Scalers → GP Model → SurrogatePipe → SurrogatePool
   ↓         ↓         ↓            ↓              ↓
  JSON     JSON      JSON        Folder         Folder
------------------------------------------------------------------------------------------------

┌─────────────────────────────────────────────────────────────────┐
│ DATA LAYER                                                      │
├─────────────────────────────────────────────────────────────────┤
│ ✓ SU_load_initial_MC_results  (existing)                        │
│ ✓ SU_load_training_data       (existing)                        │
│ + SU_train_test_split         (split data)                      │
├─────────────────────────────────────────────────────────────────┤
│ SCALER LAYER                                                    │
├─────────────────────────────────────────────────────────────────┤
│ + SU_fit_STD_scalers           (fit X/Y StandardScalers)        │
│ + SU_load_STD_scalers          (load from JSON)                 │
├─────────────────────────────────────────────────────────────────┤
│ MODEL LAYER                                                     │
├─────────────────────────────────────────────────────────────────┤
│ + SU_create_GPR                (create + fit GP in one step)    │
│ + SU_load_GPR                  (load GP from JSON)              │
├─────────────────────────────────────────────────────────────────┤
│ PIPE LAYER                                                      │
├─────────────────────────────────────────────────────────────────┤
│ + SU_create_pipe              (assemble pipe from parts)        │
│ + SU_save_pipe                (save to folder)                  │
│ + SU_load_pipe                (load from folder)                │
│ + SU_predict                  (make predictions)                │
├─────────────────────────────────────────────────────────────────┤
│ POOL LAYER                                                      │
├─────────────────────────────────────────────────────────────────┤
│ + SU_init_pool                (initialize empty pool)           │
│ + SU_add_to_pool              (add current pipe to pool)        │
│ + SU_save_pool                (save pool folder)                │
│ + SU_load_pool                (load pool folder)                │
│ + SU_pool_summary             (get pool info)                   │
└─────────────────────────────────────────────────────────────────┘

"""

# ============================================================================
# Define MCP Tools
# ============================================================================


@mcp.tool(name="SU_detect_files")
def detect_files(
    directory: str = ".",
    file_extension: str = ".json"
) -> Dict[str, Any]:
    """
    Detect files with a specific extension in a directory.
    
    Args:
        directory: Directory to search (relative to project root)
        file_extension: File extension to filter (e.g., ".json", ".csv")
    
    Returns:
        Dictionary with list of matching files
    """
    try:
        directory_path = PROJECT_ROOT / directory
        if not directory_path.exists():
            return {
                "success": False,
                "files": [],
                "message": f"Directory not found: {directory}"
            }
        
        # Use the file_extension parameter
        files = [
            str(f.relative_to(PROJECT_ROOT)) 
            for f in directory_path.iterdir() 
            if f.is_file() and f.suffix == file_extension
        ]
        
        return {
            "success": True,
            "files": files,
            "message": f"Found {len(files)} {file_extension} files in '{directory}'"
        }
    except Exception as e:
        return {
            "success": False,
            "files": [],
            "message": f"Error scanning directory: {str(e)}"
        }


@mcp.tool(name="SU_load_initial_MC_results")
def load_initial_MC_results(
    input_file: str = "MC_results_init.json",
    output_file: str = "Xy_data_init.json"
) -> Dict[str, Any]:
    """
    Load Monte Carlo results and process them for surrogate model training.
    
    This tool reads MC simulation results and extracts inputs/outputs into
    a format suitable for training surrogate models (like in test_SMC_basics.py).
    
    Args:
        input_file: Path to MC results JSON file (relative to project directory)
        output_file: Path to save processed data (relative to project directory)
    
    Returns:
        Dictionary with:
        - success: bool
        - message: str
        - n_samples: int (number of samples loaded)
        - n_vars: int (number of input variables)
        - variable_names: list of variable names
        - output_key: name of output variable
        - X_shape: shape of input array [n_samples, n_vars]
        - y_shape: shape of output array [n_samples]
        - output_path: path to saved file
    """
    try:
        # Resolve file paths
        input_path = PROJECT_ROOT / input_file
        output_path = PROJECT_ROOT / output_file
        
        # Load MC results
        if not input_path.exists():
            return {
                "success": False,
                "message": f"Input file not found: {input_path}",
                "n_samples": 0,
                "n_vars": 0
            }
        
        with open(input_path, 'r') as f:
            mc_data = json.load(f)
        
        # Extract metadata
        n_samples = mc_data.get("n_samples", 0)
        n_vars = mc_data.get("n_vars", 0)
        variable_names = mc_data.get("variable_names", [])
        output_key = mc_data.get("output_key", "output")
        results = mc_data.get("results", [])
        
        if not results:
            return {
                "success": False,
                "message": "No results found in MC data file",
                "n_samples": 0,
                "n_vars": 0
            }
        
        # Extract X (inputs) and y (outputs) from results
        X = []
        y = []
        
        for sample in results:
            inputs = sample.get("inputs", {})
            output = sample.get("output", None)
            
            # Build input row in order of variable_names
            row = [inputs.get(var_name, 0.0) for var_name in variable_names]
            X.append(row)
            y.append(output)
        
        X = np.array(X)
        y = np.array(y)
        
        # Compute statistics for variable bounds (useful for surrogate setup)
        var_stats = {}
        for i, var_name in enumerate(variable_names):
            var_values = X[:, i]
            var_stats[var_name] = {
                "min": float(np.min(var_values)),
                "max": float(np.max(var_values)),
                "mean": float(np.mean(var_values)),
                "std": float(np.std(var_values))
            }
        
        output_stats = {
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y))
        }
        
        # Create processed data structure
        processed_data = {
            "description": "Processed MC results for surrogate model training",
            "source_file": str(input_file),
            "n_samples": int(len(X)),
            "n_vars": int(n_vars),
            "variable_names": variable_names,
            "output_key": output_key,
            "variable_stats": var_stats,
            "output_stats": output_stats,
            "X": X.tolist(),  # Convert to list for JSON serialization
            "y": y.tolist()
        }
        
        # Save processed data
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        return {
            "success": True,
            "message": f"Successfully processed {len(X)} samples with {n_vars} variables",
            "n_samples": int(len(X)),
            "n_vars": int(n_vars),
            "variable_names": variable_names,
            "output_key": output_key,
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "variable_stats": var_stats,
            "output_stats": output_stats,
            "output_path": str(output_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing MC results: {str(e)}",
            "n_samples": 0,
            "n_vars": 0
        }


@mcp.tool(name="SU_load_training_data")
def load_training_data(
    data_file: str = "Xy_data_init.json"
) -> Dict[str, Any]:
    """
    Load processed training data for surrogate modeling.
    
    This loads the data previously processed by load_mc_results_for_surrogate()
    and returns it in a format ready for training.
    
    Args:
        data_file: Path to processed data JSON file (relative to project directory)
    
    Returns:
        Dictionary with:
        - success: bool
        - message: str
        - n_samples: int
        - n_vars: int  
        - variable_names: list
        - output_key: str
        - X: list (2D array as nested list)
        - y: list (1D array as list)
        - variable_stats: dict with min/max/mean/std per variable
        - output_stats: dict with min/max/mean/std of outputs
    """
    try:
        data_path = PROJECT_ROOT / data_file
        
        if not data_path.exists():
            return {
                "success": False,
                "message": f"Data file not found: {data_path}. Run load_mc_results_for_surrogate() first.",
                "n_samples": 0,
                "n_vars": 0
            }
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        return {
            "success": True,
            "message": f"Loaded training data with {data['n_samples']} samples",
            "n_samples": data["n_samples"],
            "n_vars": data["n_vars"],
            "variable_names": data["variable_names"],
            "output_key": data["output_key"],
            "X": data["X"],
            "y": data["y"],
            "variable_stats": data.get("variable_stats", {}),
            "output_stats": data.get("output_stats", {})
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error loading training data: {str(e)}",
            "n_samples": 0,
            "n_vars": 0
        }


@mcp.tool(name="SU_train_test_split")
def train_test_split(
    data_file: str = "Xy_data_init.json",
    output_file: str = "Xy_data_split.json",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Split data into training and test sets.
    
    Single Responsibility: Only splits data and saves the result.
    
    Args:
        data_file: Path to input data JSON file
        output_file: Path to save split data
        test_size: Fraction of data for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with split info and file path
    """
    try:
        from pySMC.core.DataWash import train_test_split as tts
        
        data_path = PROJECT_ROOT / data_file
        output_path = PROJECT_ROOT / output_file
        
        if not data_path.exists():
            return {"success": False, "message": f"Data file not found: {data_path}"}
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        X = np.array(data["X"])
        y = np.array(data["y"])
        
        X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=random_state)
        
        split_data = {
            "description": "Train/test split data",
            "source_file": data_file,
            "test_size": test_size,
            "random_state": random_state,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_vars": data["n_vars"],
            "variable_names": data["variable_names"],
            "output_key": data["output_key"],
            "X_train": X_train.tolist(),
            "y_train": y_train.tolist(),
            "X_test": X_test.tolist(),
            "y_test": y_test.tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        return {
            "success": True,
            "message": f"Split {len(X)} samples: {len(X_train)} train, {len(X_test)} test",
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "output_path": str(output_path)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error splitting data: {str(e)}"}


# ============================================================================
# SCALER LAYER
# ============================================================================

@mcp.tool(name="SU_fit_STD_scalers")
def fit_STD_scalers(
    data_file: str = "Xy_data_split.json",
    output_dir: str = "scalers"
) -> Dict[str, Any]:
    """
    Fit StandardScalers on training data and save to JSON files.
    
    Single Responsibility: Only fits scalers and saves them.
    
    Args:
        data_file: Path to split data JSON file (with X_train, y_train)
        output_dir: Directory to save scaler JSON files
    
    Returns:
        Dictionary with scaler info and file paths
    """
    try:
        data_path = PROJECT_ROOT / data_file
        scaler_dir = PROJECT_ROOT / output_dir
        scaler_dir.mkdir(parents=True, exist_ok=True)
        
        if not data_path.exists():
            return {"success": False, "message": f"Data file not found: {data_path}"}
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        X_train = np.array(data["X_train"])
        y_train = np.array(data["y_train"])
        
        # Fit scalers
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        
        # Save scalers
        x_scaler.save(scaler_dir / "x_scaler.json")
        y_scaler.save(scaler_dir / "y_scaler.json")
        
        return {
            "success": True,
            "message": f"Fitted and saved scalers for {X_train.shape[1]} input dims",
            "x_scaler_path": str(scaler_dir / "x_scaler.json"),
            "y_scaler_path": str(scaler_dir / "y_scaler.json"),
            "x_mean": x_scaler.mean_.tolist(),
            "x_scale": x_scaler.scale_.tolist(),
            "y_mean": float(y_scaler.mean_[0]),
            "y_scale": float(y_scaler.scale_[0])
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error fitting scalers: {str(e)}"}


@mcp.tool(name="SU_load_STD_scalers")
def load_STD_scalers(
    scaler_dir: str = "scalers"
) -> Dict[str, Any]:
    """
    Load StandardScalers from JSON files.
    
    Single Responsibility: Only loads scalers and returns their info.
    
    Args:
        scaler_dir: Directory containing scaler JSON files
    
    Returns:
        Dictionary with scaler parameters
    """
    try:
        scaler_path = PROJECT_ROOT / scaler_dir
        
        x_scaler_path = scaler_path / "x_scaler.json"
        y_scaler_path = scaler_path / "y_scaler.json"
        
        if not x_scaler_path.exists() or not y_scaler_path.exists():
            return {"success": False, "message": f"Scaler files not found in {scaler_path}"}
        
        x_scaler = StandardScaler.load(x_scaler_path)
        y_scaler = StandardScaler.load(y_scaler_path)
        
        # Store in global for later use
        _PIPE_STORAGE["x_scaler"] = x_scaler
        _PIPE_STORAGE["y_scaler"] = y_scaler
        
        return {
            "success": True,
            "message": "Scalers loaded and stored in memory",
            "x_mean": x_scaler.mean_.tolist(),
            "x_scale": x_scaler.scale_.tolist(),
            "y_mean": float(y_scaler.mean_[0]),
            "y_scale": float(y_scaler.scale_[0])
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error loading scalers: {str(e)}"}


# ============================================================================
# MODEL LAYER
# ============================================================================

@mcp.tool(name="SU_create_GPR")
def create_GPR(
    data_file: str = "Xy_data_split.json",
    scaler_dir: str = "scalers",
    kernel_type: str = "rbf",
    length_scale: float = 0.3,
    noise_std: float = 0.05,
    opt_steps: int = 100,
    opt_lr: float = 0.02,
    output_file: str = "gp_model.json"
) -> Dict[str, Any]:
    """
    Create and fit a Gaussian Process Regression model.
    
    Open/Closed: kernel_type parameter allows extension without code change.
    
    Args:
        data_file: Path to split data JSON file
        scaler_dir: Directory with fitted scalers
        kernel_type: Kernel type ("rbf", "matern32", "matern52")
        length_scale: Initial length scale for kernel
        noise_std: Observation noise standard deviation
        opt_steps: Number of optimization steps
        opt_lr: Learning rate for optimizer
        output_file: Path to save the fitted GP model
    
    Returns:
        Dictionary with model info and metrics
    """
    try:
        data_path = PROJECT_ROOT / data_file
        scaler_path = PROJECT_ROOT / scaler_dir
        output_path = PROJECT_ROOT / output_file
        
        if not data_path.exists():
            return {"success": False, "message": f"Data file not found: {data_path}"}
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        X_train = np.array(data["X_train"])
        y_train = np.array(data["y_train"])
        n_dim = X_train.shape[1]
        
        # Load scalers
        x_scaler = StandardScaler.load(scaler_path / "x_scaler.json")
        y_scaler = StandardScaler.load(scaler_path / "y_scaler.json")
        
        # Scale data
        X_train_scaled = x_scaler.transform(X_train)
        y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
        
        # Create kernel based on type
        signal_std = float(np.std(y_train_scaled))
        length_scales = jnp.ones(n_dim) * length_scale
        
        if kernel_type.lower() == "rbf":
            kernel = RBF.from_params(signal_std=signal_std, length_scale=length_scales)
        elif kernel_type.lower() == "matern32":
            kernel = Matern32.from_params(signal_std=signal_std, length_scale=length_scales)
        elif kernel_type.lower() == "matern52":
            kernel = Matern52.from_params(signal_std=signal_std, length_scale=length_scales)
        else:
            return {"success": False, "message": f"Unknown kernel type: {kernel_type}"}
        
        # Create GP model
        gp = GaussianProcess.from_params(kernel=kernel, noise_std=noise_std)
        
        # Fit with optimization
        opt_config = optSetup(
            optimizer='adam',
            steps=opt_steps,
            lr=opt_lr,
            verbose=False,
            log_every=max(1, opt_steps // 5)
        )
        
        gp_fitted = gp.fit(
            jnp.array(X_train_scaled),
            jnp.array(y_train_scaled),
            opt_config=opt_config
        )
        
        # Save model
        gp_fitted.save(output_path)
        
        # Store in global for later use
        _PIPE_STORAGE["gp_model"] = gp_fitted
        _PIPE_STORAGE["x_scaler"] = x_scaler
        _PIPE_STORAGE["y_scaler"] = y_scaler
        
        return {
            "success": True,
            "message": f"GP model fitted with {kernel_type} kernel on {len(X_train)} samples",
            "kernel_type": kernel_type,
            "n_samples": int(len(X_train)),
            "n_dims": int(n_dim),
            "opt_steps": opt_steps,
            "output_path": str(output_path)
        }
        
    except Exception as e:
        import traceback
        return {"success": False, "message": f"Error creating GP: {str(e)}\n{traceback.format_exc()}"}


@mcp.tool(name="SU_load_GPR")
def load_GPR(
    model_file: str = "gp_model.json"
) -> Dict[str, Any]:
    """
    Load a saved Gaussian Process model.
    
    Single Responsibility: Only loads the GP model into memory.
    
    Args:
        model_file: Path to the saved GP model JSON file
    
    Returns:
        Dictionary with model info
    """
    try:
        model_path = PROJECT_ROOT / model_file
        
        if not model_path.exists():
            return {"success": False, "message": f"Model file not found: {model_path}"}
        
        gp_model, _ = GaussianProcess.load(model_path)
        
        # Store in global
        _PIPE_STORAGE["gp_model"] = gp_model
        
        # Get model info
        n_train = gp_model.X.shape[0] if gp_model.X is not None else 0
        n_dims = gp_model.X.shape[1] if gp_model.X is not None else 0
        
        return {
            "success": True,
            "message": f"GP model loaded with {n_train} training samples",
            "n_samples": int(n_train),
            "n_dims": int(n_dims),
            "model_path": str(model_path)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error loading GP: {str(e)}"}


# ============================================================================
# PIPE LAYER
# ============================================================================

@mcp.tool(name="SU_create_pipe")
def create_pipe(
    data_file: str = "Xy_data_split.json",
    scaler_dir: str = "scalers",
    model_file: str = "gp_model.json",
    pipe_name: str = "default"
) -> Dict[str, Any]:
    """
    Assemble a SurrogatePipe from data, scalers, and model.
    
    Single Responsibility: Only assembles the pipe from existing components.
    
    Args:
        data_file: Path to split data JSON file
        scaler_dir: Directory with fitted scalers
        model_file: Path to the fitted GP model
        pipe_name: Name to store the pipe under
    
    Returns:
        Dictionary with pipe info
    """
    try:
        data_path = PROJECT_ROOT / data_file
        scaler_path = PROJECT_ROOT / scaler_dir
        model_path = PROJECT_ROOT / model_file
        
        # Load all components
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        X_train = np.array(data["X_train"])
        y_train = np.array(data["y_train"])
        X_test = np.array(data["X_test"])
        y_test = np.array(data["y_test"])
        
        x_scaler = StandardScaler.load(scaler_path / "x_scaler.json")
        y_scaler = StandardScaler.load(scaler_path / "y_scaler.json")
        
        gp_model, _ = GaussianProcess.load(model_path)
        
        # Create pipe
        pipe = SurrogatePipe(
            model=gp_model,
            X=np.vstack([X_train, X_test]),
            y=np.concatenate([y_train, y_test]),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            verbose=False
        )
        
        # Store in global
        _PIPE_STORAGE[f"pipe_{pipe_name}"] = pipe
        
        return {
            "success": True,
            "message": f"SurrogatePipe '{pipe_name}' created",
            "pipe_name": pipe_name,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "model_type": pipe._detect_model_type(),
            "has_x_scaler": pipe.x_scaler is not None,
            "has_y_scaler": pipe.y_scaler is not None
        }
        
    except Exception as e:
        import traceback
        return {"success": False, "message": f"Error creating pipe: {str(e)}\n{traceback.format_exc()}"}

def create_pipe_from_copy(
    pipe_name: str = "default",
    copy_from_pipe: str = "default"
) -> Dict[str, Any]:
    """
    Create a new pipe from a copy of an existing pipe.
    
    Single Responsibility: Only creates a new pipe from a copy of an existing pipe.
    
    Args:
        pipe_name: Name of the new pipe
        copy_from_pipe: Name of the pipe to copy from
    """
    try:
        pipe_key = f"pipe_{pipe_name}"
        copy_from_pipe_key = f"pipe_{copy_from_pipe}"
        
        if copy_from_pipe_key not in _PIPE_STORAGE:
            return {"success": False, "message": f"Pipe '{copy_from_pipe}' not found. Create it first."}
        
        pipe = _PIPE_STORAGE[copy_from_pipe_key].copy()
        pipe.name = pipe_name
        _PIPE_STORAGE[pipe_key] = pipe
        
        return {
            "success": True,
            "message": f"Pipe '{pipe_name}' created from copy of '{copy_from_pipe}'",
            "pipe_name": pipe_name
        }
    except Exception as e:
        return {"success": False, "message": f"Error creating pipe from copy: {str(e)}"}


@mcp.tool(name="SU_save_pipe")
def save_pipe(
    pipe_name: str = "default",
    output_dir: str = "pipe"
) -> Dict[str, Any]:
    """
    Save a SurrogatePipe to a folder.
    
    Single Responsibility: Only saves the pipe to disk.
    
    Args:
        pipe_name: Name of the pipe to save
        output_dir: Directory to save the pipe folder
    
    Returns:
        Dictionary with save info
    """
    try:
        pipe_key = f"pipe_{pipe_name}"
        
        if pipe_key not in _PIPE_STORAGE:
            return {"success": False, "message": f"Pipe '{pipe_name}' not found. Create it first."}
        
        pipe = _PIPE_STORAGE[pipe_key]
        output_path = PROJECT_ROOT / output_dir
        
        pipe.save(output_path, as_folder=True, include_data=True)
        
        return {
            "success": True,
            "message": f"Pipe '{pipe_name}' saved to folder",
            "output_path": str(output_path)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error saving pipe: {str(e)}"}


@mcp.tool(name="SU_load_pipe")
def load_pipe(
    pipe_dir: str = "pipe",
    pipe_name: str = "default"
) -> Dict[str, Any]:
    """
    Load a SurrogatePipe from a folder.
    
    Single Responsibility: Only loads the pipe into memory.
    
    Args:
        pipe_dir: Directory containing the saved pipe
        pipe_name: Name to store the loaded pipe under
    
    Returns:
        Dictionary with pipe info
    """
    try:
        pipe_path = PROJECT_ROOT / pipe_dir
        
        if not pipe_path.exists():
            return {"success": False, "message": f"Pipe folder not found: {pipe_path}"}
        
        pipe = SurrogatePipe.load(pipe_path)
        
        # Store in global
        _PIPE_STORAGE[f"pipe_{pipe_name}"] = pipe
        
        n_samples = pipe.X.shape[0] if pipe.X is not None else 0
        n_dims = pipe.X.shape[1] if pipe.X is not None else 0
        
        return {
            "success": True,
            "message": f"Pipe '{pipe_name}' loaded from folder",
            "pipe_name": pipe_name,
            "n_samples": int(n_samples),
            "n_dims": int(n_dims),
            "model_type": pipe._detect_model_type()
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error loading pipe: {str(e)}"}


@mcp.tool(name="SU_predict")
def predict(
    pipe_name: str = "default"
) -> Dict[str, Any]:
    """
    Make predictions using a SurrogatePipe.
    
    Single Responsibility: Only makes predictions.
    
    Args:
        pipe_name: Name of the pipe to use
        X: Input points as 2D list [[x1, x2, ...], ...]
           If None, uses the test data from the pipe
    
    Returns:
        Dictionary with predictions
    """
    try:
        pipe_key = f"pipe_{pipe_name}"
        
        if pipe_key not in _PIPE_STORAGE:
            return {"success": False, "message": f"Pipe '{pipe_name}' not found. Load or create it first."}
        
        pipe = _PIPE_STORAGE[pipe_key]
        
        # Use test data if X not provided
        # if X is None:
        #     if pipe.X_test is None:
        #         return {"success": False, "message": "No input provided and pipe has no test data"}
        #     X_pred = pipe.X_test
        # else:
        #     X_pred = np.array(X)

        X_pred = pipe.X_test
        # Make predictions
        predict_fn = pipe.make_predict_fn()
        y_mean, y_std = predict_fn(X_pred)
        
        # Convert to numpy for JSON serialization
        y_mean_np = np.asarray(y_mean)
        y_std_np = np.asarray(y_std) if y_std is not None else None
        
        result = {
            "success": True,
            "message": f"Predictions made for {len(X_pred)} points",
            "n_points": int(len(X_pred)),
            "y_mean": y_mean_np.tolist(),
            "y_std": y_std_np.tolist() if y_std_np is not None else None
        }
        
        # Calculate metrics if using test data
        if pipe.X_test is not None and pipe.y_test is not None:
            result["metrics"] = calc_metrics(pipe.y_test, y_mean_np)
        
        return result
        
    except Exception as e:
        import traceback
        return {"success": False, "message": f"Error predicting: {str(e)}\n{traceback.format_exc()}"}


# ============================================================================
# POOL LAYER
# ============================================================================

@mcp.tool(name="SU_init_pool")
def init_pool(
    pool_name: str = "default"
) -> Dict[str, Any]:
    """
    Initialize an empty SurrogatePool.
    
    Single Responsibility: Only initializes the pool.
    
    Returns:
        Dictionary with pool status
    """
    try:
        global _POOL
        _POOL = SurrogatePool([])
        
        return {
            "success": True,
            "message": "Empty SurrogatePool initialized",
            "n_pipes": str(0)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error initializing pool: {str(e)}"}


@mcp.tool(name="SU_add_to_pool")
def add_to_pool(
    pipe_name: str = "default"
) -> Dict[str, Any]:
    """
    Add a pipe to the SurrogatePool.
    
    Single Responsibility: Only adds a pipe to the pool.
    
    Args:
        pipe_name: Name of the pipe to add
    
    Returns:
        Dictionary with pool status
    """
    try:
        global _POOL
        
        if _POOL is None:
            _POOL = SurrogatePool([])
        
        pipe_key = f"pipe_{pipe_name}"
        if pipe_key not in _PIPE_STORAGE:
            return {"success": False, "message": f"Pipe '{pipe_name}' not found. Create it first."}
        
        pipe = _PIPE_STORAGE[pipe_key]
        _POOL.add(pipe)
        
        return {
            "success": True,
            "message": f"Pipe '{pipe_name}' added to pool",
            "n_pipes": str(len(_POOL))
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error adding to pool: {str(e)}"}


@mcp.tool(name="SU_save_pool")
def save_pool(
    output_dir: str = "pool"
) -> Dict[str, Any]:
    """
    Save the SurrogatePool to a folder.
    
    Single Responsibility: Only saves the pool to disk.
    
    Args:
        output_dir: Directory to save the pool folder
    
    Returns:
        Dictionary with save info
    """
    try:
        global _POOL
        
        if _POOL is None or len(_POOL) == 0:
            return {"success": False, "message": "Pool is empty. Add pipes first."}
        
        output_path = PROJECT_ROOT / output_dir
        _POOL.save(output_path, include_data=True)
        
        return {
            "success": True,
            "message": f"Pool saved with {len(_POOL)} pipes",
            "n_pipes": str(len(_POOL)),
            "output_path": str(output_path)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error saving pool: {str(e)}"}


@mcp.tool(name="SU_load_pool")
def load_pool(
    pool_dir: str = "pool"
) -> Dict[str, Any]:
    """
    Load a SurrogatePool from a folder.
    
    Single Responsibility: Only loads the pool from disk.
    
    Args:
        pool_dir: Directory containing the saved pool
    
    Returns:
        Dictionary with pool info
    """
    try:
        global _POOL
        
        pool_path = PROJECT_ROOT / pool_dir
        
        if not pool_path.exists():
            return {"success": False, "message": f"Pool folder not found: {pool_path}"}
        
        _POOL = SurrogatePool.load(pool_path)
        
        # Also store individual pipes in _PIPE_STORAGE
        for i, pipe in enumerate(_POOL):
            _PIPE_STORAGE[f"pipe_pool_{i}"] = pipe
        
        return {
            "success": True,
            "message": f"Pool loaded with {len(_POOL)} pipes",
            "n_pipes": str(len(_POOL)),
            "pool_path": str(pool_path)
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error loading pool: {str(e)}"}


@mcp.tool(name="SU_pool_summary")
def pool_summary() -> Dict[str, Any]:
    """
    Get a summary of the current SurrogatePool.
    
    Single Responsibility: Only returns pool information.
    
    Returns:
        Dictionary with pool summary
    """
    try:
        global _POOL
        
        if _POOL is None or len(_POOL) == 0:
            return {
                "success": True,
                "message": "Pool is empty",
                "n_pipes": 0,
                "pipes": []
            }
        
        summary = _POOL.to_summary()
        
        return {
            "success": True,
            "message": f"Pool contains {len(_POOL)} pipes",
            "n_pipes": summary["num_surrogates"],
            "pipes": summary["pipes"]
        }
        
    except Exception as e:
        return {"success": False, "message": f"Error getting pool summary: {str(e)}"}





# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
