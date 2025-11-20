# Sensitivity Analysis (SA) MCP Server
## Project Directory: proj0_SA

This server provides tools for sensitivity analysis workflows, following the same abstraction pattern as the FE server (`proj0_FE`).

---

## Overview

The SA server abstracts the sensitivity analysis workflow from `examples/exp_40barTruss.py` into reusable MCP tools. It manages the complete workflow:

1. **Variable Definition** - Define uncertain parameters
2. **Sampling** - Generate input samples
3. **Model Evaluation** - Run model for all samples
4. **Surrogate Modeling** - Train fast approximation models
5. **Sensitivity Analysis** - Compute sensitivity indices

---

## Data Structures

All data is stored in JSON files for persistence and MCP compatibility. See `SA_DATA_STRUCTURES.md` for detailed specifications.

### Key Files

```
proj0_SA/
├── server.py                         # MCP server implementation ✅
├── sa_workflow.json                  # Workflow state & variables ✅
├── sa_samples.json                   # Generated input samples ✅
├── sa_results.json                   # Model evaluation outputs (TODO)
├── sa_surrogate.json                 # Surrogate model config (TODO)
├── sa_indices.json                   # Sensitivity indices (TODO)
├── VARIABLE_MANAGEMENT_GUIDE.md      # Variable guide ✅
├── SAMPLING_GUIDE.md                 # Sampling guide ✅
├── IMPLEMENTATION_SUMMARY.md         # Implementation status ✅
├── demo_sampling.py                  # Demo script ✅
├── SA_DATA_STRUCTURES.md             # Data structure specs
├── DATA_STRUCTURE_COMPARISON.md      # Comparison with FE server
└── README.md                         # This file
```

---

## Quick Start

### 1. Start the MCP Server

```bash
cd proj0_SA
fastmcp run server.py
```

### 2. Connect via MCP Client

Connect using Claude Desktop or another MCP client. The server will be available as "SA-Analysis".

### 3. Create Variables

```
User: "Create three variables for my structural analysis:
- Young's modulus E: uniform from 200 to 220 GPa
- Load F_y: uniform from 500 to 1500 N  
- Density rho: uniform from 7500 to 8000 kg/m³"
```

The LLM will call `SA.create_variable()` for each.

### 4. Generate Samples

```
User: "Generate 100 samples using Latin Hypercube sampling with seed 42"
```

The LLM will call `SA.generate_samples(n_samples=100, method="lhs", seed=42)`.

### 5. Check Status

```
User: "What's the status of my workflow?"
```

The LLM will call `SA.get_workflow_status()` and report progress.

### 6. View Sample Statistics

```
User: "Show me the statistics for the generated samples"
```

The LLM will call `SA.get_sample_statistics()` and present the results.

For detailed guides, see:
- **VARIABLE_MANAGEMENT_GUIDE.md** - Complete variable management guide
- **SAMPLING_GUIDE.md** - Sampling strategies and examples
- **demo_sampling.py** - Runnable demonstration

---

## Tool Categories

### 1. Variable Management ✅ IMPLEMENTED
- `SA.create_variable` - Define a new uncertain variable
- `SA.list_variables` - List all defined variables
- `SA.delete_variable` - Remove a variable from workflow
- `SA.clear_workflow` - Reset workflow to start fresh
- `SA.get_workflow_status` - Check workflow progress

### 2. Sampling ✅ IMPLEMENTED
- `SA.generate_samples` - Generate input samples (random/Sobol/LHS)
- `SA.get_samples` - Retrieve generated samples
- `SA.get_sample_statistics` - Get sample statistics

### 3. Model Evaluation ✅ IMPLEMENTED
- `SA.evaluate` - Evaluate model for all samples (simulation or surrogate)
- `SA.get_results` - Retrieve evaluation results with statistics

### 4. Surrogate Modeling
- `SA.train_surrogate` - Train GP or PCE surrogate
- `SA.predict_surrogate` - Make predictions
- `SA.evaluate_surrogate` - Get performance metrics
- `SA.save_surrogate` - Save surrogate config

### 5. Sensitivity Analysis
- `SA.compute_sobol` - Compute Sobol indices
- `SA.compute_morris` - Compute Morris indices
- `SA.plot_indices` - Visualize SA results
- `SA.save_indices` - Save SA results

### 6. Workflow Management ✅ IMPLEMENTED (Basic)
- `SA.get_workflow_status` - Check progress
- `SA.clear_workflow` - Reset workflow
- `SA.get_project_info` - Get project information

---

## Usage Example

### Step 1: Define Variables

```python
# Create variables for FE model
SA.create_variable(
    name="E",
    kind="uniform",
    params={"low": 2.0e9, "high": 2.2e9}
)

SA.create_variable(
    name="F_y",
    kind="uniform",
    params={"low": 500, "high": 1500}
)

# Add targets for config injection
SA.add_target(
    variable_name="E",
    doc="FE",
    path="elements[*].E"
)

SA.add_target(
    variable_name="F_y",
    doc="FE",
    path="loads[29]"
)
```

### Step 2: Generate Samples

```python
SA.generate_samples(
    n_samples=1000,
    method="random",
    seed=42
)
# Creates: workflows/<id>/sa_samples.json
```

### Step 3: Evaluate Model

```python
SA.evaluate_model(
    model_type="FE_static",
    model_config_path="../proj0_FE/fe_config.json"
)
# Creates: workflows/<id>/sa_results.json
```

### Step 4: Train Surrogate

```python
SA.train_surrogate(
    model_type="GaussianProcess",
    kernel_type="RBF",
    train_config={
        "optimizer": "adam",
        "steps": 200,
        "lr": 0.01
    }
)
# Creates: 
#   workflows/<id>/sa_surrogate.json
#   workflows/<id>/sa_surrogate_model.pkl
```

### Step 5: Compute Sensitivity Indices

```python
SA.compute_sobol(
    n_samples=10000,
    use_surrogate=True,
    seed=123
)
# Creates: workflows/<id>/sa_indices.json
# Returns: {"S1": [...], "ST": [...], ...}
```

---

## Abstraction Pattern

This server follows the same abstraction pattern as `proj0_FE/server.py`:

### FE Server Pattern
```
CSV files → Load → Python dicts → Save → JSON config
                                    ↓
                                FE.solve_static()
                                    ↓
                                Displacements
```

### SA Server Pattern
```
Variables → Sample → Evaluate → Train Surrogate → Compute SA
    ↓          ↓         ↓             ↓              ↓
 JSON       JSON      JSON          JSON+PKL       JSON
```

### Key Principles

1. **JSON-Serializable Types Only**
   - Parameters: `str`, `int`, `float`, `bool`, `Dict`, `List`
   - Returns: same types
   - NO numpy arrays, custom objects in tool signatures

2. **File-Based Persistence**
   - All state saved to JSON files
   - Models saved separately as pickle
   - No in-memory state between tool calls

3. **Separate Tools for Separate Operations**
   - Each step is a separate tool
   - Tools can be called independently
   - Workflow can be paused/resumed

4. **Type Conversions**
   | Original | MCP Server |
   |----------|------------|
   | `np.ndarray` | `List[float]` |
   | `Variable` | `Dict[str, Any]` |
   | `VariableSet` | `List[Dict]` |
   | `GaussianProcess` | Config dict + pickle |

---

## Integration with Other Models

The SA server can integrate with any model that follows the config injection pattern:

### FE Model Integration
```python
# Variables target FE config
variables = [
    {
        "name": "E",
        "targets": [{"doc": "FE", "path": "elements[*].E"}]
    }
]

# Evaluation uses FE solver
SA.evaluate_model(
    model_type="FE_static",
    model_config_path="../proj0_FE/fe_config.json"
)
```

### Custom Model Integration
```python
# Variables target custom config
variables = [
    {
        "name": "param1",
        "targets": [{"doc": "model", "path": "parameters.param1"}]
    }
]

# Evaluation uses custom solver
SA.evaluate_model(
    model_type="custom",
    model_config_path="./my_model_config.json"
)
```

---

## Implementation Status

- ✅ Data structures identified
- ✅ Server skeleton created
- ✅ Tool signatures defined
- ✅ **Variable management implementation** (Phase 1 - COMPLETE)
- ✅ **Sampling implementation** (Phase 2 - COMPLETE)
- ✅ **Workflow management implementation** (Basic - COMPLETE)
- ✅ **Documentation with examples** (Variable & Sampling - COMPLETE)
- ⬜ Model evaluation implementation (Phase 3 - TODO)
- ⬜ Surrogate modeling implementation (Phase 4 - TODO)
- ⬜ SA computation implementation (Phase 5 - TODO)
- ⬜ Integration tests (TODO)
- ⬜ Visualization tools (TODO)

---

## References

- **Source Example**: `examples/exp_40barTruss.py`
- **FE Server**: `proj0_FE/server.py`
- **Core Classes**: 
  - `core/Variables.py` - Variable and VariableSet
  - `core/GPax.py` - Gaussian Process
  - `core/DataWash.py` - Data scaling utilities
- **Documentation**:
  - `SA_DATA_STRUCTURES.md` - Complete data structure specs
  - `DATA_STRUCTURE_COMPARISON.md` - Comparison with FE server

---

## Next Steps

1. Implement variable management tools
2. Implement sampling tools (with DoEs integration)
3. Implement model evaluation dispatcher
4. Implement surrogate training wrapper
5. Implement SA computation tools
6. Add workflow management layer
7. Create example notebooks
8. Write integration tests

---

## Design Notes

### Why File-Based Persistence?

MCP servers are stateless - each tool call is independent. File-based persistence allows:
- Resume workflows after interruption
- Share state between different MCP clients
- Version control of workflow configurations
- Easy debugging and inspection

### Why Separate Config and Model Files?

- **Config (JSON)**: Human-readable, version-controllable, MCP-compatible
- **Model (pickle)**: Binary, efficient, contains trained model state
- **Separation**: Allows inspecting config without loading heavy models

### Why Not Return Model Objects?

MCP requires JSON-serializable returns. Instead:
- Return file paths to saved models
- Return performance metrics as dicts
- Client loads models when needed

---

## Comparison Table

| Feature | exp_40barTruss.py | SA Server |
|---------|-------------------|-----------|
| Variables | `Variable` objects | JSON dicts |
| Samples | `np.ndarray` | JSON lists |
| Results | List of dicts | JSON file |
| Surrogate | `GaussianProcess` | JSON config + pickle |
| SA Results | Dict from SALib | JSON file |
| State | In-memory | File-based |
| Usage | Script | MCP tools |
| Workflow | Linear script | Composable tools |

---

## License

Same as parent project.

