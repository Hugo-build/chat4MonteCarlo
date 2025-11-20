# Project 0: FE Analysis Demo

This project demonstrates finite element analysis using the MCP (Model Context Protocol) framework.

## Contents

- `server_fe.py` - MCP server providing FE analysis tools
- `server_GP.py` - MCP server for Gaussian Process tools (legacy)
- `nodes.csv` - Node definitions for the 40-bar truss model
- `elements.csv` - Element definitions for the 40-bar truss model

## Available MCP Tools

The FE server provides the following tools:

### File Detection
1. **detect_folders** - List all folders in the project directory
2. **detect_json** - Detect JSON files in the project directory
3. **FE.detect_files** - Detect FE model files (nodes.csv, elements.csv)

### Model Management
4. **FE.load_model** - Load and validate FE model from CSV files (auto-saves to `fe_config.json`)
5. **FE.get_info** - Get detailed information about the FE model

### Boundary Conditions
6. **FE.add_fixed_dofs** - Add fixed DOFs (boundary conditions) to the config
7. **FE.add_load** - Add a load at a specific node and direction

### Visualization and Analysis
8. **FE.plot_model** - Generate 3D visualization of the undeformed FE model
9. **FE.plot_deformed** - Plot deformed structure overlaid with original (after solving)
10. **FE.solve_static** - Solve static FE model and save displacements to config

## Usage

1. Select this project (proj0) in the MCP Chat Demo app
2. The server will automatically work within this project directory
3. All file paths are relative to the project root for security

## FE Model: 40-Bar Truss

The included model is a 40-bar truss structure commonly used for structural optimization studies.

- Nodes: 20
- Elements: 40
- Type: 3D space truss
- Material: Steel (E = 200 GPa)

## Example Queries

Try asking the AI assistant:

### Basic Model Operations
- "What FE files are available in this project?"
- "Load the FE model and show me the statistics"
- "Give me detailed information about the model"

### Adding Boundary Conditions
- "Fix the first 4 nodes (DOFs 0-11)"
- "Add fixed DOFs to the config"
- "Apply a 1000N load on node 10 in the y-direction"
- "Add a 500N load in the z-direction on node 15"
- "Add multiple loads: 1000N on node 10 (y) and 500N on node 15 (z)"

### Visualization
- "Plot the FE model in 3D"
- "Show me the 3D visualization of the model"
- "Plot the deformed structure"
- "Show me the original and deformed structures together"
- "Plot with a scale factor of 500"

### Solving
- "Solve the static FE model"
- "Run the FE solver and show me the displacements"
- "What is the maximum displacement?"

## FE Config Format

### Initial Config (after `FE.load_model`)

When you call `FE.load_model`, it automatically saves a JSON config file (`fe_config.json`):

```json
{
  "nodes": [
    {"id": 1, "coords": [0.0, 0.0, 0.0]},
    {"id": 2, "coords": [100.0, 0.0, 0.0]},
    ...
  ],
  "elements": [
    {"id": 1, "node_i": 1, "node_j": 2, "E": 2e11, "A": 100, "rho": 7.85e-6},
    {"id": 2, "node_i": 2, "node_j": 3, "E": 2e11, "A": 100, "rho": 7.85e-6},
    ...
  ]
}
```

### Complete Config (after `FE.add_fixed_dofs` and `FE.add_load`)

After adding boundary conditions with `FE.add_fixed_dofs` and loads with `FE.add_load`, the config includes:

```json
{
  "nodes": [...],
  "elements": [...],
  "fixed_dofs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  "loads": [0, 0, 0, ..., 1000.0, ...]
}
```

This format exactly matches the one used in `exp_40barTruss.py` for easy integration with FE solvers.

### DOF Numbering

- Total DOFs = 3 × N (where N = number of nodes)
- Node i (1-based) has DOFs:
  - X: `3 × (i-1) + 0`
  - Y: `3 × (i-1) + 1`
  - Z: `3 × (i-1) + 2`

**Example:** For Node 10:
- X-DOF: 27 (index in loads array)
- Y-DOF: 28
- Z-DOF: 29

So `loads[28] = 1000.0` applies a 1000N load in the Y-direction at node 10.

### Solved Config (after `FE.solve_static`)

After solving with `FE.solve_static`, the config is updated with solution results:

```json
{
  "nodes": [...],
  "elements": [...],
  "fixed_dofs": [...],
  "loads": [...],
  "displacements": [0, 0, 0, ..., 0.0002, ...],
  "max_displacement": 0.000234,
  "solved": true
}
```

The `displacements` array follows the same DOF numbering as the `loads` array.

