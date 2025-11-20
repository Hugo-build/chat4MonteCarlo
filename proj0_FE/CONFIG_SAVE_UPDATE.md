# FE Config Auto-Save Feature

## Summary

Modified the `FE.load_model` tool to automatically save the loaded FE model to a JSON configuration file. This makes it easy to export model data for use with FE solvers.

## Changes Made

### Modified: `server_fe.py`

Added automatic config saving in the `load_fe_model()` function:

```python
# Save FE config to JSON file (matching exp_40barTruss.py format)
fe_config = {
    "nodes": nodes,
    "elements": elements
}

config_path = directory_path / "fe_config.json"
try:
    with open(config_path, 'w') as f:
        json.dump(fe_config, f, indent=2)
    config_saved = True
    config_file = str(config_path.relative_to(PROJECT_ROOT))
except Exception as e:
    config_saved = False
    config_file = None
```

### Return Value Updates

The `load_fe_model` function now returns two additional fields:
- `config_saved`: boolean indicating if the config file was successfully saved
- `config_file`: relative path to the saved config file (e.g., "fe_config.json")

## Usage

Simply load the model:

```python
# Via MCP tool call
result = FE.load_model()

# The config is automatically saved to proj0/fe_config.json
```

## Config File Format

The generated `fe_config.json` matches the format used in `exp_40barTruss.py`:

```json
{
  "nodes": [
    {
      "id": 1,
      "coords": [0.0, 0.0, 0.0]
    },
    {
      "id": 2,
      "coords": [100.0, 0.0, 0.0]
    }
  ],
  "elements": [
    {
      "id": 1,
      "node_i": 1,
      "node_j": 2,
      "E": 200000000000.0,
      "A": 100.0,
      "rho": 7.85e-06
    }
  ]
}
```

## Benefits

1. **Easy Export**: Model data is immediately available in JSON format
2. **Format Compatibility**: Matches `exp_40barTruss.py` format for direct use
3. **No Extra Step**: Automatically happens when loading the model
4. **Version Control**: Config files can be committed to track model changes

## Future Extensions

The config format can be extended to include:
- Boundary conditions (`fixed_dofs`)
- Load vectors (`loads`)
- Material sets
- Analysis settings

These can be added later without breaking compatibility.


