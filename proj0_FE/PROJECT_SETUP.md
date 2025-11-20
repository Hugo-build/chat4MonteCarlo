# Project 0 Setup Guide

## Overview

This project demonstrates an MCP (Model Context Protocol) server for Finite Element Analysis. The server is designed to work within a project-specific directory, with all file paths relative to the project root for security.

## Files Structure

```
proj0/
├── server_fe.py        # MCP server with FE analysis tools
├── server_GP.py        # Legacy GP tools server
├── nodes.csv           # FE model node definitions
├── elements.csv        # FE model element definitions
├── README.md           # Project documentation
└── PROJECT_SETUP.md    # This file
```

## Key Features

### 1. Project-Relative Paths
The MCP server automatically uses the project directory as its root:
- All file operations are relative to `proj0/`
- Security: Prevents access to files outside the project directory
- The server determines its location using `Path(__file__).parent.resolve()`

### 2. Modified App Integration
The main `app.py` has been updated to:
- Discover available project folders (starting with "proj")
- Allow project selection via UI dropdown
- Launch the MCP server in the selected project's working directory
- Automatically reset the connection when switching projects

### 3. MCP Server Tools

The server provides these tools:

#### FE.detect_files
```python
# Args: subdirectory (default: ".")
# Returns: Dict with found files and subdirectories
```

#### FE.load_model
```python
# Args: subdirectory (default: ".")
# Returns: Dict with nodes, elements, stats, and success status
```

#### FE.plot_model
```python
# Args: subdirectory (default: ".")
# Returns: Dict with base64-encoded PNG image
```

#### FE.get_info
```python
# Args: subdirectory (default: ".")
# Returns: Formatted string with model details
```

#### FE.solve_model
```python
# Args: subdirectory (default: ".")
# Returns: Placeholder (not yet implemented)
```

## Usage

### Running the App

1. Ensure dependencies are installed:
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

2. Set up your `.env` file with OpenAI API key:
```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
MCP_SERVER_SCRIPT=server_fe.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Select "proj0" from the project dropdown in the sidebar

5. Ask the AI assistant about the FE model:
   - "What files are in this project?"
   - "Load the FE model"
   - "Plot the 3D model"
   - "Give me information about the model"

## Technical Details

### App.py Changes

**Added imports:**
```python
from pathlib import Path
```

**Configuration:**
```python
WORKSPACE_ROOT = Path(__file__).parent.resolve()
```

**MCPClient class:**
- Added `working_directory` parameter to `__init__`
- Passes `cwd=working_directory` to subprocess.Popen

**main() function:**
- Discovers projects automatically
- Manages project selection state
- Resets MCP client when project changes
- Launches server in project-specific directory

### Server Changes

**Project root detection:**
```python
PROJECT_ROOT = Path(__file__).parent.resolve()
```

**Path security:**
- All paths resolved relative to PROJECT_ROOT
- Security check: `directory_path.relative_to(PROJECT_ROOT)`
- Prevents directory traversal attacks

**Tool modifications:**
- Changed `directory` parameter to `subdirectory`
- Default to "." (project root)
- All returned paths are relative to project root

## Testing

The project includes a test script `test_proj0_setup.py` (in parent directory) that verifies:
1. All required files exist
2. FE files can be loaded
3. MCP server responds correctly

**Note:** Tests may fail if dependencies are not installed. This is expected in development environments without full package installation.

## Next Steps

To create additional projects:

1. Create a new project folder (e.g., `proj1/`)
2. Copy `server_fe.py` to the new folder
3. Add your FE model files (nodes.csv, elements.csv)
4. The app will automatically detect the new project
5. Select it from the dropdown

## Dependencies

Required packages (from pyproject.toml):
- fastmcp
- streamlit
- openai
- pandas
- numpy
- matplotlib
- python-dotenv

## Security Notes

- The server restricts file access to the project directory
- Attempts to access parent directories are blocked
- All file paths are validated before use
- Subprocess runs with project directory as working directory

