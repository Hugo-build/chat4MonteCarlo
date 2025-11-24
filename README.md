# Chat4MonteCarlo

This repository provides example projects with MCP (Model Context Protocol) servers for LLM integration. 
The MCP servers enable LLMs to perform Monte-Carlo simulation-based tasks in science and engineering topics.



## 📦 Installation

For dependencies in Python, the packages can be added from pip using:
```bash
pip install -r requirements.txt
```

If using `uv` as the virtual environment manager, please run:
```bash
uv pip install -r requirements.txt
```

## 🚀 Quick Start

Please include a `.env` file at the repository root. The `.env` file should include the OpenAI-compatible API format:
```bash
# Get your API key from OpenAI or other providers
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://theAIprovider.com
```

#### 🔍 Check API Key (Optional)

To test your API key validity manually:

```bash
python test_api_key.py
```

The API key should be in **"OpenAI-compatible"** format.

Simply run the app - it will show a beautiful web-based setup wizard:

```bash
streamlit run app.py
```

After running the UI app from the command line, the UI app will be shown as a web page in the browser.

The UI allows the user to focus on the project of interest. The working project can be re-selected, and the corresponding MCP tools will be loaded for the LLM. The **project selection** is currently set at the top of the sidebar. 

<img src="figs/demo_selectProj.png" alt="Alt Text" width="30%">

After selecting the project, at the sidebar, the available mcp tools within that working dictionary can be found as

<img src="figs/demo_mcpList.png" alt="Alt Text" width="30%">

After navigating into the exact project, the user can start to work with LLM for project work. The main chat interface is on the right-hand side of the sidebar.

<img src="figs/demo_chat.png" alt="Alt Text" width="80%">

In the demo project of `proj0_FE`, the finite element solving can be executed by LLM. The graphical results can be visualized by the AI assistant through functional calls.

<img src="figs/demo_callFEResult.png" alt="Alt Text" width="80%">



---
## 🔐 Security Features

This project includes **.env encryption** with three security levels:

1. **Unencrypted** (⚠️): For testing only
2. **Password-protected** (🔒): AES-256 encryption with password
3. **2FA-protected** (🔐): AES-256 + password + TOTP authenticator

**Quick Commands:**
```bash
# Manual encryption (advanced users)
python encrypt_env.py encrypt           # Password-only encryption
python encrypt_env.py decrypt           # Decrypt with password
python encrypt_env.py setup-2fa         # Setup 2FA (one-time)
python encrypt_env.py encrypt --2fa     # Encrypt with password + 2FA
python encrypt_env.py decrypt --2fa     # Decrypt with password + 2FA
python encrypt_env.py verify-2fa        # Test your 2FA code
python encrypt_env.py disable-2fa       # Remove 2FA (keep password)
```





## 📁 Repository's Structure

This repository contains:

- **`app.py`** - Main Streamlit application with MCP client integration
- **`encrypt_env.py`** - Environment file encryption tool with password and 2FA support
- **`proj0_FE/`** - Finite Element analysis example project with MCP server
- **`proj0_MC/`** - Monte Carlo simulation example project with MCP server
- **`proj0_SU/`** - Surrogate modeling example project with MCP server
- **`pySMC/`** - Core Python package for Sequential Monte Carlo methods (added from another project --> [pySMC](https://github.com/Hugo-build/pySMC) 
- **`requirements.txt`** - Python package dependencies

Each example project (`proj0_*`) includes its own MCP server (`server.py`) that exposes specialized tools for LLM integration.