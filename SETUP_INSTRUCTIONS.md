# Quick Setup Instructions

## 🚀 Setup on Each Machine

### Step 1: Create Conda Environment

```bash
# Create environment
conda create -n chat4MonteCarlo python=3.10

# Activate
conda activate chat4MonteCarlo
```

### Step 2: Install Dependencies

**Choose ONE method:**

#### Option A: Using requirements.txt (Simple)
```bash
cd /path/to/chat4MonteCarlo
pip install -r requirements.txt
```

#### Option B: Using pyproject.toml (Modern)
```bash
cd /path/to/chat4MonteCarlo
pip install -e .
```

**Both work!** Use whichever you prefer.

### Step 3: Test Installation

```bash
python -c "import numpy, pandas, streamlit; print('✅ All working!')"
```

---

## 🔄 Daily Workflow

### Activate Environment
```bash
conda activate chat4MonteCarlo
```

### Run Your App
```bash
cd /path/to/chat4MonteCarlo
streamlit run app.py
```

### Deactivate When Done
```bash
conda deactivate
```

---

## ➕ Adding New Packages

### If using requirements.txt:
```bash
# 1. Add to requirements.txt:
echo "seaborn>=0.12" >> requirements.txt

# 2. Install
pip install -r requirements.txt
```

### If using pyproject.toml:
```bash
# 1. Edit pyproject.toml - add to dependencies list
# 2. Reinstall
pip install -e .
```

OneDrive syncs automatically! Other machine just needs to run install command.

---

## 📋 Quick Commands

```bash
# Activate environment
conda activate chat4MonteCarlo

# Install dependencies
pip install -r requirements.txt        # Using requirements.txt
# OR
pip install -e .                       # Using pyproject.toml

# Install with dev tools
pip install -r requirements-dev.txt

# List installed packages
pip list

# Check specific package
pip show numpy

# Run app
streamlit run app.py

# Analyze project dependencies
python list_project_dependencies.py
```

---

## 📚 Documentation

- **`docs/REQUIREMENTS_TXT_GUIDE.md`** - Full guide for requirements.txt
- **`docs/PYPROJECT_VS_REQUIREMENTS.md`** - Comparison and clarification
- **`docs/CONDA_SETUP.md`** - Conda environment setup
- **`docs/CONDA_QUICK_REFERENCE.md`** - Command cheatsheet

---

## ⚠️ Important Notes

1. **Virtual environments stay local** - Not synced by OneDrive
2. **Code syncs via OneDrive** - Source files, requirements.txt, pyproject.toml
3. **Each machine has its own environment** - Recreated independently
4. **No conflicts!** - This is the correct setup

---

## 🆘 Troubleshooting

### Environment not found?
```bash
conda env list  # Check if it exists
conda create -n chat4MonteCarlo python=3.10  # Recreate if needed
```

### Package not found?
```bash
conda activate chat4MonteCarlo
pip install -r requirements.txt  # Reinstall all
```

### Something broken?
```bash
# Nuclear option: delete and recreate
conda remove -n chat4MonteCarlo --all
conda create -n chat4MonteCarlo python=3.10
conda activate chat4MonteCarlo
pip install -r requirements.txt
```

---

## ✅ You're All Set!

Your setup:
- ✅ Conda environment (local to each machine)
- ✅ Dependencies defined (syncs via OneDrive)
- ✅ Simple workflow
- ✅ Works across multiple machines

**Next step:** Activate environment and start coding! 🎉

```bash
conda activate chat4MonteCarlo
streamlit run app.py
```

