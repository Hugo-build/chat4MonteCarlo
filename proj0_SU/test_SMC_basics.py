from pySMC import (
    SurrogatePipe, StandardScaler,
    GaussianProcess, RBF, optSetup,
    VariableSet, Variable,
    sample_inputs,
    sobol_g
)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def print_metrics(y_true, y_pred, y_std=None):
    """
    Calculate and print regression metrics.
    
    Args:
        y_true: True values (numpy array)
        y_pred: Predicted values (numpy array)
        y_std: Predictive standard deviations (optional)
    
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Regression metrics
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
    
    # Add uncertainty metrics if available
    if y_std is not None:
        y_std = np.array(y_std).flatten()
        # Check calibration: are 95% of points within ±2σ?
        in_bounds = np.abs(y_true - y_pred) <= 2 * y_std
        calibration = np.mean(in_bounds) * 100
        avg_uncertainty = np.mean(y_std)
        
        metrics['Calibration (%)'] = float(calibration)  # Should be ~95%
        metrics['Avg Uncertainty'] = float(avg_uncertainty)
    
    return metrics


def plot_results(y_true, y_pred, y_std, title="Sobol G-Function Surrogate"):
    """
    Create comprehensive visualization of surrogate model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Predictive standard deviations
        title: Plot title
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_std = np.array(y_std).flatten()
    
    # Define professional color palette
    color_true = '#2E86AB'      # Blue for true values
    color_pred = '#A23B72'      # Magenta/purple for predictions
    color_fill = '#F18F01'      # Orange for confidence interval
    color_perfect = '#06A77D'   # Teal for perfect fit line
    color_grid = '#E0E0E0'      # Light gray for grid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Predicted vs True scatter
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50, color=color_pred, 
                edgecolors='white', linewidths=0.5)
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, 
             label='Perfect fit', color=color_perfect)
    ax1.set_xlabel('True Value', fontsize=11)
    ax1.set_ylabel('Predicted Value', fontsize=11)
    ax1.set_title('Predicted vs True', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, color=color_grid, linestyle='--')
    ax1.set_aspect('equal', adjustable='box')
    
    # Add metrics text
    r2 = 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ax1.text(0.05, 0.95, f"R² = {r2:.4f}\nRMSE = {rmse:.4f}", 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Plot 2: Sequential predictions with uncertainty
    ax2 = axes[0, 1]
    idx = np.arange(len(y_true))
    ax2.plot(idx, y_pred, label="Predicted", color=color_pred, 
             linewidth=2, alpha=0.8, marker='o', markersize=4)
    ax2.scatter(idx, y_true, label="True", color=color_true, 
                s=50, alpha=0.7, zorder=3, marker='x', linewidths=2)
    ax2.fill_between(idx, y_pred - 2*y_std, y_pred + 2*y_std, 
                     alpha=0.25, color=color_fill, label="95% confidence", zorder=1)
    ax2.set_xlabel("Sample Index", fontsize=11)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title("Sequential Predictions with Uncertainty", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, color=color_grid, linestyle='--')
    ax2.legend(fontsize=9, framealpha=0.9)
    
    # Plot 3: Residuals
    ax3 = axes[1, 0]
    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.6, s=50, color=color_pred,
                edgecolors='white', linewidths=0.5)
    ax3.axhline(y=0, color='red', linestyle='--', lw=2, label='Zero error')
    ax3.fill_between([y_pred.min(), y_pred.max()], 
                     [-2*y_std.mean(), -2*y_std.mean()],
                     [2*y_std.mean(), 2*y_std.mean()],
                     alpha=0.2, color=color_fill, label='±2σ band')
    ax3.set_xlabel('Predicted Value', fontsize=11)
    ax3.set_ylabel('Residual (True - Predicted)', fontsize=11)
    ax3.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, color=color_grid, linestyle='--')
    
    # Plot 4: Error distribution
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=20, alpha=0.7, color=color_pred, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero error')
    ax4.set_xlabel('Residual', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, color=color_grid, linestyle='--')
    
    # Add mean and std of residuals
    ax4.text(0.95, 0.95, f"Mean: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}", 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig


# ============================================================================
# Main Script
# ============================================================================

print("="*70)
print("Testing pySMC Surrogate Model with Sobol G-Function")
print("="*70)

# Define variables and generate data
print("\n1. Setting up problem...")
vset = VariableSet([
    Variable(name="x1", kind="uniform", params={"low": 0.0, "high": 1.0}),
    Variable(name="x2", kind="uniform", params={"low": 0.0, "high": 1.0})
])
X_train = sample_inputs(vset, 100, kind="lhs", seed=42)

a = np.array([1.0, 1.0])
sample_fn = sobol_g(a)
y_train = np.array([sample_fn(x)["y"] for x in X_train])

print(f"   - Training samples: {len(X_train)}")
print(f"   - Variable dimensions: {X_train.shape[1]}")
print(f"   - y_train range: [{y_train.min():.3f}, {y_train.max():.3f}]")

# Create and fit scalers
print("\n2. Creating and fitting scalers...")
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_scaler.fit(X_train)
y_scaler.fit(y_train.reshape(-1, 1))

# Scale training data
X_train_scaled = x_scaler.transform(X_train)
y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()

# Fit GP on scaled data
print("\n3. Training Gaussian Process surrogate model...")
kernel = RBF.from_params(signal_std=1.0, length_scale=jnp.ones(2) * 0.2)
gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1)
opt_config = optSetup(optimizer='adam', steps=100, lr=0.02, verbose=True)
gp_fitted = gp.fit(jnp.array(X_train_scaled), jnp.array(y_train_scaled), opt_config=opt_config)

# Create surrogate pipe (handles scaling automatically)
print("\n4. Creating SurrogatePipe...")
pipe = SurrogatePipe(
    model=gp_fitted,
    varSet=vset,
    X=X_train,        # Original unscaled data
    y=y_train,        # Original unscaled data
    x_scaler=x_scaler,
    y_scaler=y_scaler
)

# Generate test data and make predictions
print("\n5. Generating test data and making predictions...")
predict_fn = pipe.make_predict_fn()  # functional method
X_test = sample_inputs(vset, 50, kind="sobol", seed=123)
y_pred, y_std = predict_fn(X_test)  # Input/output in original scale

# Calculate true values for test data
y_test = np.array([sample_fn(x)["y"] for x in X_test])

print(f"   - Test samples: {len(X_test)}")
print(f"   - y_test range: [{y_test.min():.3f}, {y_test.max():.3f}]")

# Calculate and print metrics
print("\n" + "="*70)
print("BENCHMARK METRICS")
print("="*70)
metrics = print_metrics(y_test, y_pred, y_std)
pprint(metrics)

# Create visualizations
print("\n6. Creating visualizations...")
fig = plot_results(y_test, y_pred, y_std, 
                   title="Sobol G-Function Surrogate Model Performance")

print("\n" + "="*70)
print("TEST COMPLETE ✓")
print("="*70)