import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def create_animation(u_pinn, u_numerical, X, Y, t, save_path=None):
    """
    Create an animated comparison of PINN and numerical solutions
    """
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Set consistent view angles
    ax1.view_init(elev=25, azim=45)
    ax2.view_init(elev=25, azim=45)
    
    # Initial surface plots
    surf1 = ax1.plot_surface(X[:,:,0], Y[:,:,0], u_pinn[:,:,0], cmap='viridis')
    surf2 = ax2.plot_surface(X[:,:,0], Y[:,:,0], u_numerical[:,:,0], cmap='viridis')
    
    # Set titles and labels
    ax1.set_title('PINN Solution')
    ax2.set_title('Numerical Solution')
    for ax in [ax1, ax2]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Displacement (m)')
        ax.set_zlim(np.min(u_pinn), np.max(u_pinn))
    
    def update(frame):
        # Clear previous frame
        ax1.collections.clear()
        ax2.collections.clear()
        
        # Update surface plots
        surf1 = ax1.plot_surface(X[:,:,0], Y[:,:,0], u_pinn[:,:,frame], 
                                cmap='viridis', alpha=0.8)
        surf2 = ax2.plot_surface(X[:,:,0], Y[:,:,0], u_numerical[:,:,frame], 
                                cmap='viridis', alpha=0.8)
        
        # Update time in title
        ax1.set_title(f'PINN Solution (t = {t[frame]:.2f}s)')
        ax2.set_title(f'Numerical Solution (t = {t[frame]:.2f}s)')
        
        return surf1, surf2

    anim = animation.FuncAnimation(fig, update, frames=len(t), 
                                 interval=50, blit=False)
    

    
    plt.tight_layout()
    return anim

def analyze_regimes(u_pinn, u_numerical, t, plot=True):
    """
    Analyze transition and permanent regimes
    """
    # Calculate energy-like metric (L2 norm of displacement)
    energy_pinn = np.sqrt(np.mean(u_pinn**2, axis=(0,1)))
    energy_numerical = np.sqrt(np.mean(u_numerical**2, axis=(0,1)))
    
    # Calculate rate of change
    energy_rate_pinn = np.abs(np.gradient(energy_pinn, t))
    energy_rate_numerical = np.abs(np.gradient(energy_numerical, t))
    
    # Estimate regime transition point (when rate of change stabilizes)
    threshold = 0.1 * np.max(energy_rate_pinn)  # 10% of max rate
    transition_idx_pinn = np.where(energy_rate_pinn < threshold)[0][0]
    transition_idx_numerical = np.where(energy_rate_numerical < threshold)[0][0]
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot energy evolution
        ax1.plot(t, energy_pinn, label='PINN', color='blue')
        ax1.plot(t, energy_numerical, label='Numerical', color='red', linestyle='--')
        ax1.axvline(x=t[transition_idx_pinn], color='blue', linestyle=':',
                    label=f'PINN Transition ({t[transition_idx_pinn]:.2f}s)')
        ax1.axvline(x=t[transition_idx_numerical], color='red', linestyle=':',
                    label=f'Numerical Transition ({t[transition_idx_numerical]:.2f}s)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('RMS Displacement')
        ax1.set_title('System Energy Evolution')
        ax1.legend()
        ax1.grid(True)
        
        # Plot rate of change
        ax2.semilogy(t, energy_rate_pinn, label='PINN', color='blue')
        ax2.semilogy(t, energy_rate_numerical, label='Numerical', color='red', 
                    linestyle='--')
        ax2.axhline(y=threshold, color='green', linestyle=':', 
                    label='Transition Threshold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Rate of Change (log scale)')
        ax2.set_title('Rate of Energy Change')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'transition_time_pinn': t[transition_idx_pinn],
        'transition_time_numerical': t[transition_idx_numerical],
        'steady_state_value_pinn': np.mean(energy_pinn[transition_idx_pinn:]),
        'steady_state_value_numerical': np.mean(energy_numerical[transition_idx_numerical:])
    }

def plot_error_evolution(u_pinn, u_numerical, t):
    """
    Plot the evolution of error between PINN and numerical solutions
    """
    # Calculate various error metrics over time
    mae = np.mean(np.abs(u_pinn - u_numerical), axis=(0,1))
    rmse = np.sqrt(np.mean((u_pinn - u_numerical)**2, axis=(0,1)))
    max_error = np.max(np.abs(u_pinn - u_numerical), axis=(0,1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, mae, label='MAE', color='blue')
    plt.plot(t, rmse, label='RMSE', color='red')
    plt.plot(t, max_error, label='Max Error', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Error Magnitude')
    plt.title('Evolution of Error Metrics Over Time')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

def comprehensive_analysis(u_pinn, u_numerical, X, Y, t, save_animation=False):
    """
    Perform comprehensive analysis of the membrane simulation
    """
    # Create and display animation
    anim = create_animation(u_pinn, u_numerical, X, Y, t, 
                          save_path='membrane_animation.gif' if save_animation else None)
    plt.show()
    
    # Analyze regimes and display results
    regime_results = analyze_regimes(u_pinn, u_numerical, t)
    print("\nRegime Analysis Results:")
    print(f"PINN Transition Time: {regime_results['transition_time_pinn']:.2f}s")
    print(f"Numerical Transition Time: {regime_results['transition_time_numerical']:.2f}s")
    print(f"PINN Steady State RMS: {regime_results['steady_state_value_pinn']:.3e}")
    print(f"Numerical Steady State RMS: {regime_results['steady_state_value_numerical']:.3e}")
    
    # Plot error evolution
    plot_error_evolution(u_pinn, u_numerical, t)