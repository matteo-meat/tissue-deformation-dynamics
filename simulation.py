import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq

def main():
    print("Starting simulation...")
    
    # Physical parameters
    L = 1.0           
    T = 1.0           
    mu = 1.0          
    k = 1.0           
    
    # Force parameters
    x_f = 0.2         
    y_f = 0.2         
    h = -3.0          
    force_spread = 400 
    
    # Numerical parameters
    Nx = 100          
    Ny = 100          
    dx = L/(Nx-1)     
    dy = L/(Ny-1)     
    dt = 0.0005       
    t_end = 10.0      
    num_frames = 200   

    c = np.sqrt(T/mu)
    
    # Create spatial grid
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    X, Y = np.meshgrid(x, y)

    # Initialize displacement arrays
    u = np.zeros((3, Nx, Ny))
    
    # Storage for time history at force point
    total_steps = int(t_end/dt)
    time_history = np.zeros(total_steps)
    time_array = np.arange(total_steps) * dt
    
    # Find indices closest to force application point
    x_f_idx = np.argmin(np.abs(x - x_f))
    y_f_idx = np.argmin(np.abs(y - y_f))

    # External force function
    def external_force(X, Y):
        return h * np.exp(-force_spread * ((X - x_f)**2 + (Y - y_f)**2))
    
    # Calculate force field
    F = external_force(X, Y)
    
    # Set up the figure with subplots
    plt.ion()
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplots
    gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0, :], projection='3d')  # Main 3D view
    ax2 = fig.add_subplot(gs[1, 0])  # Time history
    ax3 = fig.add_subplot(gs[1, 1])  # Frequency spectrum
    
    # Main simulation and animation loop
    step = 0
    try:
        for frame in range(num_frames):
            if frame % 10 == 0:
                print(f"Processing frame {frame}/{num_frames}")
            
            # Clear 3D plot
            ax1.clear()
            
            # Update membrane multiple times between frames
            for _ in range(int(t_end/(dt*num_frames))):
                # Update interior points
                u[2, 1:-1, 1:-1] = (
                    2 * u[1, 1:-1, 1:-1] * (1 - k*dt/2) 
                    - u[0, 1:-1, 1:-1] * (1 - k*dt/2)
                    + (c * dt)**2 * (
                        (u[1, 2:, 1:-1] - 2*u[1, 1:-1, 1:-1] + u[1, :-2, 1:-1])/dx**2 +
                        (u[1, 1:-1, 2:] - 2*u[1, 1:-1, 1:-1] + u[1, 1:-1, :-2])/dy**2
                    )
                    + dt**2 * F[1:-1, 1:-1]
                ) / (1 + k*dt/2)
                
                # Apply boundary conditions
                u[2, 0, :] = u[2, -1, :] = u[2, :, 0] = u[2, :, -1] = 0
                
                # Store time history at force point
                if step < total_steps:
                    time_history[step] = u[1, y_f_idx, x_f_idx]
                    
                # Shift time levels
                u[0] = u[1].copy()
                u[1] = u[2].copy()
                step += 1
            
            current_time = frame*t_end/num_frames
            
            # 3D surface plot
            surf = ax1.plot_surface(X, Y, u[1], cmap='seismic',
                                  linewidth=0, antialiased=True)
            ax1.set_xlabel('x (m)')
            ax1.set_ylabel('y (m)')
            ax1.set_zlabel('Displacement (m)')
            ax1.set_zlim(-0.21, 0.0)
            ax1.view_init(elev=20, azim=45)
            ax1.set_title(f'Membrane Displacement (t={current_time:.2f}s)')
            
            # Time history plot
            ax2.clear()
            ax2.plot(time_array[:step], time_history[:step], 'b-', linewidth=1)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Displacement at force point (m)')
            ax2.set_title('Time History')
            ax2.grid(True)
            
            # Frequency spectrum (updated every frame)
            if step > 1:
                ax3.clear()
                # Calculate FFT of the time history so far
                yf = fft(time_history[:step])
                xf = fftfreq(step, dt)
                # Plot only positive frequencies
                pos_freq_mask = xf > 0
                ax3.semilogy(xf[pos_freq_mask], 2.0/step * np.abs(yf[pos_freq_mask]))
                ax3.set_xlabel('Frequency (Hz)')
                ax3.set_ylabel('Amplitude')
                ax3.set_title('Frequency Spectrum')
                ax3.grid(True)
                ax3.set_ylim(1e-6, 1e0)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
            
        print("Animation complete!")
        plt.ioff()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred during animation: {e}")
        raise

if __name__ == "__main__":
    print("Script started")
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
    print("Script finished")
    
    plt.show(block=True)