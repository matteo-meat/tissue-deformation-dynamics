import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from animation import comprehensive_analysis

# Global parameters
u_min, u_max = -0.21, 0.0
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
t_f = 10
f_min, f_max = -3.0, 0.0

def load_model(model_path):
    print(f"Loading model from {model_path}")
    try:
        model = torch.load(model_path)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def hard_constraint(x_in, U_theta):
    delta_u = u_max - u_min
    delta_x = x_max - x_min
    delta_y = y_max - y_min

    X = x_in[0]
    Y = x_in[1]
    tau = x_in[-1]
    x = X*delta_x + x_min
    y = Y*delta_y + y_min
    t = tau * t_f
    u_theta = U_theta*delta_u + u_min

    # se siamo sui bordi (qualsiasi dimensione) u=0, altrimenti abbiamo applicato una trasformazione non lineare a u_theta
    u = u_theta * (x-x_min) *(x-x_max) * (y-y_min) * (y-y_max) * t
    U = (u - u_min)/delta_u # forma esplicita: riga 72 in 73, poi righe 66-69 

    # output normalizzato
    return U

def get_pinn_predictions(model, X_norm, Y_norm, T_norm):
    inputs = torch.tensor(np.column_stack([X_norm, Y_norm, T_norm]), dtype=torch.float32)
    print(f"Inputs shape: {inputs.shape}")
    
    # Let's first check what the model outputs directly
    with torch.no_grad():
        raw_predictions = model(inputs)
        if raw_predictions.shape[1] > 1:
            raw_predictions = raw_predictions[:, 0]
        print(f"Raw model predictions - min: {raw_predictions.min()}, max: {raw_predictions.max()}, mean: {raw_predictions.mean()}")
    
    predictions = []
    # Let's test with just a few points first
    test_indices = [0, len(inputs)//2, -1]  # Beginning, middle, and end points
    
    for idx in test_indices:
        x_in = [X_norm[idx], Y_norm[idx], T_norm[idx]]
        pred = raw_predictions[idx].item()
        print(f"\nTesting point {idx}:")
        print(f"Input coordinates (normalized): {x_in}")
        print(f"Raw model prediction: {pred}")
        constrained_pred = hard_constraint(x_in, pred)
        predictions.append(constrained_pred)
        
    print("\nTest predictions:", predictions)
    
    # Now process all points
    all_predictions = []
    for i in range(len(inputs)):
        x_in = [X_norm[i], Y_norm[i], T_norm[i]]
        pred = raw_predictions[i].item()
        constrained_pred = hard_constraint(x_in, pred)
        all_predictions.append(constrained_pred)
    
    return np.array(all_predictions)
# def get_pinn_predictions(model, X_norm, Y_norm, T_norm):
#     # Stack the coordinates into a single input tensor
#     inputs = torch.tensor(np.column_stack([X_norm, Y_norm, T_norm]), dtype=torch.float32)
#     print(f"Inputs shape: {inputs.shape}")
    
#     predictions = []
#     batch_size = 50000
    
#     with torch.no_grad():
#         for i in range(0, len(inputs), batch_size):
#             batch = inputs[i:i+batch_size]
#             pred = model(batch)
#             if pred.shape[1] > 1:
#                 pred = pred[:, 0]
#             print(f"Batch {i // batch_size}: predictions shape = {pred.shape}")
#             predictions.append(pred.numpy())
    
#     predictions = np.concatenate(predictions)
#     print(f"Final predictions size: {predictions.size}")
#     return predictions


# def compute_numerical_solution(Nx, Ny, Nt, x, y, t):
#     print("Computing numerical solution...")
#     u_numerical = np.zeros((Nx, Ny, Nt))
    
#     T = 1.0
#     mu = 1.0
#     k = 1.0
#     c = np.sqrt(T / mu)
    
#     dx = x[1] - x[0]
#     dy = y[1] - y[0]
#     dt = t[1] - t[0]
    
#     x_f, y_f = 0.2, 0.2
#     h = f_min
#     force_spread = 400
#     X, Y = np.meshgrid(x, y)
#     F = h * np.exp(-force_spread * ((X - x_f)**2 + (Y - y_f)**2))
    
#     u = np.zeros((3, Nx, Ny))
    
#     for n in range(Nt):
#         u_numerical[:, :, n] = u[1]
        
#         u[2, 1:-1, 1:-1] = (
#             2 * u[1, 1:-1, 1:-1] * (1 - k*dt/2) 
#             - u[0, 1:-1, 1:-1] * (1 - k*dt/2)
#             + (c * dt)**2 * (
#                 (u[1, 2:, 1:-1] - 2*u[1, 1:-1, 1:-1] + u[1, :-2, 1:-1]) / dx**2 +
#                 (u[1, 1:-1, 2:] - 2*u[1, 1:-1, 1:-1] + u[1, 1:-1, :-2]) / dy**2
#             )
#             + dt**2 * F[1:-1, 1:-1]
#         ) / (1 + k*dt/2)
        
#         u[2, 0, :] = u[2, -1, :] = u[2, :, 0] = u[2, :, -1] = 0
        
#         u[0] = u[1].copy()
#         u[1] = u[2].copy()
        
#     return u_numerical

# def visualize_comparison(u_pinn, u_numerical, X, Y, t, times_to_plot):
#     print("Creating visualization...")
#     fig = plt.figure(figsize=(15, 10))
    
#     for idx, t_idx in enumerate(times_to_plot):
#         current_time = t[t_idx]
        
#         ax1 = fig.add_subplot(2, len(times_to_plot), idx+1, projection='3d')
#         ax1.plot_surface(X[:, :, 0], Y[:, :, 0], u_pinn[:, :, t_idx], cmap='seismic')
#         ax1.set_title(f'PINN t={current_time:.2f}s')
#         ax1.set_zlim(u_min, u_max)
#         ax1.set_xlabel('x (m)')
#         ax1.set_ylabel('y (m)')
#         ax1.set_zlabel('Displacement (m)')
        
#         ax2 = fig.add_subplot(2, len(times_to_plot), len(times_to_plot) + idx + 1, projection='3d')
#         ax2.plot_surface(X[:, :, 0], Y[:, :, 0], u_numerical[:, :, t_idx], cmap='seismic')
#         ax2.set_title(f'Numerical t={current_time:.2f}s')
#         ax2.set_zlim(u_min, u_max)
#         ax2.set_xlabel('x (m)')
#         ax2.set_ylabel('y (m)')
#         ax2.set_zlabel('Displacement (m)')
    
#     plt.tight_layout()
#     plt.show()

def save_predictions(predictions, X, Y, T, filename='predictions.csv'):
    """
    Save predictions along with their coordinates to a CSV file.
    
    Args:
        predictions: numpy array of predicted values
        X: x-coordinates
        Y: y-coordinates
        T: time values
        filename: output CSV filename
    """
    # Create a DataFrame with all the data
    results_df = pd.DataFrame({
        'x': X,
        'y': Y,
        't': T,
        'prediction': predictions
    })
    
    # Save to CSV
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    # Also save as NPZ file for backup
    np.savez('predictions.npz', 
             predictions=predictions,
             x=X,
             y=Y,
             t=T)
    print(f"Results also saved to predictions.npz")

def main():
    model_path = '/Users/francesco/Desktop/PAI1/training/KAN_2/model/model_324.pt'
    csv_path = 'matlab/only_damp/space_time_points.csv'
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert pandas Series to numpy arrays
    X = df['x'].to_numpy()
    Y = df['y'].to_numpy()
    T = df['t'].to_numpy()

    # Normalize the coordinates
    X_norm = (X - x_min) / (x_max - x_min)
    Y_norm = (Y - y_min) / (y_max - y_min)
    T_norm = T / t_f
    
    # Load the model
    model = load_model(model_path)
    
    # Get predictions with hard constraint applied
    predictions = get_pinn_predictions(model, X_norm, Y_norm, T_norm)
    
    # Note: Don't denormalize here since hard_constraint already handles that
    # The predictions are already in the correct range
    
    # Save the results
    save_predictions(predictions, X, Y, T, 'matlab/only_damp/pinn_predictions.csv')
    
    # Calculate and print some statistics
    print("\nPrediction Statistics:")
    print(f"Min value: {np.min(predictions):.6f}")
    print(f"Max value: {np.max(predictions):.6f}")
    print(f"Mean value: {np.mean(predictions):.6f}")
    print(f"Number of points: {len(predictions)}")

if __name__ == "__main__":
    main()
