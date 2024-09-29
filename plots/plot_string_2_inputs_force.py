from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
from pinns.model import PINN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinns.train import train
from pinns.dataset import DomainDataset, ICDataset
import json

name = "output"
experiment_name = "string_2inputs_nostiffness_force_damping_ic0hard_icv0_t10.0_optimized_MLP_rff1.0"
current_file = os.path.abspath(__file__)
output_dir = os.path.join(os.path.dirname(current_file), name)
output_dir = os.path.join(output_dir, experiment_name)

model_dir = os.path.join(output_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'model.pt')


exact_solution = "C:\\Users\\desan\\Documents\\Wolfram Mathematica\\file.csv"

def exact():
    sol = []
    with open(exact_solution, "r") as f:
        for line in f:
            s = line.split(",")[2].strip()
            s = s.replace('"', '').replace("{", "").replace("}", "").replace("*^", "E")
            s = float(s)
            sol.append(s)
    return np.array(sol)

def load_params(path):
    global u_min, u_max, f_min, f_max, x_min, x_max, t_f, delta_u, delta_x, delta_f
    with open(path, "r") as fp:
        params = json.load(fp)["additionalData"]
        u_min = params["u_min"]
        u_max = params["u_max"]
        x_min = params["x_min"]
        x_max = params["x_max"]
        f_min = params["f_min"]
        f_max = params["f_max"]
        t_f = params["t_f"]
        delta_u = u_max - u_min
        delta_x = x_max - x_min
        delta_f = f_max - f_min
    return

load_params(os.path.join(output_dir, "params.json"))

""" def hard_constraint(x, y):
    res = x[:, 0].reshape(-1, 1) * (1 - x[:, 0]).reshape(-1, 1) * y * x[:, -1].reshape(-1, 1)
    res = (res - u_min)/delta
    return res """

def hard_constraint(x, y):
    """ s = x[:, 0].reshape(-1, 1)*(delta_x) + x_min #x
    t = x[:, -1].reshape(-1, 1)*t_f #t
    y = y*(delta_u)+u_min
    u = (x_min - s) * (x_max - s) * y * t
    U = (u-u_min)/delta_u """
    X = x[:, 0].reshape(-1, 1)
    tau = x[:, -1].reshape(-1, 1)
    U = ((X-1)*X*(delta_x**2)*t_f*tau)*(y+(u_min/delta_u)) - (u_min/delta_u)
    return U

def compose_input(x, t):
    X = (x-x_min)/delta_x
    X = np.hstack((X, np.ones_like(x)*(t/t_f)))
    X = torch.Tensor(X).to(torch.device("cuda:0")).requires_grad_()
    return X

model = torch.load(model_path)
x = torch.randn(1, 2).to(torch.device("cuda:0"))
torch.onnx.export(model, x, os.path.join(output_dir, "nn.onnx"), input_names = ["x, t"], output_names = ["u"])

model.train(False)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tt = np.linspace(0, t_f, num=1001, endpoint=True)
x = np.linspace(x_min, x_max, num=101, endpoint=True).reshape(-1, 1)
x_f = 0.2
f = -1.0

preds = []
for t in tt:     
    X = compose_input(x, t)
    pred = model(X)
    pred = pred.cpu().detach().numpy()
    pred = pred*delta_u + u_min
    preds.append(pred)
preds = np.array(preds)

ttrue = exact()

xx, tt = np.meshgrid(x, tt)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
la = len(np.unique(X[:, 0:1]))
le = len(np.unique(X[:, 1:]))

pred = preds.reshape((le, la))
true = ttrue.reshape((le, la), order="F")

# Plot Theta Predicted
im1 = axes[0].imshow(pred, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])#, vmin=true.min(), vmax = true.max())
axes[0].set_title(f'Predicted')
axes[0].set_xlabel('X')
axes[0].set_ylabel('T')
plt.colorbar(im1, ax=axes[0])

# Plot Theta True
im2 = axes[1].imshow(true, cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
axes[1].set_title(f'True')
axes[1].set_xlabel('X')
axes[1].set_ylabel('T')
plt.colorbar(im2, ax=axes[1])

# Plot Difference
im3 = axes[2].imshow(np.abs(pred-true), cmap='inferno', aspect='auto', origin='lower',
                        extent=[np.unique(X[:, 0:1]).min(), np.unique(X[:, 0:1]).max(), np.unique(X[:, 1:]).min(), np.unique(X[:, 1:]).max()])
axes[2].set_title(f'Difference')
axes[2].set_xlabel('X')
axes[2].set_ylabel('T')
plt.colorbar(im3, ax=axes[2])

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f'{output_dir}/plot_corda_semplice.png')

plt.show()