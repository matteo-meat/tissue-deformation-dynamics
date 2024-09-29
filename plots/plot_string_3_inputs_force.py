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
experiment_name = "string_3inputs_nostiffness_force_ic0hard_icv0"
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

def f(x):
    f=[]
    for t in tt:     
        x_f = 0.8
        h = 1.0
        alpha = 53.59
        za = -h * torch.exp(-400*((x-x_f)**2)) * (4*t*(1 - t))**(alpha - 1)
        f.append(za)
    return np.array(f)

num_inputs = 3 #x, x_f, t

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

def hard_constraint(x, y):
    X = x[:, 0].reshape(-1, 1)
    tau = x[:, -1].reshape(-1, 1)
    U = ((X-1)*X*(delta_x**2)*t_f*tau)*(y+(u_min/delta_u)) - (u_min/delta_u)
    return U
   

def compose_input(x, x_f, t):
    X = (x-x_min)/delta_x
    X = np.hstack((X, np.ones_like(x)*((x_f-x_min)/delta_x)))
    X = np.hstack((X, np.ones_like(x)*(t/t_f)))
    X = torch.Tensor(X).to(torch.device("cuda:0")).requires_grad_()
    return X

model = torch.load(model_path)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

tt = np.linspace(0, t_f, num=101, endpoint=True)
x = np.linspace(x_min, x_max, num=101, endpoint=True).reshape(-1, 1)
x_f = 0.2
preds = []
for t in tt:     
    X = compose_input(x, x_f, t)
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


#3D PLOT
fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"})
im1 = axes[0].plot_surface(xx, tt, pred, cmap='inferno')#, vmin=true.min(), vmax = true.max())
axes[0].set_title(f'Predicted')
axes[0].set_xlabel('X')
axes[0].set_ylabel('T')
plt.colorbar(im1, ax=axes[0])


# Plot Theta True
im2 = axes[1].plot_surface(xx, tt, true, cmap='inferno')
axes[1].set_title(f'True')
axes[1].set_xlabel('X')
axes[1].set_ylabel('T')
plt.colorbar(im2, ax=axes[1])

# Plot Difference
im3 = axes[2].plot_surface(xx, tt, np.abs(pred-true), cmap='inferno')
axes[2].set_title(f'Difference')
axes[2].set_xlabel('X')
axes[2].set_ylabel('T')
plt.colorbar(im3, ax=axes[2])
plt.show()