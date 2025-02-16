import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pinns_v2.model import MLP, MLP_RWF, KAN_NET
from pinns_v2.components import ComponentManager, ResidualComponent, ICComponent
from pinns_v2.rff import GaussianEncoding 
#from pinns.train import train
from pinns_v2.train import train
from pinns_v2.gradient import _jacobian
from pinns_v2.dataset import DomainDataset, ICDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 2000
num_inputs = 3 #x, y, t

u_min = -0.21
u_max = 0.0
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0
t_f = 10
f_min = -3.0
f_max = 0.0
delta_u = u_max - u_min
delta_x = x_max - x_min
delta_y = y_max - y_min
delta_f = f_max - f_min

params = {
    "u_min": u_min,
    "u_max": u_max,
    "x_min": x_min,
    "x_max": x_max,
    "y_min": y_min,
    "y_max": y_max,
    "t_f": t_f,
    "f_min": f_min,
    "f_max": f_max
}

def hard_constraint(x_in, U_theta):
    X = x_in[0]
    Y = x_in[1]
    tau = x_in[-1]
    # riportiamo gli input nelle variabili non normalizzate
    x = X*delta_x + x_min
    y = Y*delta_y + y_min
    t = tau * t_f
    u_theta = U_theta*delta_u + u_min

    # se siamo sui bordi (qualsiasi dimensione) u=0, altrimenti abbiamo applicato una trasformazione non lineare a u_theta
    u = u_theta * (x-x_min) *(x-x_max) * (y-y_min) * (y-y_max) * t
    U = (u - u_min)/delta_u # forma esplicita: riga 72 in 73, poi righe 66-69 

    # output normalizzato
    return U

def f(sample):
    # Scale and translate sample coordinates to the actual domain
    x = sample[0] * delta_x + x_min
    y = sample[1] * delta_y + y_min
    
    # Center of Gaussian force application (e.g., around x=0.2, y=0.2 in the domain)
    x_f = 0.2 * delta_x + x_min
    y_f = 0.2 * delta_y + y_min
    
    # Force amplitude
    h = f_min  # or another value depending on desired intensity
    
    # 2D Gaussian function representing the applied force
    z = h * torch.exp(-400 * ((x - x_f) ** 2 + (y - y_f) ** 2))
    return z


# PDE membrane
def pde_fn(model, sample):
    T = 1
    mu = 1
    k = 1
    alpha_2 = (T/mu)*(t_f**2)/(delta_x**2)
    beta_2 = (T/mu)*(t_f**2)/(delta_y**2)
    gamma = (t_f**2)/delta_u
    K = k * t_f
    J, d = _jacobian(model, sample)
    dX = J[0][0]
    dY = J[0][1]
    dtau = J[0][-1]
    ddX = _jacobian(d, sample, i=0, j=0)[0][0]
    ddY = _jacobian(d, sample, i=1, j=1)[0][0]
    ddtau = _jacobian(d, sample, i=2, j=2)[0][0]
    
    return ddtau - alpha_2*ddX -beta_2*ddY - gamma*f(sample) + K*dtau

# Velocity constraints
def ic_fn_vel(model, sample):
    J, d = _jacobian(model, sample)
    dtau = J[0][-1]
    dt = dtau*delta_u/t_f
    ics = torch.zeros_like(dt)
    return dt, ics

def errorMessage():
    print("Invalid input arguments!\n")
    print('If you want to train a KAN, the correct format is:\n')
    print('"python main.py KAN [tanh or relu or silu] [setup1 or setup2] [noreg or orig_reg or eff_reg]\n')
    print('If you want to train MLP or RWF, the correct format is:\n')
    print('python main.py [MLP or RWF] [tanh or relu or silu] [setup1 or setup2]')

def main():

    args = sys.argv[1:]

    if (args[0] == 'KAN' and len(args) != 4) or ((args[0] == 'MLP' or args[0] == 'RWF') and len(args) != 3):
        errorMessage()
        return
    else:
        if args[1] == 'tanh':
            act_fun = nn.Tanh
        elif args[1] == 'relu':
            act_fun = nn.ReLU
        elif args[1] == 'silu':
            act_fun = nn.SiLU
        else:
            errorMessage()
            return
        
        if args[2] == 'setup1':
            
            # SETUP 1: batchsize = 500, lr = 0.002203836177626117
            batchsize = 500
            learning_rate = 0.002203836177626117

            print("Building Domain Dataset")
            domainDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, 10000, batchsize, period = 3)
            print("Building IC Dataset")
            icDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), 10000, batchsize, period = 3)
            print("Building Validation Dataset")
            validationDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, 500, batchsize, shuffle = False)
            print("Building Validation IC Dataset")
            validationicDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), 500, batchsize, shuffle = False)
        
        elif args[2] == 'setup2':

            #SETUP 2: batchsize = None, lr = 0.001

            batchsize = None
            learning_rate = 0.001

            print("Building Domain Dataset")
            domainDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, 10000, period = 3)
            print("Building IC Dataset")
            icDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), 10000, period = 3)
            print("Building Validation Dataset")
            validationDataset = DomainDataset([0.0]*num_inputs,[1.0]*num_inputs, 500, shuffle = False)
            print("Building Validation IC Dataset")
            validationicDataset = ICDataset([0.0]*(num_inputs-1),[1.0]*(num_inputs-1), 500, shuffle = False)
        
        else:
            errorMessage()
            return

        if args[0] == 'KAN':
            model = KAN_NET([num_inputs, 50, 1],grid_size=7, scale_noise=0.05, scale_spline=1.2, scale_base=1.5, activation_function=act_fun, hard_constraint_fn=hard_constraint)
            reg = args[3]

        elif args[0] == 'MLP':
            model = MLP([num_inputs] + [308]*8 + [1], act_fun, hard_constraint, p_dropout=0.3)
            reg = 'noreg'

        elif args[0] == 'RWF':
            model = MLP_RWF([num_inputs] + [308]*8 + [1], act_fun, hard_constraint, p_dropout=0.3)
            reg = 'noreg'

        else:
            errorMessage()
            return

    # encoding = GaussianEncoding(sigma = 1.0, input_size=num_inputs, encoded_size=154)

    component_manager = ComponentManager()
    r = ResidualComponent(pde_fn, domainDataset)
    component_manager.add_train_component(r)
    ic = ICComponent([ic_fn_vel], icDataset)
    component_manager.add_train_component(ic)
    r = ResidualComponent(pde_fn, validationDataset)
    component_manager.add_validation_component(r)
    ic = ICComponent([ic_fn_vel], validationicDataset)
    component_manager.add_validation_component(ic)


    def init_normal(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    model = model.apply(init_normal)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1721, gamma=0.15913059595003437)

    data = {
        "name": args[0], # MLP, RWF, KAN set one of these before training
        "model": model,
        "epochs": epochs,
        "batchsize": batchsize,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "component_manager": component_manager,
        "regularization": reg,
        "additional_data": params
    }

    train(data, output_to_file=True)

main()
 
