import torch
from torch.func import vmap
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json
import gc


train_loss = []  # To store losses
test_loss = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#If the validation loss does not improve after t -> stop training
class EarlyStopping:

    def __init__(self, t = 50, d = 0):
        #t (int): how long to wait
        #d (float): minimum change
        self.t = t
        self.c = 0
        self.best_v_loss = None
        self.e_s = False
        self.v_loss_min = np.inf
        self.d = d

    def __call__(self, v_loss, model):

        if np.isnan(v_loss):
            print("Validation loss: NaN")
            return

        if self.best_v_loss is None:
            self.best_v_loss = v_loss

        elif v_loss < self.best_v_loss - self.d:

            self.best_v_loss = v_loss
            self.c = 0

        else:

            self.c += 1
            
            if self.c >= self.t:
                self.e_s = True

def train(data, output_to_file = True):  
    name = data.get("name", "main")
    model = data.get("model")
    epochs = data.get("epochs")
    batchsize = data.get("batchsize")
    optimizer = data.get("optimizer")
    scheduler = data.get("scheduler")
    component_manager = data.get("component_manager")
    additional_data = data.get("additional_data")

    if output_to_file:
        current_file = os.getcwd()
        mother_dir = os.path.join(current_file, "training")
        counter = 1
        train_dir = os.path.join(mother_dir, name + "_" + str(counter))
        
        if os.path.exists(train_dir):
            while True:
                counter += 1
                train_dir = os.path.join(current_file, "training", name + "_" + str(counter))
                if not os.path.exists(train_dir):
                    break
                    
        model_dir = os.path.join(train_dir, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        params_path = f"{train_dir}/params.json"
        params = {
            "name": name,
            "model": str(model),
            "epochs": epochs,
            "batchsize": batchsize,
            "optimizer": str(optimizer),
            "scheduler": str(scheduler.state_dict()) if scheduler!=None else "None",
            "modules": str(component_manager)
        } 
        if additional_data != None:
            params["additionalData"] = additional_data
        fp = open(params_path, "w", newline='\r\n') 
        json.dump(params, fp)
        fp.close()  


    #for temporal causality weights
    model = model.to(device)

    early_stopping = EarlyStopping(t = 50)

    for epoch in range(epochs):
        model.train(True)
        train_losses = []
        for i in range(component_manager.number_of_iterations(train = True)):
            l = component_manager.apply(model, train = True)

            # No regularization
            reg_loss = model.regularization_loss(regularize_activation=0.0, regularize_entropy=0.0, use_original=False)
            # Efficient L1 regularization
            # reg_loss = model.regularization_loss(regularize_activation=1.0, regularize_entropy=1.0, use_original=False)
            # # Paper L1 regularization
            # reg_loss = model.regularization_loss(regularize_activation=1.0, regularize_entropy=1.0, use_original=True)
            
            total_loss = l + reg_loss
            l.backward()    
            optimizer.step() 
            optimizer.zero_grad()

            train_losses.append(l.item())
        
        epoch_train_loss = np.average(train_losses)
        train_loss.append(epoch_train_loss)

        model.eval()
        validation_losses = []
        for i in range(component_manager.number_of_iterations(train = False)):
            l = component_manager.apply(model, train = False)
            validation_losses.append(l.item())

            del l
            gc.collect()
            
        epoch_val_loss = np.average(validation_losses)
        test_loss.append(epoch_val_loss)
        

        print(f"Epoch nr. {epoch}, avg train loss: {epoch_train_loss}, avg validation loss: {epoch_val_loss}")
        
        if output_to_file and epoch % 25 == 0:
            epoch_path = os.path.join(model_dir, f"model_{epoch}.pt")
            torch.save(model, epoch_path)
        
        if scheduler != None:
            scheduler.step()

        torch.cuda.empty_cache()

        #Early Stopping
        early_stopping(epoch_val_loss, model)
        if early_stopping.e_s:
            print("Early Stopping")
            break

    print(f"Finished training! Avg train loss: {np.average(train_loss)}; Avg val loss: {np.average(test_loss)}")
    model_path = os.path.join(model_dir, f"model_{epoch}.pt")

    # Save the model
    if output_to_file:
        torch.save(model, model_path)
        
        plt.plot(train_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(f'{train_dir}/training_loss.png')
        plt.clf()
        plt.plot(train_loss)
        plt.plot(test_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.savefig(f'{train_dir}/train_and_test_loss.png')
        plt.clf()
        label = ["Residual loss", "IC loss"]
        residual_losses = component_manager.search("Residual", train = False).loss.history
        ic_losses = component_manager.search("IC", train = False).loss
        plt.plot(residual_losses)
        for i in range(len(ic_losses)):
            plt.plot(ic_losses[i].history)
            label.append("IC_loss_"+str(i))
        plt.legend(label)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.savefig(f'{train_dir}/res_ic_train_losses.png')
        plt.show()

    return np.min(test_loss)
