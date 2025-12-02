import os
import json

import numpy as np # Developed with 1.26.4
import torch # Developed with 2.0.1
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from torchvision.transforms import functional, Resize # Developed with 0.15.2
from tqdm import tqdm # Developed with 4.66.2

ML_OUT_SZ = 6
DEF_IN_SIZE = (40,30,3)

ACTIVS = {
    'relu': F.relu,
    'tanh': F.tanh,
    'leaky_relu': F.leaky_relu
}

# filters 31, kernel_sz 5,  dense_sz 111, activs relu, padding 0
class ConvModel(nn.Module):
    def __init__(self, in_size=DEF_IN_SIZE, filts=32, kerns=3, pad="same",
                activ=F.relu, dense_sz=100, out_size=ML_OUT_SZ, out_activ=F.sigmoid,
                grey_scaled=False, square_in=False):
        super().__init__()
        H, W, C = in_size
        self.activ = activ
        self.out_activ = out_activ
        self.ML = out_size == ML_OUT_SZ
        
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=filts, kernel_size=kerns, padding=pad)
        out1 = (H - kerns + 2 * pad + 1, W - kerns + 2 * pad + 1, filts)

        self.conv2 = nn.Conv2d(in_channels=filts, out_channels=filts, kernel_size=kerns, padding=pad)
        out2 = (out1[0] - kerns + 2 * pad + 1, out1[1] - kerns + 2 * pad + 1, filts)

        self.pool = nn.MaxPool2d(2, 2)
        out_p = (out2[0] // 2, out2[1] // 2, filts)

        self.conv3 = nn.Conv2d(in_channels=filts, out_channels=2*filts, kernel_size=kerns, padding=pad)
        out3 = (out_p[0] - kerns + 2 * pad + 1, out_p[1] - kerns + 2 * pad + 1, 2 * filts)

        self.conv4 = nn.Conv2d(in_channels=2*filts, out_channels=2*filts, kernel_size=kerns, padding=pad)
        out4 = (out3[0] - kerns + 2 * pad + 1, out3[1] - kerns + 2 * pad + 1, 2 * filts)

        out_p2 = (out4[0] // 2, out4[1] // 2, 2 * filts)
        self.fc1 = nn.Linear(np.prod(out_p2), dense_sz)
        self.fc2 = nn.Linear(dense_sz, out_size)
        
        self.grey_scaled = grey_scaled
        self.square_in = square_in
        self.height, self.width = H, W
        
        assert not square_in or H == W, 'Error: provided input size is not square while square_in flag is set.'
        assert not grey_scaled or C == 1, 'Error: provided input size has more than 1 channel while grey_scaled flag is set.'
        
        
    def forward(self, x):
        # From: [batch_size, height, width, channels]
        # To: [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        if self.square_in:
            x = self._square(x)
        if self.grey_scaled:
            x = self._grey_scale(x)
            
        x = self.activ(self.conv1(x))
        x = self.pool(self.activ(self.conv2(x)))
        x = self.activ(self.conv3(x))
        x = self.pool(self.activ(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.activ(self.fc1(x))
        if self.ML:
            x = self.out_activ(self.fc2(x))
        else:
            x = self.fc2(x)
        return x
    
    def forward_with_activs(self, x):
        # From: [batch_size, height, width, channels]
        # To: [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        if self.square_in:
            x = self._square(x)
        if self.grey_scaled:
            x = self._grey_scale(x)
            
        compiled_layers = {}
            
        x = self.activ(self.conv1(x))
        compiled_layers['conv1'] = x
        x = self.activ(self.conv2(x))
        compiled_layers['conv2'] = x
        x = self.activ(self.conv3(self.pool(x)))
        compiled_layers['conv3'] = x
        x = self.activ(self.conv4(x))
        compiled_layers['conv4'] = x
        x = torch.flatten(self.pool(x), 1)
        x = self.activ(self.fc1(x))
        compiled_layers['fc1'] = x
        if self.ML:
            x = self.out_activ(self.fc2(x))
        else:
            x = self.fc2(x)
        return x, compiled_layers


    def _grey_scale(self, x):
        # Default Assumption: colour-bands in RGB Convention
        # input shape: [batch_size, channels, height, width]
        x = functional.rgb_to_grayscale(x)
        return x
    
    def _square(self, x):
        # input shape: [batch_size, channels, height, width]
        x = Resize((self.height, self.width))(x)
        return x


class ModelWrapper:       
    
    def __init__(self, in_size=(40,30,3), filts=32, kerns=3, pad=1, out_size=ML_OUT_SZ,  # same = 1, valid = 0
                activ=F.relu, dense_sz=100, beta_1=0.9, beta_2=0.999, lr=0.001, l2=0.002,
                grey_scaled=False, square_in=False, dev='cuda'):
        self.param_dict = {
            "in_size": in_size,
            "filts": filts,
            "kerns": kerns,
            "pad": pad,
            "out_size": out_size,
            "activ": activ.__name__,
            "dense_sz": dense_sz,
            "beta_2": beta_2,
            "beta_1": beta_1,
            "l2": l2,
            "lr": lr,
            "grey_scaled": grey_scaled,
            "square_in": square_in,
        }
        self.dev = dev
        torch.set_default_device(dev)
        self.ML = out_size == ML_OUT_SZ
        
        self.out_activ = F.sigmoid if self.ML else F.softmax
        criterion = nn.BCELoss() if self.ML else nn.CrossEntropyLoss()
        
        self.model = ConvModel(in_size=in_size, filts=filts, kerns=kerns, pad=pad, activ=activ, 
                               dense_sz=dense_sz, out_size=out_size, out_activ=self.out_activ, grey_scaled=grey_scaled, square_in=square_in)
        self.criterion = criterion
        self.optimiser = optim.Adam(self.model.parameters(), lr=lr, betas=(beta_1, beta_2), weight_decay=l2)

        self.trained_epochs = 0
        self.conv_epoch = 0
        self.min_weights = self.model.state_dict()
        self.history = {}
        self.l2 = l2


    def train(self, train_loader, val_loader, epochs=10, patience=3, verbose=2):
        self.model.to(self.dev)
        self.criterion.to(self.dev)

        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        sample_difficulty = []
        min_v_loss = np.inf
        stagnant_count = 0
        
        prev_l, prev_a, prev_vl, prev_va = 0., 0., 0., 0.
        for epoch in (pbar := tqdm(range(epochs), total=epochs, disable=verbose!=0)):
            pbar.set_description(f"Training model - l {round(prev_l,4)}, vl {round(prev_vl,4)}; a {round(prev_a,4)}, va {round(prev_va,4)}")
            
            self.model.train()
            for inputs, labels in tqdm(train_loader, disable=(verbose < 2),
                                desc=f"Epoch {epoch}"):
                # inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                
                self.optimiser.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimiser.step()
            
            del inputs, labels
            torch.cuda.empty_cache()
                
            val_loss, val_acc, _ = self.reevaluate(val_loader, verbose=verbose, data_name="Validation set")
            train_loss, train_acc, train_sd = self.reevaluate(train_loader, verbose=verbose, data_name="Train set")
            
            if len(sample_difficulty) == 0:
                sample_difficulty = train_sd
            else:
                sample_difficulty = [sd + tsd for sd, tsd in zip(sample_difficulty, train_sd)]
            
            if epoch > 1:
                if prev_vl - val_loss > 0:
                    stagnant_count = 0
                    if val_loss < min_v_loss:
                        min_v_loss = val_loss
                        self.min_weights = self.model.state_dict()
                        self.conv_epoch = epoch
                else:
                    stagnant_count += 1
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            prev_l, prev_a, prev_vl, prev_va = train_loss, train_acc, val_loss, val_acc
            
            if verbose > 0:
                print(f'E{epoch}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f} -- train acc: {train_acc:.4f}, val acc: {val_acc:.4f}')
                print("Saving history state...")
                
            self.history['train_loss'] = train_losses
            self.history['val_loss'] = val_losses
            self.history['train_acc'] = train_accs
            self.history['val_acc'] = val_accs
            self.trained_epochs = epoch+1
            
            if stagnant_count > patience:
                break
        
        self.load_weights(self.min_weights)
        sample_difficulty = [sd / self.trained_epochs for sd in sample_difficulty]
        self.history['sample_difficulty'] = sample_difficulty
        
        
        return self.history

    
    def reevaluate(self, data_loader, verbose=0, data_name="dataset"):
        loss = 0.0
        acc = 0.0
        self.model.eval()
        sd = []
        with torch.no_grad():
            
            for inputs, labels in tqdm(data_loader, disable=(verbose < 2),
                                desc=f"Re-evaluating model on {data_name}"):
                inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                
                outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()
                                
                outputs = (outputs > 0.5).int() if self.ML else torch.argmax(outputs, dim=1)
                labels = labels if self.ML else torch.argmax(labels, dim=1)
                correct = outputs == labels
                if len(correct.shape) == 2:
                    correct = correct.all(dim=1)
                
                sd += (~correct).int().tolist()
                acc += correct.sum().item()
        
        loss = loss / len(data_loader)
        acc = acc / len(data_loader.dataset)
        return loss, acc, sd
    
    def predict_proba(self, data_loader, verbose=0):
        outputs_list = []
        self.model.eval()
        with torch.no_grad():
            
            for j, (inputs, labels) in tqdm(enumerate(data_loader), disable=(verbose < 2),
                                desc=f"Obtaining model predictions", total=len(data_loader)):
                inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                
                outputs = self.model(inputs)
                # if not self.ML:
                #     outputs = self.out_activ(outputs)
                out_np = outputs.to('cpu').numpy()
                
                outputs_list.append(out_np)
        self.model.train()
        return np.vstack(outputs_list)

    def predict(self, data_loader, return_proba=False, verbose=0):
        pred_probas = self.predict_proba(data_loader, verbose)
        preds = np.int16(pred_probas > 0.5) if self.ML else np.argmax(pred_probas, axis=1)
        
        if return_proba:
            return preds, pred_probas
        return preds
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        params = self.param_dict
        hist = {
            "trained_epochs": self.trained_epochs,
            "conv_epoch": self.conv_epoch,
            "history": self.history
        }
        weights = self.min_weights
        
        torch.save(weights, os.path.join(path, "model_weights.pt"))
        with open(os.path.join(path, "parameters.json"), 'w') as params_file:
            json.dump(params, params_file)
        with open(os.path.join(path, "history.json"), 'w') as hist_file:
            json.dump(hist, hist_file)


    def load(self, path):        
        with open(os.path.join(path, "parameters.json"), 'r') as pd_file:
            param_dict = json.load(pd_file)
        self.param_dict = param_dict
        
        activ = ACTIVS[param_dict['activ']] if param_dict["activ"] in ACTIVS else None
        self.ML = param_dict["out_size"] == ML_OUT_SZ
        self.out_activ = F.sigmoid if self.ML else F.softmax
        criterion = nn.BCELoss() if self.ML else nn.CrossEntropyLoss()
        self.criterion = criterion

        with open(os.path.join(path, "history.json"), 'r') as hist_file:
            hists = json.load(hist_file)

        self.history = hists['history']
        self.trained_epochs = hists['trained_epochs']
        self.conv_epoch = hists['conv_epoch']
        
        self.model = ConvModel(in_size=param_dict["in_size"], filts=param_dict["filts"], kerns=param_dict["kerns"], pad=param_dict["pad"],
                               activ=activ, dense_sz=param_dict["dense_sz"], out_size=param_dict["out_size"], out_activ=self.out_activ,
                               grey_scaled=param_dict["grey_scaled"], square_in=param_dict["square_in"])
        self.l2 = param_dict["l2"]

        weights = torch.load(os.path.join(path, "model_weights.pt"), map_location=self.dev)
        self.load_weights(weights)
        self.min_weights = weights
        
        self.optimiser = optim.Adam(self.model.parameters(), lr=param_dict["lr"], betas=(param_dict["beta_1"], param_dict["beta_2"]), weight_decay=param_dict["l2"])    
        

    def load_weights(self, weights):
        self.model.load_state_dict(weights)

def main():
    """Sample code for loading and running a saved Convolutional Braille Recognition (CBR) model.
    """
    dev = 'cpu'
    
    print("Running sample model loading and predictions...")
    # path to model folder. Should contain `history.json`, `model_weights.pt` and `parameters.json` files.
    model_path = os.path.join('models', 'base_ml_model')
    cbr_model = ModelWrapper(dev=dev) # default parameters are overwritten by load, except for pytorch `dev`
    cbr_model.load(model_path)
    
    ss = 5
    sample_x = np.random.uniform(size=(ss, DEF_IN_SIZE[0], DEF_IN_SIZE[1], DEF_IN_SIZE[2])) # sample image data with float RGB values
    sample_y = np.round(np.random.uniform(size=(ss, ML_OUT_SZ)), 0).astype(int) # sample multilabel target data
    tensor_dataset = DataLoader(
        TensorDataset(torch.Tensor(sample_x).to(dev),
                      torch.Tensor(sample_y).to(dev)),
        batch_size=ss)
    
    predictions, probabilities = cbr_model.predict(tensor_dataset, return_proba=True, verbose=0)
    for i in range(ss):
        target = ''.join(sample_y[i].astype(str))
        pred = ''.join(predictions[i].astype(str))
        probs = np.round(probabilities[i], 2)
        print(f"Sample {i}, target {target}, prediction {pred}.")
        print(f">>> from binary probabilities {probs}")
        print("-"*100)

if __name__ == "__main__":
    main()
