import copy
import torch
import torch.nn.utils.prune as prune
import inspect
import numpy as np
import logging

from .baseclient import BaseClient
from src import MetricManager
from src.GTL_utils import checkpoint_exists, mask_exists
import time
import matplotlib.pyplot as plt

import math
import os
logger = logging.getLogger(__name__)


class FedlexClient(BaseClient):
    def __init__(self, id, args, training_set, test_set):
        super(FedlexClient, self).__init__()
        
        self.id = id  # Assign the client ID here
        
        self.args = args
        self.training_set = training_set
        self.test_set = test_set

        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)
        
        # Save the client's data
        self.save_client_data()
        
        # Get the absolute path to the directory containing this script
        self.base_dir = os.path.dirname(os.path.realpath(__file__)) 
        check_dir = "checkpoints/"
        self.checkpoint_path = os.path.join(check_dir, f"tl_base{self.args.exp_name}.ckpt")
        
    def save_client_data(self):
        # Create directory to save the data
        data_dir = os.path.join(self.args.result_path, 'client_data', f'client_{self.id}')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f'Created directory: {data_dir}')
        else:
            print(f'Directory already exists: {data_dir}')

        # Extract training data and labels
        train_data = []
        train_labels = []
        for idx in range(len(self.training_set)):
            data, label = self.training_set[idx]
            train_data.append(data)
            train_labels.append(label)

        # Save training data
        train_data_file = os.path.join(data_dir, 'training_data.pt')
        torch.save({'data': train_data, 'labels': train_labels}, train_data_file)
        print(f'Saved training data for client {self.id} at {train_data_file}')

        # Extract test data and labels
        test_data = []
        test_labels = []
        for idx in range(len(self.test_set)):
            data, label = self.test_set[idx]
            test_data.append(data)
            test_labels.append(label)

        # Save test data
        test_data_file = os.path.join(data_dir, 'test_data.pt')
        torch.save({'data': test_data, 'labels': test_labels}, test_data_file)
        print(f'Saved test data for client {self.id} at {test_data_file}') 
	
    def find_project_root(self,current_directory, marker):
        root = current_directory
        while root != '/':  # stop at the root of the filesystem
            if marker in os.listdir(root):
                return root
            root = os.path.dirname(root)
        raise FileNotFoundError("Project root marker not found")
                


    def print_model_info(self, model, title="Model information"):
        total_params = sum(p.numel() for p in model.parameters())
        total_zero_params = sum((p == 0).sum().item() for p in model.parameters())
        total_size = sum(p.element_size() * p.nelement() for p in model.parameters())
        sparsity = total_zero_params / total_params
        compression_rate = total_params / (total_params - total_zero_params) if total_zero_params else 1

        logger.info(f"{title}:")
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Zero parameters: {total_zero_params}")
        logger.info(f"Sparsity: {sparsity*100:.2f}%")
        logger.info(f"Compression rate: {compression_rate:.2f}x")
        logger.info(f"Total size (bytes): {total_size}")
        for name, param in model.named_parameters():
            logger.info(f"{name}: {param.size()}, sparsity: {torch.sum(param == 0) / param.nelement()}")



    def create_scout(self, model):
        if not checkpoint_exists('tl_base'):
            print('wait! we dont have a tl_base model...letz create it!')
            tl_base = model.train()
            model.to(self.args.device)
            optimizer = self.optim(model.parameters(), **self._refine_optim_args(self.args))
            for e in range(1) :
                for inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                    outputs = model(inputs)
                    loss = self.criterion()(outputs, targets)

                    for param in model.parameters():
                        param.grad = None

                    loss.backward()
                    optimizer.step()  # !!! the application of those deltas is happening here

            torch.save(tl_base.state_dict(),
             f'checkpoints/tl_base{self.args.exp_name}.ckpt')

        if not mask_exists(f'mask{self.args.exp_name}'):
            print('The scout is exploring....')
            mm = MetricManager(self.args.eval_metrics)
            mask = model.train()
            model.to(self.args.device)
            optimizer = self.optim(model.parameters(), **self._refine_optim_args(self.args))

            best_loss = float('inf')  # initialize best loss as infinity
            patience_counter = 0  # counter for early stopping
            patience = self.args.Patience_mask  # number of epochs to wait before stopping

            # for e in range(self.args.epoochs_mask):
            for e in range(self.args.epoochs_mask):
                epoch_correct = 0  # number of correct predictions in this epoch
                epoch_total = 0  # total number of predictions in this epoch
                for inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                    outputs = model(inputs)
                    loss = self.criterion()(outputs, targets)

                    for param in model.parameters():
                        param.grad = None

                    loss.backward()
                    optimizer.step() # !!! the application of those deltas is happening here

                    # calculate accuracy
                    predicted = outputs.argmax(dim=1)
                    correct = (predicted == targets).sum().item()  # number of correct predictions
                    total = targets.shape[0]  # total number of predictions
                    epoch_correct += correct
                    epoch_total += total


                    mm.track(loss.item(), outputs, targets)
                else:
                    mm.aggregate(len(self.training_set), e + 1)

                epoch_accuracy = epoch_correct / epoch_total  # calculate accuracy for this epoch
                print(f'Epoch: {e + 1}, Loss: {loss.item()}, Accuracy: {epoch_accuracy}')  # print the epoch number, loss, and accuracy

                # early stopping
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0  # reset counter when loss improves
                else:
                    patience_counter += 1  # increment counter when loss does not improve

                if patience_counter >= patience:  # if counter reaches the threshold
                    print(f'Early stopping at epoch {e + 1}, best loss was {best_loss}')
                    break  # stop the training

            return mask


    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument):
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0 :
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)


    def update(self):

        #def get_nested_module(model, module_path): #I dont remember for what is this, might be AUTOFLIP
            #logger.info("Module path:", module_path)
            #modules = module_path.split('.')
            #for module in modules:
                #logger.info("Current module:", module)
                #if hasattr(model, module):
                    #model = getattr(model, module)
                    #logger.info("Submodule found:", model)
                #else:
                    #logger.info("Submodule not found:", module)
                    #return None
            #return model

        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        global_model = copy.deepcopy(self.model)
        for param in global_model.parameters():
            param.requires_grad = False
        
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        
        # Create directory for saving client models and optimizer states
        client_dir = os.path.join(self.args.result_path, f'round_{self.args.curr_round}', f'client_{self.id}')
        if not os.path.exists(client_dir):
            os.makedirs(client_dir)
            print(f'Created directory: {client_dir}')
        else:
            print(f'Directory already exists: {client_dir}')

        # Save model parameters before local training
        torch.save(self.model.state_dict(), os.path.join(client_dir, 'model_before_training.pth'))
        print(f'Saved model before training at {client_dir}')

        # Save optimizer state before local training
        torch.save(optimizer.state_dict(), os.path.join(client_dir, 'optimizer_before_training.pth'))
        print(f'Saved optimizer state before training at {client_dir}')

        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)
                
                initial_params = {}  # Store initial parameter values
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        initial_params[name] = param.clone()

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                
                
                if mask_exists(f'mask{self.args.exp_name}'):
                    mask = torch.load(f'/home/cinterno/storage/FL2/Federated-Learning-in-PyTorch/checkpoints/mask{self.args.exp_name}.pt')
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            param.grad *= torch.tensor(mask[name], device=param.device)
                
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                optimizer.step()
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        initial_value = initial_params[name]
                        new_value = param.clone()  # Get the updated value after the gradient update
                        percent_change = torch.abs(new_value - initial_value) / (torch.abs(initial_value) + 1e-8) * 100
                        percent_change = percent_change.detach().cpu().numpy()  # Convert to numpy array for easier processing
                        avg_percent_change = np.mean(percent_change)

                        print(f"Parameter: {name}, Percentage Change: {avg_percent_change:.2f}%")
                
                #Client Pruning
                #if mask_exists(f'mask{self.args.exp_name}'):
                    #if self.args.mask_pruining == 'True': #AutoFLIP

                        #updated_mask = torch.load(f'checkpoints/InitPruinedGlobalModel/mask{self.args.exp_name}.pt')

                        #for path, mask_value in updated_mask.items():
                            #Split the path to access specific submodule and parameter
                            #submodule_path, param_name = path.rsplit('.', 1)
                            #submodule = get_nested_module(self.model, submodule_path)

                            #if submodule:
                                #mask_tensor = torch.tensor(mask_value, device=submodule.weight.device)
                                #Ensure the mask is correctly shaped for the parameter it's applied to
                                #target_param = getattr(submodule, param_name)
                                #assert mask_tensor.shape == target_param.shape, f"Shape mismatch for {path}: {mask_tensor.shape} vs {target_param.shape}"

                                #try:
                                    #Apply pruning to the specific parameter of the submodule
                                    #prune.custom_from_mask(submodule, name=param_name, mask=mask_tensor)
                                #except AttributeError as e:
                                    #logger.info(f"Error applying pruning to {path}: {e}")
                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
                
        # Save model parameters after local training
        torch.save(self.model.state_dict(), os.path.join(client_dir, 'model_after_training.pth'))
        print(f'Saved model after training at {client_dir}')

        # Save optimizer state after local training
        torch.save(optimizer.state_dict(), os.path.join(client_dir, 'optimizer_after_training.pth'))
        print(f'Saved optimizer state after training at {client_dir}')
        
        #Other AutoFLIP stuff
        #def finalize_pruning(model):
            #for name, module in model.named_modules():
                #Check if the module has the 'weight_orig' attribute, indicating pruning was applied to the weight
                #if hasattr(module, 'weight_orig'):
                    #Remove pruning reparameterization for 'weight'
                    #prune.remove(module, 'weight')

                #Similarly, check if pruning was applied to the bias
                #if hasattr(module, 'bias_orig'):
                    #Remove pruning reparameterization for 'bias'
                    #prune.remove(module, 'bias')

            #return model

        #self.model = finalize_pruning(self.model)
        #self.print_model_info(self.model, f"Client{self.id} Model after pruning")
        #torch.save(self.model.state_dict(), f'/home/cinterno/storage/FL2/Federated-Learning-in-PyTorch/checkpoints/ClientPruinedGModel/Client{self.id}for{self.args.exp_name}.pt')
        return mm.results

    @torch.inference_mode()
    def evaluate(self):

        if self.args._train_only:  # `args.test_fraction` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.test_set))
        return mm.results

    def download(self, model):

        self.model = copy.deepcopy(model)

    def upload(self):
        self.model.to('cuda')
        return self.model.named_parameters()

    def __len__(self):
        return len(self.training_set), len(self.test_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'

