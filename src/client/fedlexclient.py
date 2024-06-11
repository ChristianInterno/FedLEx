import copy
import torch
import inspect
import numpy as np

from .baseclient import BaseClient
from src import MetricManager
from src.GTL_utils import checkpoint_exists, mask_exists

import os

class FedlexClient(BaseClient):    
    def __init__(self, args, training_set, test_set):
        super(FedlexClient, self).__init__()

        self.args = args
        self.training_set = training_set
        self.test_set = test_set

        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

	# Get the absolute path to the directory containing this script
        self.base_dir = os.path.dirname(os.path.realpath(__file__)) 
        check_dir = "checkpoints/"
        self.checkpoint_path = os.path.join(check_dir, f"tl_base{self.args.exp_name}.ckpt")
	
    def find_project_root(self,current_directory, marker):
        root = current_directory
        while root != '/':  # stop at the root of the filesystem
            if marker in os.listdir(root):
                return root
            root = os.path.dirname(root)
        raise FileNotFoundError("Project root marker not found")
                


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

            torch.save(tl_base.state_dict(),self.checkpoint_path)

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
                    optimizer.step()  # !!! the application of those deltas is happening here

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
                print(
                    f'Epoch: {e + 1}, Loss: {loss.item()}, Accuracy: {epoch_accuracy}')  # print the epoch number, loss, and accuracy

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
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))

        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                for param in self.model.parameters():
                    param.grad = None

                loss.backward()
                initial_params = {}  # Store initial parameter values
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        initial_params[name] = param.clone()

                if mask_exists(f'mask{self.args.exp_name}'):
                    mask = torch.load(self.checkpoint_path)

                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            param.grad *= torch.tensor(mask[name], device=param.device)

                    optimizer.step()# !!! the application of those deltas is happening here

                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            initial_value = initial_params[name]
                            new_value = param.clone()  # Get the updated value after the gradient update
                            percent_change = torch.abs(new_value - initial_value) / (torch.abs(initial_value) + 1e-8) * 100
                            percent_change = percent_change.detach().cpu().numpy()  # Convert to numpy array for easier processing
                            avg_percent_change = np.mean(percent_change)

                            #print(f"Parameter: {name}, Percentage Change: {avg_percent_change:.2f}%")

                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)

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
        self.model.to('cpu')
        return self.model.named_parameters()

    def __len__(self):
        return len(self.training_set), len(self.test_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'
