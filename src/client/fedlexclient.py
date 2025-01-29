import copy
import torch
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
        
        # path for the base checkpoint model
        check_dir = "./checkpoints/"
        self.checkpoint_path = os.path.join(check_dir, f"tl_base{self.args.exp_name}.ckpt")


    def save_client_data(self):
        """Save each client’s training and test data in `result_path/client_data/client_{id}/`."""
        data_dir = os.path.join(self.args.result_path, 'client_data', f'client_{self.id}')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f'Created directory: {data_dir}')
        else:
            print(f'Directory already exists: {data_dir}')

        # Extract training data and labels
        train_data, train_labels = [], []
        for idx in range(len(self.training_set)):
            data, label = self.training_set[idx]
            train_data.append(data)
            train_labels.append(label)

        # Save training data
        train_data_file = os.path.join(data_dir, 'training_data.pt')
        torch.save({'data': train_data, 'labels': train_labels}, train_data_file)
        print(f'Saved training data for client {self.id} at {train_data_file}')

        # Extract test data and labels
        test_data, test_labels = [], []
        for idx in range(len(self.test_set)):
            data, label = self.test_set[idx]
            test_data.append(data)
            test_labels.append(label)

        # Save test data
        test_data_file = os.path.join(data_dir, 'test_data.pt')
        torch.save({'data': test_data, 'labels': test_labels}, test_data_file)
        print(f'Saved test data for client {self.id} at {test_data_file}')


    def find_project_root(self, current_directory, marker):
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
        compression_rate = (
            total_params / (total_params - total_zero_params)
            if total_zero_params != 0
            else 1
        )

        logger.info(f"{title}:")
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Zero parameters: {total_zero_params}")
        logger.info(f"Sparsity: {sparsity*100:.2f}%")
        logger.info(f"Compression rate: {compression_rate:.2f}x")
        logger.info(f"Total size (bytes): {total_size}")
        for name, param in model.named_parameters():
            sp = torch.sum(param == 0).item() / param.nelement()
            logger.info(f"{name}: {param.size()}, sparsity: {sp:.4f}")


    def create_scout(self, model):
        """Create (or load) a base model and exploration mask if not already existing."""

        # ------------------------------------------------------
        # 1. Check if the baseline model exists, otherwise train it once
        # ------------------------------------------------------
        if not checkpoint_exists('tl_base'):
            print('No "tl_base" model found. Creating it...')
            tl_base = model.train()
            model.to(self.args.device)

            optimizer = self.optim(model.parameters(), **self._refine_optim_args(self.args))
            for e in range(1):
                for inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                    outputs = model(inputs)
                    loss = self.criterion()(outputs, targets)

                    for param in model.parameters():
                        param.grad = None

                    loss.backward()
                    optimizer.step()

            torch.save(
                tl_base.state_dict(),
                f'FedLEx/checkpoints/tl_base{self.args.exp_name}.ckpt'
            )
            print("Baseline model 'tl_base' saved.")

        # ------------------------------------------------------
        # 2. Check if a mask is already created
        # ------------------------------------------------------
        if not mask_exists(f'mask{self.args.exp_name}'):
            print('The scout is exploring.... (No mask found.)')
            mm = MetricManager(self.args.eval_metrics)

            # Put model in train mode
            model.train()
            model.to(self.args.device)
            optimizer = self.optim(model.parameters(), **self._refine_optim_args(self.args))

            best_loss = float('inf')  
            patience_counter = 0  
            patience = self.args.Patience_mask  

            # ------------------------------
            #    2a. Train / Explore
            # ------------------------------
            for e in range(self.args.epoochs_mask):
                epoch_correct = 0
                epoch_total = 0
                for inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                    outputs = model(inputs)
                    loss = self.criterion()(outputs, targets)

                    # zero grads
                    for param in model.parameters():
                        param.grad = None

                    # backward + update
                    loss.backward()
                    optimizer.step()

                    # track accuracy
                    predicted = outputs.argmax(dim=1)
                    correct = (predicted == targets).sum().item()
                    total = targets.size(0)
                    epoch_correct += correct
                    epoch_total += total

                    mm.track(loss.item(), outputs, targets)

                # end of epoch
                mm.aggregate(len(self.training_set), e + 1)
                epoch_accuracy = epoch_correct / epoch_total
                print(f'Epoch: {e+1}, Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy:.4f}')

                # 2b. Early stopping logic
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f'Early stopping at epoch {e+1}, best loss was {best_loss:.4f}')
                    break

            # ------------------------------
            #    2c. Build your mask
            # ------------------------------
            ### CHANGES HERE: Build the actual mask dictionary
            ###
            # For example, you might base your mask on some criterion. 
            # Here, we’ll just create an “all-ones” mask as a placeholder:
            mask_dict = {}
            for name, param in model.named_parameters():
                # Example placeholder: fully dense mask
                mask_dict[name] = torch.ones_like(param)

            # 2d. Save the mask to checkpoints
            mask_path = f'FedLEx/checkpoints/mask{self.args.exp_name}.pt'
            torch.save(mask_dict, mask_path)
            print(f"Global exploration mask saved at {mask_path}.")

            # 2e. ALSO SAVE the mask to this client's folder
            data_dir = os.path.join(self.args.result_path, 'client_data', f'client_{self.id}')
            mask_file = os.path.join(data_dir, f'exploration_local_matrix_{self.id}.pt')
            torch.save(mask_dict, mask_file)
            print(f"Exploration mask saved for client {self.id} at {mask_file}.\n")

            # return the mask_dict as well
            return mask_dict
        
        else:
            # Mask already exists, just load it
            mask_path = f'FedLEx/checkpoints/mask{self.args.exp_name}.pt'
            print(f"Mask already exists. Loading from {mask_path}")
            existing_mask = torch.load(mask_path, map_location=self.args.device)
            return existing_mask


    def _refine_optim_args(self, args):
        """Collect only the arguments that `torch.optim.Optimizer` actually needs."""
        required_args = inspect.getfullargspec(self.optim)[0]
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument):
                refined_args[argument] = getattr(args, argument)
        return refined_args


    def _create_dataloader(self, dataset, shuffle):
        if self.args.B == 0:
            self.args.B = len(self.training_set)
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)


    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)

        global_model = copy.deepcopy(self.model)
        for param in global_model.parameters():
            param.requires_grad = False
        
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))

        # Create directory for saving client model & optimizer states
        client_dir = os.path.join(self.args.result_path, f'round_{self.args.curr_round}', f'client_{self.id}')
        if not os.path.exists(client_dir):
            os.makedirs(client_dir)
            print(f'Created directory: {client_dir}')
        else:
            print(f'Directory already exists: {client_dir}')

        # Save model before local training
        torch.save(self.model.state_dict(), os.path.join(client_dir, 'model_before_training.pth'))
        print(f'Saved model before training at {client_dir}')

        # Save optimizer before local training
        torch.save(optimizer.state_dict(), os.path.join(client_dir, 'optimizer_before_training.pth'))
        print(f'Saved optimizer state before training at {client_dir}')

        # ========== LOCAL EPOCHS ========== #
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                # store initial param values
                initial_params = {}
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        initial_params[name] = param.clone()

                # zero grads + backward
                for param in self.model.parameters():
                    param.grad = None
                loss.backward()

                # if mask exists, apply it to grad
                if mask_exists(f'mask{self.args.exp_name}'):
                    global_mask = torch.load(f'FedLEx/checkpoints/mask{self.args.exp_name}.pt')
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name in global_mask:
                            # multiply grad by mask
                            param.grad *= global_mask[name].to(param.device)

                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                optimizer.step()

                # track parameter changes
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        initial_value = initial_params[name]
                        new_value = param.clone()
                        percent_change = torch.abs(new_value - initial_value) / (torch.abs(initial_value) + 1e-8) * 100
                        avg_percent_change = percent_change.detach().cpu().mean().item()
                        print(f"Parameter: {name}, Percentage Change: {avg_percent_change:.2f}%")

                mm.track(loss.item(), outputs, targets)
            # end epoch
            mm.aggregate(len(self.training_set), e + 1)

        # Save model after local training
        torch.save(self.model.state_dict(), os.path.join(client_dir, 'model_after_training.pth'))
        print(f'Saved model after training at {client_dir}')

        # Save optimizer after local training
        torch.save(optimizer.state_dict(), os.path.join(client_dir, 'optimizer_after_training.pth'))
        print(f'Saved optimizer state after training at {client_dir}')

        return mm.results


    @torch.inference_mode()
    def evaluate(self):
        if self.args._train_only:  # if `test_fraction == 0`
            return {'loss': -1, 'metrics': {'none': -1}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)
            mm.track(loss.item(), outputs, targets)
        mm.aggregate(len(self.test_set))
        return mm.results


    def download(self, model):
        """Download (copy) global model from the server."""
        self.model = copy.deepcopy(model)

    def upload(self):
        """Upload local model parameters to the server."""
        self.model.to(self.args.device)
        return self.model.named_parameters()

    def __len__(self):
        return len(self.training_set), len(self.test_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'
