from torch.nn import Parameter, Module
import numpy as np
import torch
import os
from src import Range, set_logger, TensorBoardRunner, check_args, set_seed, load_dataset, load_model


checkpoint_dir_name = 'checkpoints'

def checkpoint_exists(name):
    return os.path.exists(f'{checkpoint_dir_name}/{name}.ckpt')

def mask_exists(name):
    return os.path.exists(f'{checkpoint_dir_name}/{name}.pt')

def save_checkpoint(model, name):
    model.save_checkpoint(f'{checkpoint_dir_name}/{name}.ckpt')

###########MASK#########
def get_mask():
    return torch.load(f'{checkpoint_dir_name}/mask.pt')


def create_mask(model_checkpoint,scouts,name_file, args):
    def _compute_mask(model, scout_dict_list):
        def learning_spread(model, scout):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            scout = scout.to(device)
            ls = torch.mean((scout - model) ** 2, 0)

            if args.guidence_normalization == 'MinMax':
                min_ls = torch.min(ls)
                max_ls = torch.max(ls)
                spread = max_ls - min_ls
                return (ls - min_ls) / spread

            if args.guidence_normalization == 'RobustScaler':
                median_ls = torch.median(ls)
                q1 = torch.quantile(ls, 0.25)
                q3 = torch.quantile(ls, 0.75)
                iqr = q3 - q1

                return (ls - median_ls) / iqr

        out_mask = {}
        model_dict = model.state_dict()

        scouts = {}
        for scout_dict in scout_dict_list:
            for name, param in model.named_parameters():
                scouts.setdefault(name, [])
                scouts[name].append(scout_dict[name])  # TODO: this should be done differently...

        for name, param in model.named_parameters():
            out_mask[name] = learning_spread(model_dict[name], torch.stack(scouts[name]))

        return out_mask

    model, args = load_model(args)

    model.load_state_dict(torch.load(f'{checkpoint_dir_name}/{model_checkpoint}.ckpt'))

    mask = _compute_mask(model, scouts)
    torch.save(mask, f'{checkpoint_dir_name}/mask{name_file}.pt')
    return mask

def predict(model, dataloader, device):
    prediction = []
    with torch.no_grad():
        model.eval()
        for data, _ in dataloader:
            data = data.to(device)
            output = model(data)
            prediction.append(output.gpu().numpy())
    prediction = np.concatenate(prediction, axis=0)
    return prediction




###Possible alternatives
def _get_guidance_matrix(scout_param, model_param: Parameter) -> torch.Tensor:
    learning_spread = torch.mean((scout_param - model_param) ** 2, 0)
    min_ls = torch.min(learning_spread)
    max_ls = torch.max(learning_spread)
    return torch.tensor((learning_spread - min_ls) / (max_ls - min_ls))


def apply_guidance_matrices(model: Module, scouts) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.grad *= _get_guidance_matrix(scouts[name], param)


def debias_layers(model: Module) -> None: #TODO: get a look on what is this
    def debias(param):
        return torch.stack([torch.mean(param, dim=0)] * len(param), dim=0)

    last_layer = list(model.children())[-1]
    last_layer.bias = debias(last_layer.bias)
    last_layer.weight = debias(last_layer.weight)