import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    pass

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, epoch, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    model_base_name = f"epoch_{epoch}"

    if best:
        files = os.listdir(checkpoint_dir)
        for file in files:
            if "best" in file:
                file_base_name = file[ : len(file) - 9]
                os.rename(os.path.join(checkpoint_dir, file), os.path.join(checkpoint_dir, file_base_name + ".pth"))
                torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                }, os.path.join(checkpoint_dir, model_base_name + "_best.pth"))
            else:
                file_base_name = file[ : len(file) - 4]
                if file_base_name == model_base_name:
                    os.rename(os.path.join(checkpoint_dir, file), os.path.join(checkpoint_dir, model_base_name + "_best.pth"))
    else:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        }, os.path.join(checkpoint_dir, model_base_name + ".pth"))

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    checkpoint_dir = os.path.join("checkpoints/scr_experiments/", args.experiment_name)
    files = os.listdir(checkpoint_dir)

    best_model = None
    for file in files:
        if "best" in file:
            best_model = file
            break
    
    if best_model == None:
        checkpoint = torch.load(os.path.join(checkpoint_dir, files[-1]))
        print(f"Loaded model {os.path.join(checkpoint_dir, files[-1])}")
    else:
        checkpoint = torch.load(os.path.join(checkpoint_dir, best_model))
        print(f"Loaded model {os.path.join(checkpoint_dir, best_model)}")

    config = T5Config.from_pretrained("t5-small")
    model = T5ForConditionalGeneration(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

