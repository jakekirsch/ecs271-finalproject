import torch


# get model from epoch state 
def get_model_from_checkpoint(experiment_dir, experiment_name, model, epoch):
    path = f"{experiment_dir}/{experiment_name}/checkpoint_epoch_{epoch}.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

