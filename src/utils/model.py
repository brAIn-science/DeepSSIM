import os
import torch 

# This function saves the best model in TorchScript format along with the optimizer state.
# A confirmation message is printed upon successful completion.
# Author: Antonio Scardace

def save_model_and_optimizer(net: torch.nn.Module, optim: torch.optim.Optimizer, path: str) -> None:
    scripted_model = torch.jit.script(net)
    scripted_model.save(os.path.join(path, 'best_model_jit.pt'))
    torch.save(optim.state_dict(), os.path.join(path, 'optimizer_state.pth'))
    print('Best model and optimizer state saved successfully.')

# These functions handle the loading of a TorchScript model and its associated optimizer state.
# The optimizer class must be provided to reconstruct the optimizer.
# This setup enables resuming inference or training from previously saved checkpoints.
# Author: Antonio Scardace

def load_model(path: str, device: str) -> torch.nn.Module:
    model = torch.jit.load(os.path.join(path, 'best_model_jit.pt'), map_location=device)
    model.to(device)
    model.eval()
    return model

def load_optimizer(model: torch.nn.Module, path: str, device: str, optimizer_class) -> torch.optim.Optimizer:
    optimizer = optimizer_class(model.parameters())
    optimizer_path = os.path.join(path, 'optimizer_state.pth')
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    return optimizer

def load_model_and_optimizer_jit(path: str, device: str, optimizer_class) -> tuple:
    model = load_model(path, device)
    optimizer = load_optimizer(model, path, device, optimizer_class)
    print('TorchScript model and optimizer loaded successfully.')
    return model, optimizer