import torch
import os


def save_checkpoint(state, is_best, out_dir='results', filename='checkpoint.pt'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(out_dir, 'best_model.pt')
        torch.save(state['model_state_dict'], best_path)


def load_model_weights(model, path, device='cpu'):
    state = torch.load(path, map_location=device)
    # handle both full checkpoint and raw state_dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    return model
