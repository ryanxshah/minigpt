import torch

util_hyperparams = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
    "seed": 0
}

model_hyperparams = {
    "batch_size": 32,
    "seq_len": 8,
    "d_model": 32,
    "num_heads": 4
}

training_hyperparams = {
    "learning_rate": 1e-3,
    "max_iters": 300,
    "eval_iters": 200,
    "eval_interval": 5000
}