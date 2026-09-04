import wandb

# This function outputs the computed epoch metrics to the console and logs them to Weights & Biases for experiment tracking.
# It tracks the Mean Squared Error (MSE) loss, its standard deviation, and the Mean Absolute Error (MAE).
# Author: Antonio Scardace
    
def log_metrics(mode: str, epoch: int, loss_mean: float, loss_std: float, mae: float) -> None:
    
    print('Loss [MSE] =', loss_mean)
    print('Standard Deviation [MSE] =', loss_std)
    print('Performance [MAE] =', mae)

    wandb.log(data={
        mode + '/loss_mean': loss_mean,
        mode + '/loss_std': loss_std,
        mode + '/mae': mae,
        'epoch': epoch,
    })