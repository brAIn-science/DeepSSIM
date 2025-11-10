import wandb

# This function logs the loss and performance value for visualization in Wandb.
# It also prints this value to console.
# Author: Antonio Scardace
    
def log_metrics(mode: str, epoch: int, loss_mean: float, loss_std: float, mae: float) -> None:
    print('Loss [MSE] =', loss_mean)
    print('Standard Deviation [MSE] =', loss_std)
    print('Performance [MAE] =', mae)

    wandb.log(step=epoch, data={
        mode + '/loss_mean': loss_mean,
        mode + '/loss_std': loss_std,
        mode + '/mae': mae
    })