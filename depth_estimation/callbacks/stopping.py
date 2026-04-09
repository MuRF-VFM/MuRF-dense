class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-5, rmse_threshold=0.3899):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.rmse_threshold = rmse_threshold
        self.counter = 0

    def __call__(self, current_loss, current_rmse):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Loss has not improved by {self.min_delta} for {self.patience} epochs. Stopping training.")
            return True
        if current_rmse < self.rmse_threshold:
            print("RMSE threshold reached. Stopping training.")
            return True

        return False