"""Simple training example."""

import torch
import torch.nn as nn
from tqdm import tqdm


def main() -> None:
    """Main training function that demonstrates PyTorch training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Simple model
    model = nn.Linear(10, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Dummy training loop
    for epoch in tqdm(range(10), desc="Training"):
        x = torch.randn(32, 10, device=device)
        y = torch.randn(32, 1, device=device)

        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
