import torch
from architecture.istvt import ISTVT

def count_parameters(model):
    """Count trainable parameters."""
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ISTVT().to(device)

    print("=" * 70)
    print("ISTVT MODEL SUMMARY")
    print("=" * 70)

    print(model)

    print("\nTrainable Parameters: {:,}".format(count_parameters(model)))

    x = torch.randn(
        2,
        6,
        3,
        128,
        128,
        device=device
    )

    with torch.no_grad():
        y = model(x)

    print("\nDummy Input :", x.shape)
    print("Output Shape:", y.shape)