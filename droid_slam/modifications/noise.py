import torch

def add_gaussian_noise(image: torch.Tensor, std: float) -> torch.Tensor:
    """
    Add Gaussian noise to a (3, H, W) image tensor with values in [0, 255] and dtype uint8.
    
    Args:
        image (torch.Tensor): Image tensor of shape (3, H, W) and dtype uint8.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy image tensor, same shape and dtype, clamped to [0, 255].
    """
    # Convert to float for noise addition
    image_float = image.to(torch.float32)
    noise = torch.randn_like(image_float) * std
    noisy_image = image_float + noise
    noisy_image = torch.clamp(noisy_image, 0, 255)

    # Convert back to original dtype
    return noisy_image.to(torch.uint8)