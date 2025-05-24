"""
This module defines the AdaptiveSuperpixelParamsPredictor, a neural network
designed to predict optimal parameters (K and m) for SLIC-based superpixel
algorithms. It also includes a helper function, predict_adaptive_params,
to facilitate the use of this predictor and scale its outputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSuperpixelParamsPredictor(nn.Module):
    """
    A neural network model to predict parameters K (number of superpixels)
    and m (compactness factor) for an adaptive superpixel algorithm.

    The model consists of a small CNN encoder followed by an MLP.
    The CNN processes the input image to extract features, and the MLP
    predicts the raw values for K and m.

    Input:
        image_tensor (torch.Tensor): A batch of images with shape [B, C, H, W].

    Output:
        raw_predictions (torch.Tensor): A tensor of shape [B, 2] containing
                                        the raw (unscaled) predictions for K and m.

    Note on Differentiability:
        The parameters K (number of superpixels) and m (compactness factor)
        predicted by this model are typically used by SLIC-based superpixel
        algorithms (e.g., FastSLIC, skimage.segmentation.slic). These
        traditional SLIC algorithms are non-differentiable iterative processes.
        Therefore, gradients cannot flow back from the superpixel generation
        step to this predictor model for end-to-end training of K and m
        *through the SLIC algorithm itself*.

        This predictor model is typically trained with a supervised loss if
        ground-truth K and m values are available, or through other means
        like reinforcement learning if a reward function based on downstream
        task performance can be defined.

        If end-to-end differentiability for K and m prediction *through* the
        superpixel generation process is desired in future research, alternative
        approaches would be necessary. These might include:
        1.  Using policy gradient methods (e.g., REINFORCE) if K and m
            prediction is framed as a stochastic policy.
        2.  Employing Gumbel-Softmax or similar techniques if K and/or m are
            treated as discrete choices from a learnable distribution.
        3.  Replacing the traditional SLIC algorithm with a differentiable
            superpixel generation algorithm.
    """
    def __init__(self):
        super().__init__()

        # CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # Adaptive Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2)  # Raw outputs for K and m
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Raw predictions for K and m, shape [B, 2].
        """
        x = self.encoder(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.mlp(x)
        return x

def predict_adaptive_params(
    predictor_model: AdaptiveSuperpixelParamsPredictor,
    image_tensor: torch.Tensor,
    k_min: int = 100,
    k_max: int = 300,
    m_min: float = 1.0,
    m_max: float = 20.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predicts adaptive superpixel parameters K (number of superpixels) and
    m (compactness factor) using a trained predictor model and scales them
    to a specified range.

    Args:
        predictor_model (AdaptiveSuperpixelParamsPredictor):
            The trained model for predicting K and m.
        image_tensor (torch.Tensor):
            Input image tensor of shape [B, C, H, W].
        k_min (int):
            Minimum desired value for K.
        k_max (int):
            Maximum desired value for K.
        m_min (float):
            Minimum desired value for m.
        m_max (float):
            Maximum desired value for m.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - k_pred (torch.Tensor): Predicted K values, shape [B].
            - m_pred (torch.Tensor): Predicted m values, shape [B].
    """
    if not isinstance(predictor_model, AdaptiveSuperpixelParamsPredictor):
        raise TypeError("predictor_model must be an instance of AdaptiveSuperpixelParamsPredictor")
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a torch.Tensor")
    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must have 4 dimensions [B, C, H, W]")

    predictor_model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        raw_predictions = predictor_model(image_tensor)

    k_raw, m_raw = raw_predictions[:, 0], raw_predictions[:, 1]

    # Sigmoid scaling
    k_pred = k_min + (k_max - k_min) * torch.sigmoid(k_raw)
    m_pred = m_min + (m_max - m_min) * torch.sigmoid(m_raw)

    return k_pred, m_pred

if __name__ == '__main__':
    # Example Usage (requires torch to be installed and an image)
    # This is a placeholder and won't run without further setup.
    print("AdaptiveSuperpixelParamsPredictor class and predict_adaptive_params function defined.")

    # Create a dummy model
    model = AdaptiveSuperpixelParamsPredictor()
    print("\nModel Architecture:")
    print(model)

    # Create a dummy input tensor (batch of 1, 3 channels, 128x128 image)
    # Requires torch to be installed
    try:
        dummy_image = torch.randn(1, 3, 128, 128)
        k_predicted, m_predicted = predict_adaptive_params(model, dummy_image)
        print(f"\nDummy prediction for K: {k_predicted.item()}")
        print(f"Dummy prediction for m: {m_predicted.item()}")

        dummy_images_batch = torch.randn(4, 3, 256, 256)
        k_batch_predicted, m_batch_predicted = predict_adaptive_params(model, dummy_images_batch, k_min=50, k_max=500, m_min=0.5, m_max=40.0)
        print(f"\nBatch predictions for K: {k_batch_predicted}")
        print(f"Batch predictions for m: {m_batch_predicted}")

    except ImportError:
        print("\nPyTorch is not installed. Skipping example usage with tensors.")
    except Exception as e:
        print(f"\nAn error occurred during example usage: {e}")

    # Test type checking
    try:
        predict_adaptive_params("not_a_model", torch.randn(1,3,32,32))
    except TypeError as e:
        print(f"\nCaught expected TypeError for model: {e}")

    try:
        predict_adaptive_params(model, "not_a_tensor")
    except TypeError as e:
        print(f"\nCaught expected TypeError for tensor: {e}")

    try:
        predict_adaptive_params(model, torch.randn(1,3,32)) # Wrong dimensions
    except ValueError as e:
        print(f"\nCaught expected ValueError for tensor dimensions: {e}")
