"""
Unit tests for the AdaptiveSuperpixelParamsPredictor and predict_adaptive_params function
from adaptive_superpixel.py.
"""
import unittest
import torch
from adaptive_superpixel import AdaptiveSuperpixelParamsPredictor, predict_adaptive_params

class TestAdaptiveSuperpixel(unittest.TestCase):
    """
    Test suite for adaptive superpixel parameter prediction components.
    """

    def test_predictor_creation(self):
        """
        Tests the instantiation of the AdaptiveSuperpixelParamsPredictor.
        """
        model = AdaptiveSuperpixelParamsPredictor()
        self.assertIsInstance(model, torch.nn.Module, "Model should be an instance of torch.nn.Module")

    def test_predictor_forward_pass(self):
        """
        Tests the forward pass of the AdaptiveSuperpixelParamsPredictor
        and checks the output shape.
        """
        model = AdaptiveSuperpixelParamsPredictor()
        model.eval() # Set to eval mode for consistent behavior, especially with BatchNorm layers.
        
        batch_size = 2
        dummy_tensor = torch.randn(batch_size, 3, 224, 224) # Batch of 2 images
        
        raw_predictions = model(dummy_tensor)
        
        expected_shape = (batch_size, 2)
        self.assertEqual(raw_predictions.shape, expected_shape,
                         f"Model output shape should be {expected_shape}, but got {raw_predictions.shape}")

    def test_predict_adaptive_params_output_shape_and_range(self):
        """
        Tests the predict_adaptive_params function for correct output shapes,
        value ranges, and dtypes.
        """
        model = AdaptiveSuperpixelParamsPredictor()
        model.eval()

        batch_size = 4
        k_min, k_max = 100, 300
        m_min, m_max = 1.0, 20.0
        
        dummy_tensor = torch.randn(batch_size, 3, 64, 64) # Batch of 4 images

        k_preds, m_preds = predict_adaptive_params(model, dummy_tensor, k_min, k_max, m_min, m_max)

        # Shape Checks
        self.assertEqual(k_preds.shape, (batch_size,),
                         f"k_preds shape should be ({batch_size},), but got {k_preds.shape}")
        self.assertEqual(m_preds.shape, (batch_size,),
                         f"m_preds shape should be ({batch_size},), but got {m_preds.shape}")

        # Range Checks
        self.assertTrue(torch.all(k_preds >= k_min), f"Not all k_preds are >= {k_min}. Values: {k_preds}")
        self.assertTrue(torch.all(k_preds <= k_max), f"Not all k_preds are <= {k_max}. Values: {k_preds}")
        self.assertTrue(torch.all(m_preds >= m_min), f"Not all m_preds are >= {m_min}. Values: {m_preds}")
        self.assertTrue(torch.all(m_preds <= m_max), f"Not all m_preds are <= {m_max}. Values: {m_preds}")
        
        # Type Checks
        self.assertEqual(k_preds.dtype, torch.float32, f"k_preds dtype should be torch.float32, but got {k_preds.dtype}")
        self.assertEqual(m_preds.dtype, torch.float32, f"m_preds dtype should be torch.float32, but got {m_preds.dtype}")

    def test_predict_adaptive_params_different_input_sizes(self):
        """
        Tests predict_adaptive_params with a different H, W for the input image
        to ensure AdaptiveAvgPool2d handles it correctly.
        """
        model = AdaptiveSuperpixelParamsPredictor()
        model.eval()

        batch_size = 1
        k_min, k_max = 50, 500
        m_min, m_max = 0.5, 40.0
        
        # Non-square image, different from training/typical size
        dummy_tensor = torch.randn(batch_size, 3, 128, 160) 

        k_preds, m_preds = predict_adaptive_params(model, dummy_tensor, k_min, k_max, m_min, m_max)

        # Shape Checks
        self.assertEqual(k_preds.shape, (batch_size,),
                         f"k_preds shape should be ({batch_size},) for different input size, but got {k_preds.shape}")
        self.assertEqual(m_preds.shape, (batch_size,),
                         f"m_preds shape should be ({batch_size},) for different input size, but got {m_preds.shape}")

        # Range Checks
        self.assertTrue(torch.all(k_preds >= k_min), f"Not all k_preds are >= {k_min} for different input size. Values: {k_preds}")
        self.assertTrue(torch.all(k_preds <= k_max), f"Not all k_preds are <= {k_max} for different input size. Values: {k_preds}")
        self.assertTrue(torch.all(m_preds >= m_min), f"Not all m_preds are >= {m_min} for different input size. Values: {m_preds}")
        self.assertTrue(torch.all(m_preds <= m_max), f"Not all m_preds are <= {m_max} for different input size. Values: {m_preds}")

    def test_predict_adaptive_params_type_errors(self):
        """
        Tests predict_adaptive_params for correct TypeError exceptions
        with invalid input types.
        """
        model = AdaptiveSuperpixelParamsPredictor()
        dummy_tensor = torch.randn(1, 3, 32, 32)

        with self.assertRaisesRegex(TypeError, "predictor_model must be an instance of AdaptiveSuperpixelParamsPredictor"):
            predict_adaptive_params("not_a_model", dummy_tensor)

        with self.assertRaisesRegex(TypeError, "image_tensor must be a torch.Tensor"):
            predict_adaptive_params(model, "not_a_tensor") # type: ignore

        with self.assertRaisesRegex(ValueError, "image_tensor must have 4 dimensions"):
            predict_adaptive_params(model, torch.randn(3, 32, 32)) # type: ignore


if __name__ == '__main__':
    unittest.main()
