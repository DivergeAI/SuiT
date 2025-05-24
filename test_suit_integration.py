"""
Unit tests for the SuperpixelVisionTransformer model in suit.py.
Focuses on integration aspects like token shapes in different superpixel modes.
"""
import unittest
import torch
from types import SimpleNamespace
import suit # Assuming suit.py is in the python path and contains model registrations

class TestSuitModelIntegration(unittest.TestCase):
    """
    Test suite for SuperpixelVisionTransformer integration, particularly
    token generation in various operational modes.
    """

    def setUp(self):
        """Define common parameters for tests."""
        self.batch_size = 2
        self.img_size = 224 # Default image size
        self.default_downsample = 2 # Default downsample factor in SuiT models

    def _create_model(self, args_override: dict[str, any]):
        """
        Helper function to create a SuperpixelVisionTransformer model (suit_tiny_224)
        with specified arguments.
        """
        # Base args, reflecting defaults in main.py and suit_tiny_224
        args = SimpleNamespace(
            # Image/Input related
            img_size=self.img_size,
            input_size=self.img_size, # Timm's create_model uses input_size
            
            # Core ViT / SuiT parameters for suit_tiny_224
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.,
            qkv_bias=True, # Default for many ViTs
            
            # SuiT specific feature extraction and tokenization parameters
            base_dim=48, # For suit_tiny
            downsample=self.default_downsample,
            aggregate=['max', 'avg'],
            pe_type='ff',
            pe_injection='concat',
            use_proj=True,
            
            # General training args (less critical for model structure but good to have)
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer='LayerNorm', # Passed as string to timm, then resolved
            act_layer='GELU',      # Passed as string
            mlp_layer='Mlp',       # Passed as string
            init_values=None,
            qk_norm=False,
            weight_init='',        # Default in timm ViT
            fix_init=False,

            # New mode-controlling arguments
            hierarchical_superpixels=False,
            num_scales=0,
            hierarchical_ks=[64, 128, 256] # Default if hierarchical is True
        )

        # Apply overrides
        for k, v in args_override.items():
            setattr(args, k, v)

        # Prepare kwargs for the suit.suit_tiny_224 registration function
        # These should match what SuperpixelVisionTransformer.__init__ and suit_tiny_224 expect
        model_kwargs = {
            'img_size': args.input_size, # suit.py uses img_size, create_model uses input_size
            'embed_dim': args.embed_dim,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'mlp_ratio': args.mlp_ratio,
            'qkv_bias': args.qkv_bias,
            'base_dim': args.base_dim,
            'downsample': args.downsample,
            'aggregate': args.aggregate,
            'pe_type': args.pe_type,
            'pe_injection': args.pe_injection,
            'use_proj': args.use_proj,
            'hierarchical_superpixels': args.hierarchical_superpixels,
            'num_scales': args.num_scales,
            # These might be passed via **kwargs in suit_tiny_224 to VisionTransformer directly
            'drop_rate': args.drop_rate,
            'attn_drop_rate': args.attn_drop_rate,
            'drop_path_rate': args.drop_path_rate,
            # norm_layer, act_layer, etc. are typically handled by partials or string resolution in timm
        }
        
        # In suit.py, the registration functions like suit_tiny_224 now accept
        # hierarchical_superpixels and num_scales.
        model = suit.suit_tiny_224(**model_kwargs)
        model.eval()
        return model

    def test_fixed_mode_token_shape(self):
        """
        Tests token shape in fixed superpixel mode.
        """
        args_override = {'hierarchical_superpixels': False, 'num_scales': 0}
        model = self._create_model(args_override)
        
        dummy_images = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        
        # Superpixel labels are typically smaller due to downsampling in dataset processing
        spix_h = self.img_size // self.default_downsample
        spix_w = self.img_size // self.default_downsample
        num_fixed_superpixels = 196 # Example K value
        dummy_spix_labels = torch.randint(0, num_fixed_superpixels, 
                                          (self.batch_size, 1, spix_h, spix_w))

        tokens_output, _ = model.prepare_tokens(dummy_images, dummy_spix_labels) # Use prepare_tokens, forward_features includes blocks

        self.assertEqual(tokens_output.shape[0], self.batch_size, "Batch size mismatch in output tokens.")
        self.assertEqual(tokens_output.shape[2], model.embed_dim, "Embedding dimension mismatch in output tokens.")
        # Number of tokens = actual superpixels + 1 (CLS token)
        # Actual superpixels can be <= num_fixed_superpixels
        self.assertGreaterEqual(tokens_output.shape[1], 1, "Should have at least CLS token.")
        self.assertLessEqual(tokens_output.shape[1], num_fixed_superpixels + 1, 
                             "Number of tokens exceeds max expected superpixels + CLS.")

    def test_hierarchical_mode_token_shape(self):
        """
        Tests token shape in hierarchical superpixel mode.
        """
        hierarchical_ks_test = [64, 128] # Test with 2 scales
        args_override = {
            'hierarchical_superpixels': True,
            'num_scales': len(hierarchical_ks_test),
            'hierarchical_ks': hierarchical_ks_test # This arg is for dataset, model gets num_scales
        }
        model = self._create_model(args_override)

        dummy_images = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        
        spix_h = self.img_size // self.default_downsample
        spix_w = self.img_size // self.default_downsample
        
        hierarchical_spix_labels = [
            torch.randint(0, k_val, (self.batch_size, 1, spix_h, spix_w)) 
            for k_val in hierarchical_ks_test
        ]

        tokens_output, _ = model.prepare_tokens(dummy_images, hierarchical_spix_labels)

        self.assertEqual(tokens_output.shape[0], self.batch_size, "Batch size mismatch in hierarchical output.")
        self.assertEqual(tokens_output.shape[2], model.embed_dim, "Embedding dimension mismatch in hierarchical output.")
        
        max_expected_total_superpixels = sum(hierarchical_ks_test)
        self.assertGreaterEqual(tokens_output.shape[1], 1, "Should have at least CLS token in hierarchical.")
        self.assertLessEqual(tokens_output.shape[1], max_expected_total_superpixels + 1,
                             "Number of tokens exceeds sum of hierarchical Ks + CLS.")

    def test_adaptive_mode_token_shape(self):
        """
        Tests token shape in adaptive superpixel mode.
        From the model's perspective, this is like fixed mode as it receives a single label tensor.
        """
        args_override = {'hierarchical_superpixels': False, 'num_scales': 0}
        model = self._create_model(args_override)
        
        dummy_images = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        
        spix_h = self.img_size // self.default_downsample
        spix_w = self.img_size // self.default_downsample
        # K would be determined adaptively by the dataset, here we simulate one possible K
        adaptive_k_example = 150 
        dummy_spix_labels = torch.randint(0, adaptive_k_example, 
                                          (self.batch_size, 1, spix_h, spix_w))

        tokens_output, _ = model.prepare_tokens(dummy_images, dummy_spix_labels)

        self.assertEqual(tokens_output.shape[0], self.batch_size, "Batch size mismatch in adaptive mode output.")
        self.assertEqual(tokens_output.shape[2], model.embed_dim, "Embedding dimension mismatch in adaptive mode output.")
        self.assertGreaterEqual(tokens_output.shape[1], 1, "Should have at least CLS token in adaptive mode.")
        self.assertLessEqual(tokens_output.shape[1], adaptive_k_example + 1,
                             "Number of tokens exceeds adaptive K example + CLS.")


if __name__ == '__main__':
    unittest.main()
