"""
Unit tests for the SpixImageFolder class in datasets.py, focusing on
adaptive and hierarchical superpixel functionalities.
"""
import unittest
import torch
from PIL import Image
import os
import shutil
import argparse # For Namespace
from torchvision import transforms

from datasets import SpixImageFolder # Assuming datasets.py is in the same path or PYTHONPATH
# adaptive_superpixel.py is implicitly tested via SpixImageFolder's adaptive mode.

# Required by Denormalize, which is used in SpixImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

DUMMY_DATASET_DIR = "tmp_test_dataset"

def create_dummy_image(path, size=(32, 32)):
    """Creates a dummy PNG image at the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new('RGB', size, color='red')
    img.save(path, "PNG")

class BaseSpixImageFolderTests:
    """
    Base class to handle dummy dataset creation and cleanup.
    Not a TestCase itself, but meant to be inherited by TestCases
    or used for setUpModule/tearDownModule.
    """
    @classmethod
    def setUpClass(cls):
        """Creates the dummy dataset structure."""
        if os.path.exists(DUMMY_DATASET_DIR):
            shutil.rmtree(DUMMY_DATASET_DIR) # Clean up if exists
        create_dummy_image(os.path.join(DUMMY_DATASET_DIR, "class1", "img1.png"), size=(32,32))
        create_dummy_image(os.path.join(DUMMY_DATASET_DIR, "class1", "img2.png"), size=(32,32))
        create_dummy_image(os.path.join(DUMMY_DATASET_DIR, "class2", "img3.png"), size=(32,32))
        
        # Minimal transform for the images to be processed by SpixImageFolder
        cls.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])


    @classmethod
    def tearDownClass(cls):
        """Removes the dummy dataset directory."""
        shutil.rmtree(DUMMY_DATASET_DIR, ignore_errors=True)


class TestSpixImageFolderFixedMode(BaseSpixImageFolderTests, unittest.TestCase):
    """
    Tests SpixImageFolder in fixed superpixel mode.
    """
    def setUp(self):
        """Initializes args for fixed mode and creates SpixImageFolder instance."""
        self.args = argparse.Namespace(
            adaptive_superpixels=False,
            hierarchical_superpixels=False,
            n_spix_segments=50, # Fixed K
            compactness=10,     # Fixed m
            spix_method='fastslic',
            downsample=1,       # Simplify for small images
            device='cpu',       # For adaptive model if it were used
            # other args that might be accessed by build_dataset or SpixImageFolder defaults
            k_min=20, k_max=60, m_min=5, m_max=15, hierarchical_ks=[10,20] 
        )
        self.dataset = SpixImageFolder(
            root=DUMMY_DATASET_DIR,
            transform=self.transform,
            n_segments=self.args.n_spix_segments,
            compactness=self.args.compactness,
            downsample=self.args.downsample,
            spix_method=self.args.spix_method,
            adaptive_superpixels=self.args.adaptive_superpixels,
            hierarchical_superpixels=self.args.hierarchical_superpixels,
            device=torch.device(self.args.device)
        )

    def test_fixed_getitem_returns_single_assignment(self):
        """
        Tests that __getitem__ returns a single tensor for assignments
        in fixed mode.
        """
        sample, assignment, target = self.dataset[0]
        self.assertIsInstance(sample, torch.Tensor, "Sample should be a torch.Tensor")
        self.assertIsInstance(assignment, torch.Tensor, "Assignment should be a single torch.Tensor in fixed mode")
        self.assertEqual(assignment.ndim, 3, "Assignment tensor should have 3 dimensions (C, H, W)")
        self.assertEqual(assignment.shape[0], 1, "Assignment tensor should have 1 channel")
        self.assertGreater(assignment.max().item(), 0, "Assignment should have superpixel labels greater than 0")


class TestSpixImageFolderAdaptiveMode(BaseSpixImageFolderTests, unittest.TestCase):
    """
    Tests SpixImageFolder in adaptive superpixel mode.
    """
    def setUp(self):
        """Initializes args for adaptive mode and creates SpixImageFolder instance."""
        self.args = argparse.Namespace(
            adaptive_superpixels=True,
            k_min=20,
            k_max=40, # Keep K small for small images
            m_min=5.0,
            m_max=15.0,
            hierarchical_superpixels=False,
            spix_method='fastslic',
            downsample=1, 
            device='cpu',
            # other args
            n_spix_segments=30, compactness=10, hierarchical_ks=[10,20]
        )
        self.dataset = SpixImageFolder(
            root=DUMMY_DATASET_DIR,
            transform=self.transform,
            downsample=self.args.downsample,
            spix_method=self.args.spix_method,
            adaptive_superpixels=self.args.adaptive_superpixels,
            k_min=self.args.k_min,
            k_max=self.args.k_max,
            m_min=self.args.m_min,
            m_max=self.args.m_max,
            hierarchical_superpixels=self.args.hierarchical_superpixels,
            device=torch.device(self.args.device)
            # predictor model will be auto-initialized
        )

    def test_adaptive_getitem_returns_single_assignment(self):
        """
        Tests that __getitem__ returns a single tensor for assignments
        in adaptive mode.
        """
        sample, assignment, target = self.dataset[0]
        self.assertIsInstance(sample, torch.Tensor, "Sample should be a torch.Tensor")
        self.assertIsInstance(assignment, torch.Tensor, "Assignment should be a single torch.Tensor in adaptive mode")
        self.assertEqual(assignment.ndim, 3, "Assignment tensor should have 3 dimensions (C, H, W)")
        self.assertEqual(assignment.shape[0], 1, "Assignment tensor should have 1 channel")
        # Check if K is roughly within k_min, k_max (indirectly)
        num_unique_labels = torch.unique(assignment).numel()
        # This is a loose check as SLIC might not produce exactly K segments.
        # And k_pred is sigmoid-scaled, so it can be very close to k_min/k_max.
        self.assertGreaterEqual(num_unique_labels, self.args.k_min // 2, "Number of unique labels seems too low for adaptive K")
        self.assertLessEqual(num_unique_labels, self.args.k_max * 2, "Number of unique labels seems too high for adaptive K")


class TestSpixImageFolderHierarchicalMode(BaseSpixImageFolderTests, unittest.TestCase):
    """
    Tests SpixImageFolder in hierarchical superpixel mode.
    """
    def setUp(self):
        """Initializes args for hierarchical mode and creates SpixImageFolder instance."""
        self.hierarchical_ks_list = [16, 32] # Small K values for small images
        self.args = argparse.Namespace(
            hierarchical_superpixels=True,
            hierarchical_ks=self.hierarchical_ks_list,
            adaptive_superpixels=False, # Test non-adaptive m first
            compactness=10, # Fixed m for this test
            spix_method='fastslic',
            downsample=1,
            device='cpu',
            # other args
            n_spix_segments=50, k_min=20, k_max=60, m_min=5, m_max=15 
        )
        self.dataset = SpixImageFolder(
            root=DUMMY_DATASET_DIR,
            transform=self.transform,
            downsample=self.args.downsample,
            spix_method=self.args.spix_method,
            hierarchical_superpixels=self.args.hierarchical_superpixels,
            hierarchical_ks=self.args.hierarchical_ks,
            adaptive_superpixels=self.args.adaptive_superpixels,
            compactness=self.args.compactness,
            device=torch.device(self.args.device)
        )

    def test_hierarchical_getitem_returns_list_of_assignments(self):
        """
        Tests that __getitem__ returns a list of tensor assignments
        in hierarchical mode.
        """
        sample, assignments, target = self.dataset[0]
        self.assertIsInstance(sample, torch.Tensor, "Sample should be a torch.Tensor")
        self.assertIsInstance(assignments, list, "Assignments should be a list in hierarchical mode")
        self.assertEqual(len(assignments), len(self.hierarchical_ks_list),
                         f"Number of assignments should be {len(self.hierarchical_ks_list)}")
        for i, assignment_tensor in enumerate(assignments):
            self.assertIsInstance(assignment_tensor, torch.Tensor,
                                  f"Assignment {i} should be a torch.Tensor")
            self.assertEqual(assignment_tensor.ndim, 3, f"Assignment tensor {i} should have 3 dimensions (C, H, W)")
            self.assertEqual(assignment_tensor.shape[0], 1, f"Assignment tensor {i} should have 1 channel")
            # Check if number of unique labels is roughly related to the k for that scale
            num_unique_labels = torch.unique(assignment_tensor).numel()
            self.assertGreaterEqual(num_unique_labels, self.hierarchical_ks_list[i] // 2, 
                                 f"Num labels for scale {i} too low.")
            self.assertLessEqual(num_unique_labels, self.hierarchical_ks_list[i] * 2,
                                 f"Num labels for scale {i} too high.")


class TestSpixImageFolderHierarchicalAdaptiveMMode(BaseSpixImageFolderTests, unittest.TestCase):
    """
    Tests SpixImageFolder with hierarchical K and adaptive m.
    """
    def setUp(self):
        """Initializes args for hierarchical K and adaptive m mode."""
        self.hierarchical_ks_list = [18, 35]
        self.args = argparse.Namespace(
            hierarchical_superpixels=True,
            hierarchical_ks=self.hierarchical_ks_list,
            adaptive_superpixels=True, # For adaptive m
            m_min=5.0,
            m_max=15.0,
            spix_method='fastslic',
            downsample=1,
            device='cpu',
            # other args
            n_spix_segments=50, compactness=10, k_min=20, k_max=60 
        )
        self.dataset = SpixImageFolder(
            root=DUMMY_DATASET_DIR,
            transform=self.transform,
            downsample=self.args.downsample,
            spix_method=self.args.spix_method,
            hierarchical_superpixels=self.args.hierarchical_superpixels,
            hierarchical_ks=self.args.hierarchical_ks,
            adaptive_superpixels=self.args.adaptive_superpixels,
            m_min=self.args.m_min,
            m_max=self.args.m_max,
            # k_min/k_max for adaptive predictor are not strictly needed by SpixImageFolder when only m is adaptive with hierarchical K
            # but AdaptiveSuperpixelParamsPredictor needs them, so SpixImageFolder passes them if adaptive_superpixels=True
            k_min=self.args.k_min, 
            k_max=self.args.k_max,
            device=torch.device(self.args.device)
        )

    def test_hierarchical_adaptive_m_getitem_returns_list(self):
        """
        Tests that __getitem__ returns a list of assignments when
        hierarchical K and adaptive m are enabled.
        """
        sample, assignments, target = self.dataset[0]
        self.assertIsInstance(sample, torch.Tensor, "Sample should be a torch.Tensor")
        self.assertIsInstance(assignments, list, "Assignments should be a list in hierarchical_adaptive_m mode")
        self.assertEqual(len(assignments), len(self.hierarchical_ks_list),
                         f"Number of assignments should be {len(self.hierarchical_ks_list)}")
        for i, assignment_tensor in enumerate(assignments):
            self.assertIsInstance(assignment_tensor, torch.Tensor,
                                  f"Assignment {i} should be a torch.Tensor")
            self.assertEqual(assignment_tensor.ndim, 3, f"Assignment tensor {i} should have 3 dimensions (C, H, W)")

if __name__ == '__main__':
    unittest.main()
