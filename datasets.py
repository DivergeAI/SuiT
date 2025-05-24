# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from typing import Any, Callable, Optional
import os
import json

import numpy as np
import torch

from skimage.transform import rescale
from skimage.segmentation import slic
from fast_slic.avx2 import SlicAvx2

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from adaptive_superpixel import AdaptiveSuperpixelParamsPredictor, predict_adaptive_params

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class SpixImageFolder(datasets.ImageFolder):
    """
    A custom ImageFolder dataset that generates superpixel assignments for images.

    This class extends torchvision's ImageFolder to additionally create superpixel
    segmentation maps for each image using either fixed parameters, adaptively
    predicted parameters (K for number of superpixels, m for compactness),
    or a hierarchy of K values.

    Superpixel generation is performed on a downsampled version of the image.
    The assignments can be a single tensor or a list of tensors (for hierarchical mode).

    Args:
        root (str): Root directory path.
        n_segments (int, optional): Default number of superpixels for fixed mode.
            Defaults to 196.
        compactness (float, optional): Default compactness factor for fixed mode.
            Defaults to 10.
        downsample (int, optional): Factor by which to downsample images before
            superpixel generation. Defaults to 2.
        transform (Optional[Callable], optional): A function/transform that takes in
            a PIL image and returns a transformed version. E.g, `transforms.RandomCrop`.
            Defaults to None.
        target_transform (Optional[Callable], optional): A function/transform that
            takes in the target and transforms it. Defaults to None.
        loader (Callable[[str], Any], optional): A function to load an image given its
            path. Defaults to `datasets.folder.default_loader`.
        is_valid_file (Optional[Callable[[str], bool]], optional): A function that takes
            path to a file and checks if the file is a valid file. Defaults to None.
        spix_method (str, optional): Superpixel algorithm to use ('fastslic' or 'slic').
            Defaults to 'fastslic'.
        adaptive_superpixels (bool, optional): Whether to use adaptively predicted K and m.
            Defaults to False.
        adaptive_predictor_model (Optional[torch.nn.Module], optional): Pre-trained model
            for predicting K and m. If None and `adaptive_superpixels` is True,
            a new `AdaptiveSuperpixelParamsPredictor` is instantiated. Defaults to None.
        k_min (int, optional): Minimum K for adaptive prediction. Defaults to 100.
        k_max (int, optional): Maximum K for adaptive prediction. Defaults to 300.
        m_min (float, optional): Minimum m for adaptive prediction. Defaults to 1.0.
        m_max (float, optional): Maximum m for adaptive prediction. Defaults to 20.0.
        device (Optional[torch.device], optional): Device for the adaptive predictor model.
            Defaults to None (CPU).
        hierarchical_superpixels (bool, optional): Whether to generate superpixels for
            multiple K values. Defaults to False.
        hierarchical_ks (Optional[list[int]], optional): List of K values for hierarchical
            mode. Defaults to [64, 128, 256] if `hierarchical_superpixels` is True and
            this is None.
    """
    def __init__(
        self,
        root: str,
        n_segments=196, # Default for non-adaptive mode
        compactness=10, # Default for non-adaptive mode
        downsample=2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        spix_method = 'fastslic',
        adaptive_superpixels: bool = False,
        adaptive_predictor_model: Optional[torch.nn.Module] = None,
        k_min: int = 100,
        k_max: int = 300,
        m_min: float = 1.0, # SLIC compactness can be float
        m_max: float = 20.0,
        device: Optional[torch.device] = None,
        hierarchical_superpixels: bool = False,
        hierarchical_ks: Optional[list[int]] = None,
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.denormalize = Denormalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.downsample = downsample
        self.spix_method = spix_method

        self.adaptive_superpixels = adaptive_superpixels
        self.adaptive_predictor_model = adaptive_predictor_model
        self.k_min = k_min
        self.k_max = k_max
        self.m_min = m_min
        self.m_max = m_max
        self.device = device

        # Initialize predictor model if adaptive mode is on and no model is provided
        if self.adaptive_superpixels and self.adaptive_predictor_model is None:
            self.adaptive_predictor_model = AdaptiveSuperpixelParamsPredictor()
        
        if self.adaptive_predictor_model is not None and self.device is not None:
            self.adaptive_predictor_model.to(self.device)
            self.adaptive_predictor_model.eval() # Ensure model is in eval mode

        # Store n_segments and compactness for non-adaptive mode
        self.n_segments = n_segments
        self.compactness = compactness

        self.hierarchical_superpixels = hierarchical_superpixels
        self.hierarchical_ks = hierarchical_ks

        if self.hierarchical_superpixels:
            if self.hierarchical_ks is None or not self.hierarchical_ks:
                # Default K values for hierarchical superpixels if not provided or empty
                self.hierarchical_ks = [64, 128, 256] 
                # Consider logging a warning here if a logger is available
            elif not all(isinstance(k_val, int) and k_val > 0 for k_val in self.hierarchical_ks):
                raise ValueError("All values in hierarchical_ks must be positive integers.")
            # Ensure K values are sorted, can be useful for some applications, though not strictly necessary for generation
            self.hierarchical_ks.sort()


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, assignment, target) where assignment is a map of
                   superpixel indices for each pixel (can be a single tensor or a
                   list of tensors in hierarchical mode), and target is the
                   class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            # augmented sample
            sample = self.transform(sample)
            
            # temporarily convert to [0, 255] and resize to acquire superpixels for SLIC.
            # SLIC algorithms typically expect uint8 images.
            # This conversion is done once, regardless of hierarchical or single superpixel generation.
            sample_for_spix_np = np.array(self.denormalize(sample.clone()) * 255).transpose(1, 2, 0) # Use .clone() to avoid in-place modification issues
            sample_for_spix_np = rescale(sample_for_spix_np, 1 / self.downsample, anti_aliasing=True, channel_axis=2).round().clip(0, 255).astype(np.uint8)

            assignment_output: Any # Can be a list of tensors or a single tensor

            if self.hierarchical_superpixels:
                assignments_list = []
                m_to_use = self.compactness # Default compactness

                # If adaptive superpixels are also enabled, predict m, but K values come from hierarchical_ks
                if self.adaptive_superpixels and self.adaptive_predictor_model:
                    img_tensor_for_predictor = sample.unsqueeze(0)
                    if self.device:
                        img_tensor_for_predictor = img_tensor_for_predictor.to(self.device)
                    
                    # We only need m from the predictor in this specific hierarchical case
                    _, m_pred = predict_adaptive_params(
                        self.adaptive_predictor_model,
                        img_tensor_for_predictor,
                        self.k_min, # k_min/k_max are for adaptive K, not used directly here but predict_adaptive_params needs them
                        self.k_max,
                        self.m_min,
                        self.m_max
                    )
                    m_to_use = float(m_pred.item())
                
                for k_value in self.hierarchical_ks:
                    # Note: The SLIC operation (FastSLIC/skimage.slic) is non-differentiable.
                    if self.spix_method == 'fastslic':
                        slic_algo = SlicAvx2(num_components=k_value, compactness=m_to_use)
                        current_assignment_np = slic_algo.iterate(sample_for_spix_np)
                    elif self.spix_method == 'slic':
                        current_assignment_np = slic(sample_for_spix_np, n_segments=k_value, compactness=m_to_use, channel_axis=2)
                    else:
                        raise NotImplementedError(f"Superpixel method {self.spix_method} not implemented for hierarchical generation.")
                    
                    assignments_list.append(torch.tensor(current_assignment_np).unsqueeze(0))
                assignment_output = assignments_list
            
            else: # Single superpixel generation (adaptive or fixed)
                if self.adaptive_superpixels:
                    if self.adaptive_predictor_model is None:
                        raise RuntimeError("Adaptive superpixels enabled but predictor model is not available.")
                    
                    img_tensor_for_predictor = sample.unsqueeze(0)
                    if self.device:
                        img_tensor_for_predictor = img_tensor_for_predictor.to(self.device)

                    k_pred, m_pred = predict_adaptive_params(
                        self.adaptive_predictor_model,
                        img_tensor_for_predictor,
                        self.k_min, self.k_max, self.m_min, self.m_max
                    )
                    current_k = int(k_pred.item())
                    current_m = float(m_pred.item())
                else:
                    current_k = self.n_segments
                    current_m = self.compactness

                # Note: The SLIC operation (FastSLIC/skimage.slic) is non-differentiable.
                if self.spix_method == 'fastslic':
                    slic_algo = SlicAvx2(num_components=current_k, compactness=current_m)
                    single_assignment_np = slic_algo.iterate(sample_for_spix_np)
                elif self.spix_method == 'slic':
                    single_assignment_np = slic(sample_for_spix_np, n_segments=current_k, compactness=current_m, channel_axis=2)
                else:
                    raise NotImplementedError(f"Superpixel method {self.spix_method} not implemented.")
                
                assignment_output = torch.tensor(single_assignment_np).unsqueeze(0)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, assignment_output, target


class Denormalize(torch.nn.Module):
    """
    Denormalizes a tensor given mean and standard deviation.
    Inverse of `torchvision.transforms.Normalize`.
    """
    def __init__(self, mean, std, inplace=False) -> None:
        """
        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace (bool, optional): Whether to perform the operation in-place.
                                      Defaults to False.
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (torch.Tensor): Tensor of shape (..., C, H, W) to denormalize.
                                   Assumes the tensor is normalized.
        Returns:
            torch.Tensor: Denormalized tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor


class INatDataset(ImageFolder):
    """
    Custom dataset for iNaturalist, handling its specific JSON structure.
    Inherits from torchvision.datasets.ImageFolder but overrides __init__
    to load samples from the iNaturalist JSON annotations.
    """
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train: bool, args: Any) -> tuple[torch.utils.data.Dataset, int]:
    """
    Builds a dataset based on the provided arguments.

    Args:
        is_train (bool): Whether to build the training set or validation set.
        args (Any): Command-line arguments or a namespace object containing
                    dataset configuration (e.g., data_set, data_path, model,
                    various superpixel parameters).

    Returns:
        tuple[torch.utils.data.Dataset, int]: A tuple containing the dataset
                                              instance and the number of classes.
    """
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if 'suit' in args.model: # Assuming 'suit' in model name implies using SpixImageFolder
            # Pass adaptive parameters if available in args, otherwise use defaults
            dataset = SpixImageFolder(
                root,
                transform=transform,
                n_segments=args.n_spix_segments, # Kept for backward compatibility / non-adaptive case
                compactness=args.compactness,     # Kept for backward compatibility / non-adaptive case
                downsample=args.downsample,
                spix_method=args.spix_method,
                adaptive_superpixels=getattr(args, 'adaptive_superpixels', False),
                # adaptive_predictor_model can be passed if pre-loaded, else None and initialized by SpixImageFolder
                k_min=getattr(args, 'k_min', 100),
                k_max=getattr(args, 'k_max', 300),
                m_min=getattr(args, 'm_min', 1.0),
                m_max=getattr(args, 'm_max', 20.0),
                device=torch.device(getattr(args, 'device', 'cpu')), # Determine device from args
                hierarchical_superpixels=getattr(args, 'hierarchical_superpixels', False),
                hierarchical_ks=getattr(args, 'hierarchical_ks', [64, 128, 256]) # Default list for hierarchical Ks
            )
        else:
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train: bool, args: Any) -> Callable:
    """
    Builds a torchvision transform pipeline for training or evaluation.

    Args:
        is_train (bool): Whether to create a training transform (with augmentations)
                         or an evaluation transform.
        args (Any): Command-line arguments or a namespace object containing
                    transform configurations (e.g., input_size, color_jitter, aa).

    Returns:
        Callable: A torchvision transform pipeline.
    """
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        #ার্get_size = args.input_size
        # Maintain aspect ratio for resizing
        size = int(round(args.input_size / args.eval_crop_ratio))
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), # Use BICUBIC for consistency
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
