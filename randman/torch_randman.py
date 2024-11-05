#!/usr/bin/env python3

import numpy as np
import pickle
import gzip
import itertools

import torch
import torch.utils
import torch.utils.data


class TorchRandman:
    """Randman (torch version) objects hold the parameters for a smooth random manifold from which datapoints can be sampled."""

    def __init__(
        self,
        embedding_dim,
        manifold_dim,
        alpha=2,
        beta=0,
        prec=1e-3,
        max_f_cutoff=1000,
        use_bias=False,
        seed=None,
        dtype=torch.float32,
        device=None,
    ):
        """Initializes a randman object.

        Args
        ----
        embedding_dim : The embedding space dimension
        manifold_dim : The manifold dimension
        alpha : The power spectrum fall-off exponenent. Determines the smoothenss of the manifold (default 2)
        use_bias: If True, manifolds are placed at random offset coordinates within a [0,1] simplex.
        seed: This seed is used to init the *global* torch.random random number generator.
        prec: The precision paramter to determine the maximum frequency cutoff (default 1e-3)
        """
        self.alpha = alpha
        self.beta = beta
        self.use_bias = use_bias
        self.dim_embedding = embedding_dim
        self.dim_manifold = manifold_dim
        self.f_cutoff = int(
            np.min((np.ceil(np.power(prec, -1 / self.alpha)), max_f_cutoff))
        )
        self.params_per_1d_fun = 3
        self.dtype = dtype

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        if seed is not None:
            torch.random.manual_seed(seed)

        self.init_random()
        self.init_spect(self.alpha, self.beta)

    def init_random(self):
        self.params = torch.rand(
            self.dim_embedding,
            self.dim_manifold,
            self.params_per_1d_fun,
            self.f_cutoff,
            dtype=self.dtype,
            device=self.device,
        )
        if not self.use_bias:
            self.params[:, :, 0, 0] = 0

    def init_spect(
        self,
        alpha=2.0,
        res=0,
    ):
        """Sets up power spectrum modulation

        Args
        ----
        alpha : Power law decay exponent of power spectrum
        res : Peak value of power spectrum.
        """
        r = torch.arange(self.f_cutoff, dtype=self.dtype, device=self.device) + 1
        s = 1.0 / (torch.abs(r - res) ** alpha + 1.0)
        self.spect = s

    def eval_random_function_1d(self, x, theta):
        tmp = torch.zeros(len(x), dtype=self.dtype, device=self.device)
        s = self.spect
        for i in range(self.f_cutoff):
            tmp += (
                theta[0, i]
                * s[i]
                * torch.sin(2 * np.pi * (i * x * theta[1, i] + theta[2, i]))
            )
        return tmp

    def eval_random_function(self, x, params):
        tmp = torch.ones(len(x), dtype=self.dtype, device=self.device)
        for d in range(self.dim_manifold):
            tmp *= self.eval_random_function_1d(x[:, d], params[d])
        return tmp

    def eval_manifold(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        tmp = torch.zeros(
            (x.shape[0], self.dim_embedding), dtype=self.dtype, device=self.device
        )
        for i in range(self.dim_embedding):
            tmp[:, i] = self.eval_random_function(x, self.params[i])
        return tmp

    def get_random_manifold_samples(self, nb_samples):
        x = torch.rand(
            nb_samples, self.dim_manifold, dtype=self.dtype, device=self.device
        )
        y = self.eval_manifold(x)
        return x, y


def standardize(x, eps=1e-7):
    mi, _ = x.min(0)
    ma, _ = x.max(0)
    return (x - mi) / (ma - mi + eps)


def make_spiking_dataset(
    nb_classes=10,
    nb_units=100,
    nb_steps=100,
    step_frac=1.0,
    dim_manifold=2,
    nb_spikes=1,
    nb_samples_per_class=1000,
    alpha=2.0,
    shuffle=True,
    classification=True,
    seed=None,
):
    """Generates event-based generalized spiking randman classification/regression dataset.
    In this dataset each unit fires a fixed number of spikes. So ratebased or spike count based decoding won't work.
    All the information is stored in the relative timing between spikes.
    For regression datasets the intrinsic manifold coordinates are returned for each target.
    Args:
        nb_classes: The number of classes to generate
        nb_units: The number of units to assume
        nb_steps: The number of time steps to assume
        step_frac: Fraction of time steps from beginning of each to contain spikes (default 1.0)
        nb_spikes: The number of spikes per unit
        nb_samples: Number of samples from each manifold per class
        alpha: Randman smoothness parameter
        shuffe: Whether to shuffle the dataset
        classification: Whether to generate a classification (default) or regression dataset
        seed: The random seed (default: None)
    Returns:
        A tuple of data,labels. The data is structured as numpy array
        (sample x event x 2 ) where the last dimension contains
        the relative [0,1] (time,unit) coordinates and labels.
    """

    data = []
    labels = []
    targets = []

    if seed is not None:
        np.random.seed(seed)

    max_value = np.iinfo(int).max
    randman_seeds = np.random.randint(max_value, size=(nb_classes, nb_spikes))

    for k in range(nb_classes):
        x = np.random.rand(nb_samples_per_class, dim_manifold)
        submans = [
            TorchRandman(nb_units, dim_manifold, alpha=alpha, seed=randman_seeds[k, i])
            for i in range(nb_spikes)
        ]
        units = []
        times = []
        for i, rm in enumerate(submans):
            y = rm.eval_manifold(x)
            y = standardize(y)
            units.append(
                np.repeat(
                    np.arange(nb_units).reshape(1, -1), nb_samples_per_class, axis=0
                )
            )
            times.append(y.numpy())

        units = np.concatenate(units, axis=1)
        times = np.concatenate(times, axis=1)
        events = np.stack([times, units], axis=2)
        data.append(events)
        labels.append(k * np.ones(len(units)))
        targets.append(x)

    data = np.concatenate(data, axis=0)
    labels = np.array(np.concatenate(labels, axis=0), dtype=int)
    targets = np.concatenate(targets, axis=0)

    if shuffle:
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]
        targets = targets[idx]

    data[:, :, 0] *= nb_steps * step_frac
    # data = np.array(data, dtype=int)

    if classification:
        return data, labels
    else:
        return data, targets


class RandmanDataset(torch.utils.data.Dataset):
    def __init__(self, **randman_config) -> None:
        self.config = randman_config
        self.data, self.targets = make_spiking_dataset(**randman_config)

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        spike_times = data[..., 0].astype(int)
        spikes = torch.nn.functional.one_hot(
            torch.tensor(spike_times), self.config["nb_steps"]
        ).T
        return spikes, target

    def __len__(self):
        return len(self.data)
