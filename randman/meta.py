from torchmeta.utils.data import ClassDataset, CombinationMetaDataset
import numpy as np
from torch.utils.data import ConcatDataset
from torchmeta.utils.data import BatchMetaDataLoader
from randman.torch_randman import RandmanDataset


def get_seeds(n_tasks):

    train_seeds = np.arange(0, 2 * n_tasks, 2)
    test_seeds = np.arange(2 * n_tasks, 4 * n_tasks, 2)
    val_seeds = np.arange(4 * n_tasks, 6 * n_tasks, 2)

    seeds = {"train": train_seeds, "test": test_seeds, "val": val_seeds}

    return seeds


class RandManClassDataset(ClassDataset):
    def __init__(
        self,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        transform=None,
        target_transform=None,
        data_config=None,
        randman_config=None,
    ):
        if meta_train:
            split_name = "train"
        if meta_val:
            split_name = "val"
        if meta_test:
            split_name = "test"
        self.split_name = split_name

        self.transform = transform
        self.target_transform = target_transform

        super().__init__(
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
        )

        self.data_config = data_config
        self.randman_config = randman_config
        self._num_classes = data_config[f"n_tasks_per_split_{self.split_name}"]

        self.seeds = get_seeds(self._num_classes)
        self._labels = np.arange(self._num_classes)
        self.loaded_datasets = {}

    @property
    def labels(self):
        return self._labels

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        seed = self.seeds[self.split_name][index]

        if seed not in self.loaded_datasets:

            trainset = RandmanDataset(
                nb_samples=self.data_config["dataset_size"], seed=seed, **self.randman_config
            )
            testset = RandmanDataset(
                nb_samples=self.data_config["dataset_size"], seed=seed + 1, **self.randman_config
            )
            self.loaded_datasets[seed] = ConcatDataset([trainset, testset])
            if self.transform is not None:
                self.loaded_datasets[seed].transform = self.transform
            self.loaded_datasets[seed].index = index
            self.loaded_datasets[seed].target_transform_append = lambda x: None

        return self.loaded_datasets[seed]


class RandmanMetaDataset(CombinationMetaDataset):
    def __init__(
        self,
        num_classes_per_task=None,
        meta_train=False,
        meta_val=False,
        meta_test=False,
        meta_split=None,
        transform=None,
        data_config=None,
        randman_config=None,
        dataset_transform=None,
    ):
        if meta_train:
            split_name = "train"
        if meta_val:
            split_name = "val"
        if meta_test:
            split_name = "test"
        self.split_name = split_name

        dataset = RandManClassDataset(
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            meta_split=meta_split,
            transform=transform,
            data_config=data_config,
            randman_config=randman_config,
        )
        super().__init__(
            dataset,
            num_classes_per_task=num_classes_per_task,
            dataset_transform=dataset_transform,
        )


def get_all_datasets(data_config, randman_config, dataset_split, encode_tranform):

    meta_train_dataset = RandmanMetaDataset(
        num_classes_per_task=1,
        meta_train=True,
        transform=encode_tranform,
        data_config=data_config,
        randman_config=randman_config,
        dataset_transform=dataset_split,
    )
    meta_val_dataset = RandmanMetaDataset(
        num_classes_per_task=1,
        meta_val=True,
        transform=encode_tranform,
        data_config=data_config,
        randman_config=randman_config,
        dataset_transform=dataset_split,
    )
    meta_test_dataset = RandmanMetaDataset(
        num_classes_per_task=1,
        meta_test=True,
        transform=encode_tranform,
        data_config=data_config,
        randman_config=randman_config,
        dataset_transform=dataset_split,
    )

    meta_train_dataloader = BatchMetaDataLoader(
        meta_train_dataset,
        data_config["meta_batch_size"],
        shuffle=True,
        num_workers=0,
    )

    meta_val_dataloader = BatchMetaDataLoader(
        meta_val_dataset,
        data_config["meta_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    meta_test_dataloader = BatchMetaDataLoader(
        meta_test_dataset,
        data_config["meta_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    return (
        (
            meta_train_dataset,
            meta_val_dataset,
            meta_test_dataset,
        ),
        (
            meta_train_dataloader,
            meta_val_dataloader,
            meta_test_dataloader,
        ),
    )
