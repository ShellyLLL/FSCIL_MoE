import json
import os
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DatasetManager:

    def __init__(self, cfg):

        # Properties
        self.cfg = cfg

        # Dataset split setting
        self.root = cfg.DATASET.ROOT
        self.dataset_name    = cfg.DATASET.NAME
        self.num_init_cls    = cfg.DATASET.NUM_INIT_CLS
        self.num_inc_cls     = cfg.DATASET.NUM_INC_CLS
        self.num_base_shot   = cfg.DATASET.NUM_BASE_SHOT
        self.num_inc_shot    = cfg.DATASET.NUM_INC_SHOT
        # A manifest fixes the actual source examples used by every FSCIL
        # session.  It deliberately uses a private RNG so unrelated calls to
        # numpy/torch RNGs cannot change a support set.
        self.support_manifest_path = getattr(cfg.DATASET, 'SUPPORT_MANIFEST_PATH', '')
        self._support_manifest_path_auto = False
        manifest_seed = getattr(cfg.DATASET, 'SUPPORT_MANIFEST_SEED', -1)
        self.support_manifest_seed = cfg.SEED if manifest_seed < 0 else manifest_seed
        self.support_manifest_auto_save = getattr(cfg.DATASET, 'SUPPORT_MANIFEST_AUTO_SAVE', True)
        checkpoint_cfg = getattr(getattr(getattr(cfg, 'TRAINER', None), 'BiMC', None), 'CHECKPOINT', None)
        self._defer_support_manifest_until_resume = bool(getattr(checkpoint_cfg, 'RESUME', ''))
        
        # training setting of data
        self.num_workers     = cfg.DATALOADER.NUM_WORKERS
        self.train_batchsize_base = cfg.DATALOADER.TRAIN.BATCH_SIZE_BASE
        self.train_batchsize_inc = cfg.DATALOADER.TRAIN.BATCH_SIZE_INC
        self.test_batchsize = cfg.DATALOADER.TEST.BATCH_SIZE

        # setup data
        self._setup_data(self.root, self.dataset_name)
        self.class_index_in_task = []
        self.class_index_in_task.append(np.arange(0, self.num_init_cls))
        for start in range(self.num_init_cls, self.num_total_classes, self.num_inc_cls):
            end = min(start + self.num_inc_cls, self.num_total_classes)
            self.class_index_in_task.append(np.arange(start, end))
        self.num_tasks = len(self.class_index_in_task)
        (self.base_train_transform, self.incremental_train_transform,
         self.test_transform) = self._set_transform()
        # Kept as a compatibility alias for external callers that used the
        # previous public attribute.
        self.train_transform = self.base_train_transform
        # A normal run must leave an inspectable support protocol behind even
        # before its first session checkpoint is written.  An explicit path
        # still takes priority; with AUTO_SAVE=False an empty path preserves
        # the previous in-memory-only behaviour for lightweight experiments.
        if not self.support_manifest_path and self.support_manifest_auto_save:
            checkpoint_dir = getattr(checkpoint_cfg, 'DIR', 'checkpoints/fscil_moe')
            safe_name = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in str(self.dataset_name).lower()
            )
            self.support_manifest_path = str(
                Path(checkpoint_dir) /
                f"{safe_name}_support_seed{int(self.support_manifest_seed)}.json"
            )
            self._support_manifest_path_auto = True
        # A resume checkpoint is the source of truth.  Do not even create a
        # transient support sample before Runner installs its checkpointed
        # manifest; otherwise an auto sidecar could be polluted by a sample
        # that was never part of the resumed experiment.
        self.support_manifest = (
            None if self._defer_support_manifest_until_resume
            else self._load_or_create_support_manifest()
        )



    def _setup_data(self, root, dataset_name):
        full_dataset = get_data_source(root, dataset_name)
        self.class_names = full_dataset.classes
        self.template = full_dataset.template
        self.train_data, self.train_targets = full_dataset.get_train_data()
        self.test_data, self.test_targets = full_dataset.get_test_data()

        # convert labels  to `np.ndarray` for convenient indexing
        if not isinstance(self.train_targets, np.ndarray):
            self.train_targets = np.array(self.train_targets)
        if not isinstance(self.test_targets, np.ndarray):
            self.test_targets = np.array(self.test_targets)
        
        self.num_total_classes = len(self.class_names)
    

    def get_dataset(self, task_id, source, mode=None, accumulated_past=False):
        '''
        source: which part of dataset
        mode: which data transform is used
        accumulated_past (Bool): Whether the training data in this contains the data from the past 
        '''
        assert 0 <= task_id < len(self.class_index_in_task), \
               f"task id {task_id} should be in range [0, {len(self.class_index_in_task) - 1}]"

        # Get data
        if source == 'train':
            # When training, using data of task [i]
            x, y = self.train_data, self.train_targets
            if accumulated_past:
                class_idx = np.concatenate(self.class_index_in_task[0: task_id + 1])
            else:
                class_idx = self.class_index_in_task[task_id]

        elif source == 'test':
            # When testing, using data of tasks [0..i]
            x, y = self.test_data, self.test_targets
            class_idx = np.concatenate(self.class_index_in_task[0: task_id + 1])

        else:
            raise ValueError(f'Invalid data source :{source}')
        
        # Get Transform
        if mode == 'train':
            transform = (self.base_train_transform if task_id == 0
                         else self.incremental_train_transform)
        elif mode == 'test':
            transform = self.test_transform
        else:
            raise ValueError(f'Invalid transform mode: {mode}')

        def find_sublist_indices(matrix, numbers):
            """
            Function to find the indices of the sublists where each number in 'numbers' is located.

            Parameters:
            matrix (list of list of int): The 2D list to search in.
            numbers (np.ndarray): The numpy array of numbers to search for.

            Returns:
            dict: A dictionary with keys as the numbers from 'numbers' and values as the indices of the sublists.
            """
            indices = {}
            for x in numbers:
                found = False
                for i, sublist in enumerate(matrix):
                    if x in sublist:
                        indices[x] = i
                        found = True
                        break
                if not found:
                    indices[x] = -1  # If number not found, set index to -1
            return indices
        
        class_to_task_id = find_sublist_indices(self.class_index_in_task, class_idx)
        num_shot = self.num_base_shot if task_id == 0 else self.num_inc_shot
        data, targets, source_ids = self._select_data_from_class_index(
            x, y, class_idx, num_shot, source, task_id, accumulated_past
        )
        task_dataset = TaskDataset(
            data, targets, transform, class_to_task_id, self.class_names, source_ids
        )
        return task_dataset
    

    
    def get_dataloader(self, task_id, source, mode=None, accumulate_past=False):
        assert source in ['train', 'test'], f'data source must be in ["train", "test"], got {source}'
        # the default mode is same as source
        if mode == None:
            mode = source
        dataset = self.get_dataset(task_id, source, mode, accumulate_past)
        if source == 'train':
            if task_id == 0:
                batchsize = self.train_batchsize_base
            else:
                batchsize = self.train_batchsize_inc
            loader = DataLoader(dataset,
                                batch_size=batchsize,
                                # Keep feature-extraction loaders deterministic: they call
                                # source="train", mode="test".  Only optimization loaders
                                # should shuffle samples between epochs.
                                shuffle=(mode == 'train'),
                                num_workers=self.num_workers,
                                drop_last=False,
                                pin_memory=True)
        elif source == 'test':
            loader = DataLoader(dataset,
                                batch_size=self.test_batchsize,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=False,
                                pin_memory=True)
        else:
            raise ValueError(f'Invalid data source: {source}')
        return loader

    def get_support_view_dataloader(self, task_id, view_id, mode='train'):
        """Return a reproducible augmented view of the fixed support sources."""
        if mode not in {'train', 'test'}:
            raise ValueError("support view mode must be 'train' or 'test'")
        dataset = self.get_dataset(task_id, source='train', mode=mode, accumulated_past=False)
        dataset.view_id = int(view_id)
        dataset.deterministic_view_seed = int(getattr(self.cfg, 'SEED', 0))
        dataset.owner_task_id = int(task_id)
        batchsize = self.train_batchsize_base if task_id == 0 else self.train_batchsize_inc
        return DataLoader(
            dataset, batch_size=batchsize, shuffle=False,
            num_workers=self.num_workers, drop_last=False, pin_memory=True,
        )
    


    def _select_data_from_class_index(self, x, y, class_idx, shot, source,
                                      task_id=None, accumulated_past=False):
        ret_x = []
        ret_y = []
        ret_source_ids = []
        if isinstance(x, list):
            x = np.array(x)
        for c in class_idx:
            idx_c = np.where(y == c)[0]
            
            if shot is not None and source == 'train':
                # All train consumers of a session (optimizer, extraction,
                # planner, etc.) index the same precomputed support examples.
                # ``task_id`` is optional only for the legacy direct helper.
                if task_id is None:
                    idx_selected = idx_c if shot == -1 else idx_c[:shot]
                else:
                    if self.support_manifest is None:
                        raise RuntimeError(
                            'Support manifest is deferred for checkpoint resume; '
                            'Runner must load the checkpoint before creating dataloaders.'
                        )
                    owner_task = self._task_id_for_class(int(c))
                    idx_selected = np.asarray(
                        self.support_manifest['sessions'][str(owner_task)][str(int(c))],
                        dtype=np.int64,
                    )
            else:
                idx_selected = idx_c

            ret_x.append(x[idx_selected])
            ret_y.append(y[idx_selected])
            ret_source_ids.append(idx_selected)
        ret_x = np.concatenate(ret_x)
        ret_y = np.concatenate(ret_y)
        ret_source_ids = np.concatenate(ret_source_ids).astype(np.int64, copy=False)

        return ret_x, ret_y, ret_source_ids

    def _task_id_for_class(self, class_id):
        for task_id, classes in enumerate(self.class_index_in_task):
            if class_id in classes:
                return task_id
        raise ValueError(f'class {class_id} does not belong to any FSCIL task')

    def _load_or_create_support_manifest(self):
        """Load a checked manifest or create every session exactly once."""
        path = Path(self.support_manifest_path) if self.support_manifest_path else None
        if path is not None and path.is_file():
            with path.open('r', encoding='utf-8') as file:
                manifest = json.load(file)
            self._validate_support_manifest(manifest)
            return manifest

        manifest = self._build_support_manifest()
        if path is not None and self.support_manifest_auto_save:
            self.save_support_manifest(path, manifest)
        return manifest

    def _build_support_manifest(self):
        rng = np.random.default_rng(self.support_manifest_seed)
        sessions = {}
        for task_id, class_ids in enumerate(self.class_index_in_task):
            shot = self.num_base_shot if task_id == 0 else self.num_inc_shot
            supports = {}
            for class_id in class_ids:
                candidates = np.flatnonzero(self.train_targets == class_id)
                if shot is None or shot == -1 or shot >= len(candidates):
                    chosen = candidates
                else:
                    chosen = rng.choice(candidates, size=shot, replace=False)
                supports[str(int(class_id))] = [int(index) for index in chosen]
            sessions[str(task_id)] = supports
        return {
            'version': 1,
            'dataset_name': str(self.dataset_name),
            'seed': int(self.support_manifest_seed),
            'sessions': sessions,
        }

    def _validate_support_manifest(self, manifest):
        if manifest.get('version') != 1:
            raise ValueError('Unsupported support manifest version')
        if str(manifest.get('dataset_name', '')).lower() != str(self.dataset_name).lower():
            raise ValueError('Support manifest dataset_name does not match the configured dataset')
        sessions = manifest.get('sessions')
        if not isinstance(sessions, dict):
            raise ValueError('Support manifest must contain a sessions mapping')
        for task_id, class_ids in enumerate(self.class_index_in_task):
            supports = sessions.get(str(task_id))
            if not isinstance(supports, dict):
                raise ValueError(f'Support manifest is missing session {task_id}')
            expected_shot = self.num_base_shot if task_id == 0 else self.num_inc_shot
            for class_id in class_ids:
                ids = supports.get(str(int(class_id)))
                if not isinstance(ids, list) or not ids:
                    raise ValueError(f'Support manifest is missing class {int(class_id)} in session {task_id}')
                source_ids = np.asarray(ids, dtype=np.int64)
                if len(np.unique(source_ids)) != len(source_ids):
                    raise ValueError(f'Support manifest has duplicate source ids for class {int(class_id)}')
                if np.any(source_ids < 0) or np.any(source_ids >= len(self.train_targets)):
                    raise ValueError(f'Support manifest has out-of-range source ids for class {int(class_id)}')
                if not np.all(self.train_targets[source_ids] == class_id):
                    raise ValueError(f'Support manifest source ids do not match class {int(class_id)}')
                if expected_shot is not None and expected_shot != -1:
                    required = min(expected_shot, int(np.sum(self.train_targets == class_id)))
                    if len(source_ids) != required:
                        raise ValueError(f'Support manifest has wrong shot count for class {int(class_id)}')

    def get_support_manifest(self):
        """Return a JSON-safe copy suitable for session checkpoints."""
        if self.support_manifest is None:
            raise RuntimeError('Support manifest has not been installed yet.')
        return json.loads(json.dumps(self.support_manifest))

    def load_support_manifest(self, manifest):
        """Install checkpointed support selections after validating source ids."""
        self._validate_support_manifest(manifest)
        self.support_manifest = json.loads(json.dumps(manifest))
        self._defer_support_manifest_until_resume = False
        # Only an automatically-derived sidecar is overwritten here.  An
        # explicit user path may deliberately point to a shared comparison
        # manifest and should not be rewritten behind their back.
        if (getattr(self, '_support_manifest_path_auto', False)
                and self.support_manifest_auto_save and self.support_manifest_path):
            self.save_support_manifest()

    def save_support_manifest(self, path=None, manifest=None):
        """Persist the selections so a fresh/resumed process never resamples."""
        raw_path = path or self.support_manifest_path
        if not raw_path:
            raise ValueError('A support manifest path is required to save the manifest')
        target = Path(raw_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.support_manifest if manifest is None else manifest
        temp_target = target.with_suffix(target.suffix + '.tmp')
        with temp_target.open('w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        os.replace(temp_target, target)
    

    def _set_transform(self):
        img_size = 224
        MEAN = [0.48145466, 0.4578275, 0.40821073]
        STD  = [0.26862954, 0.26130258, 0.27577711]
        base_train_transform  = transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomResizedCrop((img_size, img_size), scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        # CUB is fine-grained and each incremental class has only five shots;
        # avoid the strong 8%-area crop used for the base session.
        inc_scale = (0.5, 1.0) if self.dataset_name.lower() == 'cub200' else (0.08, 1.0)
        incremental_train_transform = transforms.Compose([
            transforms.RandomResizedCrop((img_size, img_size), scale=inc_scale,
                                         ratio=(0.75, 1.333),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return base_train_transform, incremental_train_transform, test_transform
    


class TaskDataset(Dataset):
    def __init__(self, images, labels, transform, class_to_task_id=None, class_name=None,
                 source_ids=None, view_id=None, deterministic_view_seed=None,
                 owner_task_id=None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.transform = transform
        self.use_path = isinstance(images[0], str)
        self.class_to_task_id = class_to_task_id
        self.class_name = class_name
        self.source_ids = (np.arange(len(images), dtype=np.int64) if source_ids is None
                           else np.asarray(source_ids, dtype=np.int64))
        assert len(self.source_ids) == len(images), 'Source id size error!'
        self.view_id = view_id
        self.deterministic_view_seed = deterministic_view_seed
        self.owner_task_id = owner_task_id


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        def load_and_transform():
            raw = pil_loader(self.images[idx]) if self.use_path else Image.fromarray(self.images[idx])
            return self.transform(raw)

        if self.view_id is None or self.deterministic_view_seed is None:
            image = load_and_transform()
        else:
            # Torchvision random transforms draw from torch/Python RNGs. Save
            # and restore all local states so validation views are reproducible
            # without perturbing optimizer-loader augmentation.
            payload = (
                f"{int(self.deterministic_view_seed)}|{int(self.owner_task_id or 0)}|"
                f"{int(self.source_ids[idx])}|{int(self.view_id)}"
            ).encode('utf-8')
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2 ** 31)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            try:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                image = load_and_transform()
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.random.set_rng_state(torch_state)
        label = self.labels[idx]
        
        if self.class_to_task_id is not None:
            task_id = self.class_to_task_id[label]
        else:
            task_id = -1
        
        if self.class_name is not None:
            cls_name = self.class_name[label]
        else:
            cls_name = ''
            
        ret = {
            # ``idx`` remains for existing engine code; unlike the old local
            # row number it is now a stable index into the original source.
            'idx': self.source_ids[idx],
            'source_id': self.source_ids[idx],
            'image': image,
            'label': label,
            'cls_name': cls_name,
            'task_id' : task_id
        }
        return ret



def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# NEED MODIFY HERE IF YOU WANT TO ADD NEW DATASETS
def get_data_source(root, name):
    from .cifar100 import CIFAR100
    from .miniimagenet import MiniImagenet
    from .cub200 import CUB200
    source_dict = {
        'cifar100' : CIFAR100,
        'miniimagenet' : MiniImagenet,
        'cub200': CUB200,
    }
    return source_dict[name.lower()](root=root)
