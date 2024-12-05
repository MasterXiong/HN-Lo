from collections import deque
from typing import Optional
import os

import jax
import numpy as np
from octo.model.octo_model import OctoModel
import tensorflow as tf
from transforms3d.euler import euler2axangle

from octo.domains.utils.action_ensemble import BatchActionEnsembler


class OctoInference:
    def __init__(
        self,
        model: Optional[OctoModel] = None,
        dataset_id: Optional[str] = None,
        model_type: str = "octo-base",
        policy_setup: str = "libero",
        horizon: int = 2,
        pred_action_horizon: int = 4,
        exec_horizon: int = 1,
        image_size: int = 256,
        action_scale: float = 1.0,
        init_rng: int = 0,
        action_ensemble: bool = False,
        crop: bool = False,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "libero":
            dataset_id = "libero" if dataset_id is None else dataset_id
            action_ensemble_temp = 0.0
            # self.sticky_gripper_num_repeat = 1
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported for octo models.")
        self.policy_setup = policy_setup
        self.dataset_id = dataset_id

        if model is not None:
            self.tokenizer, self.tokenizer_kwargs = None, None
            self.model = model
            if dataset_id in self.model.dataset_statistics:
                self.action_mean = self.model.dataset_statistics[dataset_id]["action"]["mean"]
                self.action_std = self.model.dataset_statistics[dataset_id]["action"]["std"]
                self.action_normalization_mask = self.model.dataset_statistics[dataset_id]["action"]["mask"].astype(float)
            else:
                if 'action' in self.model.dataset_statistics:
                    self.action_mean = self.model.dataset_statistics["action"]["mean"]
                    self.action_std = self.model.dataset_statistics["action"]["std"]
                    self.action_normalization_mask = self.model.dataset_statistics["action"]["mask"].astype(float)
                else:
                    self.action_mean = np.zeros(7)
                    self.action_std = np.ones(7)
                    self.action_normalization_mask = np.ones(7)
        elif model_type in ["octo-base", "octo-small"]:
            # released huggingface octo models
            self.model_type = f"hf://rail-berkeley/{model_type}"
            self.tokenizer, self.tokenizer_kwargs = None, None
            self.model = OctoModel.load_pretrained(self.model_type)
            self.action_mean = self.model.dataset_statistics[dataset_id]["action"]["mean"]
            self.action_std = self.model.dataset_statistics[dataset_id]["action"]["std"]
        else:
            raise NotImplementedError()

        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp
        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            # the purpose of this for loop is just to match octo server's inference seeds
            self.rng, _key = jax.random.split(self.rng)  # each shape [2,]

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = BatchActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.crop = crop

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.crop:
            scale = 0.9
            offset = int((1 - scale) / 2 * self.image_size + 0.5)
            target_size = int(scale * self.image_size + 0.5)
            image = tf.image.crop_to_bounding_box(image, offset, offset, target_size, target_size)
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        # Alternative implementation below; but looks like for real eval, filling the entire buffer at the first step is not necessary
        # if self.num_image_history == 0:
        #     self.image_history.extend([image] * self.horizon)
        # else:
        #     self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64)  # note: this should be of float type, not a bool type
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        # pad_mask = np.ones(self.horizon, dtype=np.float64) # note: this should be of float type, not a bool type
        # pad_mask[:self.horizon - self.num_image_history] = 0
        return images, pad_mask

    def reset(self, task_description: str, remove_useless_token=False, env_num=20) -> None:
        self.task = self.model.create_tasks(texts=[task_description])
        if remove_useless_token:
            instruction_length = self.task['language_instruction']['attention_mask'].sum(1)
            self.task['language_instruction']['input_ids'][:, instruction_length - 1] = 0
            self.task['language_instruction']['attention_mask'][:, instruction_length - 1] = 0
        self.task = jax.tree_map(lambda x: np.repeat(x, env_num, axis=0), self.task)

        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()
        # shape: horizon_len * batch_size -> batch_size * horizon_len
        images = images.transpose(1, 0, 2, 3, 4)
        pad_mask = np.stack([pad_mask for _ in range(images.shape[0])], axis=0)

        # we need use a different rng key for each model forward step; this has a large impact on model performance
        self.rng, key = jax.random.split(self.rng)  # each shape [2,]
        # print("octo local rng", self.rng, key)

        input_observation = {"image_primary": images, "timestep_pad_mask": pad_mask}
        # hard-coded solution to align the batch size
        norm_raw_actions, attention_weights = self.model.sample_actions(
            input_observation,
            self.task,
            rng=key,
        )
        raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]
        # use the original policy output for unnormalized action dimension
        raw_actions = raw_actions * self.action_normalization_mask + norm_raw_actions * (1. - self.action_normalization_mask)

        # assert raw_actions.shape == (self.pred_action_horizon, 7)
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)
        else:
            raw_actions = raw_actions[:, 0]

        raw_action = {
            "world_vector": np.array(raw_actions[:, :3]),
            "rotation_delta": np.array(raw_actions[:, 3:6]),
            "open_gripper": np.array(raw_actions[:, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action["rot_euler"] = raw_action["rotation_delta"] * self.action_scale
        action_rotation_delta = raw_action["rotation_delta"].astype(np.float64)
        action_rotation_ax, action_rotation_angle = [], []
        for i in range(action_rotation_delta.shape[0]):
            roll, pitch, yaw = action_rotation_delta[i]
            ax, angle = euler2axangle(roll, pitch, yaw)
            action_rotation_ax.append(ax)
            action_rotation_angle.append(angle)
        action_rotation_ax = np.array(action_rotation_ax)
        action_rotation_angle = np.array(action_rotation_angle)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle[:, None]
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        
        if self.policy_setup == 'libero':
            action["gripper"] = 2 * raw_action["open_gripper"] - 1

        return raw_action, action, attention_weights, image, (self.task_description, self.task)
