from mani_skill2_real2sim.envs.custom_scenes.grasp_single_in_scene import GraspSingleCustomInSceneEnv, GraspSingleCustomOrientationInSceneEnv
from mani_skill2_real2sim.utils.registration import register_env

@register_env("GraspSingleRandomTrainObjectInScene-v0", max_episode_steps=80)
class GraspSingleRandomTrainObjectInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = [
            "opened_pepsi_can",
            # "opened_coke_can",
            "opened_sprite_can",
            "opened_fanta_can",
            "opened_redbull_can",
            "blue_plastic_bottle",
            # "apple",
            "orange",
            "sponge",
            # "bridge_spoon_generated_modified",
            "bridge_carrot_generated_modified",
            # "green_cube_3cm",
            "yellow_cube_3cm",
            "eggplant"
        ]
        super().__init__(**kwargs)

@register_env("GraspSingleGreenCubeInScene-v0", max_episode_steps=80)
class GraspSingleGreenCubeInSceneEnv(GraspSingleCustomInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop("model_ids", None)
        kwargs["model_ids"] = ["green_cube_3cm"]
        super().__init__(**kwargs)