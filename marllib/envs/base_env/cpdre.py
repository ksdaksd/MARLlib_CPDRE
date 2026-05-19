"""
cpdre.py

MARLlib adapter for CPDRE:
Coal-Power Direct Reciprocity Environment.
"""

from ray.tune.registry import register_env

from custom_envs.coal_power_direct_reciprocity_env import CPDRE


# class RLLibCPDRE(CPDRE):
#     """
#     MARLlib 识别用的 CPDRE 环境类。
#
#     真正环境逻辑在 custom_envs/coal_power_direct_reciprocity_env.py。
#     这里负责让 MARLlib 的 ENV_REGISTRY 找到这个环境。
#     """
#
#     def __init__(self, env_config=None):
#         env_config = env_config or {}
#         super().__init__(env_config)
#
#         # self.env_name = "cpdre"
#         # self.map_name = self.config.__dict__.get("map_name", "direct_1c3u")
#         self.env_name = "cpdre"
#         self.map_name = self.config.__dict__.get("map_name", "direct_1c3u")
#         self.env_info = self.get_env_info()



class RLLibCPDRE(CPDRE):
    def __init__(self, env_config=None):
        env_config = env_config or {}
        raw_env_args = env_config.get("env_args", env_config)

        super().__init__(env_config)

        self.env_name = "cpdre"
        self.map_name = raw_env_args.get("map_name", "direct_1c3u")
        self.env_info = self.get_env_info()



def env_creator(env_config):
    return RLLibCPDRE(env_config)


def register_cpdre():
    register_env("cpdre", env_creator)


register_cpdre()