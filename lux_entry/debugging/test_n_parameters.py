from lux_entry.training.env import make_env
from lux_entry.training.model import JobActionNet, JobNet, MapFeaturesExtractor
from lux_entry.training.observations import get_full_obs_space


def test_n_parameters():
    env = make_env(0)()
    obs_space = get_full_obs_space(env.state.env_cfg)
    for name, model in {
        "JobNet": JobNet(),
        "JobActionNet": JobActionNet(),
        "MapFeaturesExtractor": MapFeaturesExtractor(obs_space),
    }.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name} has {n_params} parameters")
