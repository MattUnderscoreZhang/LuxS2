import io
import sys
import torch
import zipfile

from lux_entry.behaviors.starter_kit import net


def load_net(model_class: type[net.Net], model_path: str) -> net.Net:
    # TODO: try replacing function with evaluate() in train.py
    # load .pth or .zip
    if model_path[-4:] == ".zip":
        with zipfile.ZipFile(model_path) as archive:
            file_path = "policy.pth"
            with archive.open(file_path, mode="r") as param_file:
                file_content = io.BytesIO()
                file_content.write(param_file.read())
                file_content.seek(0)
                sb3_state_dict = torch.load(file_content, map_location="cpu")
    else:
        sb3_state_dict = torch.load(model_path, map_location="cpu")
    # print(sb3_state_dict, file=sys.stderr)

    # net_keys = []
    # for sb3_key in sb3_state_dict.keys():
        # if sb3_key.startswith("features_extractor."):
            # net_keys.append(sb3_key)
            # # TODO: check if f.e. keys are == pi_f.e., vf_f.e., mlp_e.
    net_keys = sb3_state_dict.keys()
    for key in net_keys:
        print(key, sb3_state_dict[key].shape)

    net = model_class()
    loaded_state_dict = {}
    for sb3_key, model_key in zip(net_keys, net.state_dict().keys()):
        loaded_state_dict[model_key] = sb3_state_dict[sb3_key]
        print("loaded", sb3_key, "->", model_key, file=sys.stderr)

    net.load_state_dict(loaded_state_dict)
    return net


if __name__ == "__main__":
    load_net(net.Net, net.WEIGHTS_PATH)
