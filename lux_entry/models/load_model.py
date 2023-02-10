import io
import sys
import torch
import torch.nn as nn
import zipfile


def load(model_class: type[nn.Module], model_path: str) -> nn.Module:
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

    model = model_class()
    loaded_state_dict = {}

    # this code here works assuming the first keys in the sb3 state dict are aligned with the ones you define above in Net
    for sb3_key, model_key in zip(sb3_state_dict.keys(), model.state_dict().keys()):
        loaded_state_dict[model_key] = sb3_state_dict[sb3_key]
        print("loaded", sb3_key, "->", model_key, file=sys.stderr)

    model.load_state_dict(loaded_state_dict)
    return model
