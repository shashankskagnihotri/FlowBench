import torch
import torch.nn as nn
import logging

from attacks.help_function.render import render


class ScaledInputWeatherModel(nn.Module):
    def __init__(self, model, **kwargs):
        super(ScaledInputWeatherModel, self).__init__()

        self.model_loaded = model

    def forward(
        self, image1, image2, weather=None, scene_data=None, args_=None, *args, **kwargs
    ):
        torch.cuda.empty_cache()
        if weather is not None:
            image1, image2 = render(image1, image2, scene_data, weather, args_)

        image_pair_tensor = torch.torch.cat(
            (image1.clone().detach(), image2.clone().detach())
        ).unsqueeze(0)
        dic = {"images": image_pair_tensor}

        # return self.model_loaded(image1, image2, *args, **kwargs)
        # return ownutilities.compute_flow(self.model_loaded, self.model_name, image1, image2, test_mode=test_mode, *args, **kwargs)
        return self.model_loaded(dic)
