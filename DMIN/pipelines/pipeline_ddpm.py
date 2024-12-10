import os
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.transforms.functional import pil_to_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
import torch.nn.functional as F
from DMIN.utils import print_model_summary, print_mem
from DMIN.dim_reducer import DimReducer
import numpy as np
import PIL.Image

# from imgcat import imgcat
# import matplotlib
# matplotlib.use("module://imgcat")


class IFDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler):
        super().__init__(unet=unet, scheduler=scheduler)

        self.load_from_cache = True
        self.save_to_cache = True
        self.cpu_offload = False

        self.DMIN_config = None
        self.cache_path = None
        self.noise = None

        self.has_init = False

    def has_adapter(self):
        self.has_adapter = False
        for name, p in self.unet.named_parameters():
            if "lora" in name:
                self.has_adapter = True
                break

    def set_DMIN_config(self, DMIN_config):
        self.DMIN_config = DMIN_config
        self.cpu_offload = self.DMIN_config.influence.cpu_offload
        self.cache_path = DMIN_config.influence.cache_path

    def set_DMIN_noise(self, noise):
        self.noise = noise

    @torch.no_grad()
    def calculate_influence_score(
        self,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
        ] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if not self.has_init:
            self.has_init = True
            self.has_adapter()

        if image is not None and not isinstance(image, list):
            image = [image]
        if self.unet.config.in_channels == 1:
            image = np.stack(
                [
                    np.array(img.convert("L")).astype(np.float32) / 255.0
                    for img in image
                ],
                axis=0,
            )
            image = torch.from_numpy(image).unsqueeze(1)
        else:
            image = np.stack(
                [
                    np.array(img.convert("RGB")).astype(np.float32) / 255.0
                    for img in image
                ],
                axis=0,
            )
            image = torch.from_numpy(image).permute(0, 3, 1, 2)
        image = image.to(self.unet.dtype).to(self.unet.device)  ###

        device = self._execution_device

        if (
            self.scheduler.timesteps is None
            or len(self.scheduler.timesteps)
            != self.scheduler.config.num_train_timesteps
        ):
            self.scheduler.set_timesteps(
                self.scheduler.config.num_train_timesteps, device=self.unet.device
            )

        # 4. Prepare timesteps
        x = np.linspace(0, 1, num_inference_steps + 1) ** 3
        x = ((x / x.max()) * (self.scheduler.config.num_train_timesteps)).astype(
            np.int32
        )[:-1]
        timesteps = self.scheduler.timesteps[x].to(self.unet.device)

        if self.noise is None:
            if self.load_from_cache and os.path.exists(f"{self.cache_path}/noise.pkl"):
                self.noise = torch.load(
                    f"{self.cache_path}/noise.pkl", map_location=image.device
                )
            else:
                self.noise = torch.randn(
                    image.shape[1:], device=image.device, dtype=image.dtype
                )
                if self.save_to_cache and self.cache_path:
                    torch.save(self.noise, f"{self.cache_path}/noise.pkl")
        noise = self.noise.to(image.device)

        loss_grad_list = []
        # 7. Denoising loop
        for i, t in enumerate(timesteps):

            noisy_images = self.scheduler.add_noise(image, noise, t)
            timestep = t.expand(noisy_images.shape[0])
            with torch.enable_grad():
                # predict the noise residual
                model_output = self.unet(
                    noisy_images,
                    timestep,
                ).sample

                loss = F.mse_loss(
                    model_output.float(), noise.float()
                )  # this could have different weights!
                loss.backward()

                loss_grad = None
                if self.has_adapter:
                    loss_grad = torch.cat(
                        [
                            p.grad.detach().reshape(-1)
                            for name, p in self.unet.named_parameters()
                            if p.grad is not None and "lora" in name
                        ]
                    )
                else:
                    loss_grad = torch.cat(
                        [
                            p.grad.detach().reshape(-1)
                            for name, p in self.unet.named_parameters()
                            if p.grad is not None
                        ]
                    )
                if self.cpu_offload:
                    loss_grad = loss_grad.cpu()

                loss_grad_list.append(loss_grad)

                self.unet.zero_grad()

        return torch.stack(loss_grad_list, dim=0)
