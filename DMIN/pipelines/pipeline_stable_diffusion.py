import os
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from torchvision.transforms.functional import pil_to_tensor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import torch.nn.functional as F
from DMIN.utils import print_model_summary, print_mem
from DMIN.dim_reducer import DimReducer
import numpy as np
import PIL.Image

# from imgcat import imgcat
# import matplotlib
# matplotlib.use("module://imgcat")


class IFStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

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
        self.cache_path = DMIN_config.influence.cache_path
        self.cpu_offload = self.DMIN_config.influence.cpu_offload

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
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        if not self.has_init:
            self.has_init = True
            self.has_adapter()
        allback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if image is not None and not isinstance(image, list):
            image = [image]
        image = self.image_processor.preprocess(image, height=height, width=width)
        image = image.to(self.unet.dtype).to(self.unet.device)  ###

        device = self._execution_device

        # 3. Encode input prompt

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

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
        )[1:]
        timesteps = self.scheduler.timesteps[x].to(self.unet.device) - 1

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents_output = self.vae.encode(image)
        ori_latents = latents_output.latent_dist.sample()
        ori_latents = ori_latents * self.vae.config.scaling_factor

        if self.noise is None:
            if self.load_from_cache and os.path.exists(f"{self.cache_path}/noise.pkl"):
                self.noise = torch.load(
                    f"{self.cache_path}/noise.pkl", map_location=ori_latents.device
                )
            else:
                self.noise = torch.randn(
                    ori_latents.shape[1:],
                    device=ori_latents.device,
                    dtype=ori_latents.dtype,
                )
                if self.save_to_cache and self.cache_path:
                    torch.save(self.noise, f"{self.cache_path}/noise.pkl")
        noise = self.noise

        loss_grad_list = []
        # 7. Denoising loop
        for i, t in enumerate(timesteps):
            t = t.unsqueeze(-1)

            latent_model_input = self.scheduler.add_noise(ori_latents, noise, t)
            timestep = t.expand(latent_model_input.shape[0])

            with torch.enable_grad():
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                )[0]

                loss = F.mse_loss(noise_pred, noise.unsqueeze(0))
                loss.backward()

                loss_grad = None
                if self.has_adapter:
                    loss_grad = torch.cat(
                        [
                            p.grad.detach().reshape(-1)
                            for name, p in self.unet.named_parameters()
                            if p.grad is not None
                            and "transformer_blocks" in name
                            and "lora" in name
                        ]
                    )
                else:
                    loss_grad = torch.cat(
                        [
                            p.grad.detach().reshape(-1).to(torch.float16)
                            for name, p in self.unet.named_parameters()
                            if p.grad is not None and "transformer_blocks" in name
                        ]
                    )
                if self.cpu_offload:
                    loss_grad = loss_grad.cpu()

                loss_grad_list.append(loss_grad)

                self.unet.zero_grad()

        # Offload all models
        self.maybe_free_model_hooks()

        return torch.stack(loss_grad_list, dim=0)
