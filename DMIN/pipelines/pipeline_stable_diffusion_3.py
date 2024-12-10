import os
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from torchvision.transforms.functional import pil_to_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
import torch.nn.functional as F
from DMIN.utils import print_model_summary, print_mem
from DMIN.dim_reducer import DimReducer
import PIL.Image
from copy import deepcopy
from diffusers.utils import BaseOutput
from tqdm import tqdm
import numpy as np

# from imgcat import imgcat
import matplotlib

# matplotlib.use("module://imgcat")


class IFStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
        )

        self.load_from_cache = True
        self.save_to_cache = True

        self.DMIN_config = None
        self.cache_path = None
        self.noise = None

        self.cpu_offload = False
        self.has_init = False

    def has_adapter(self):
        self.has_adapter = False
        for name, p in self.transformer.named_parameters():
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
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        index: int = None,
    ):
        r"""
                Function invoked when calling the pipeline for generation.

                Args:
                    prompt (`str` or `List[str]`, *optional*):
                        The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                        instead.
                    prompt_2 (`str` or `List[str]`, *optional*):
                        The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                        will be used instead
                    prompt_3 (`str` or `List[str]`, *optional*):
                        The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                        will be used instead
                    height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The height in pixels of the generated image. This is set to 1024 by default for the best results.
                    width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The width in pixels of the generated image. This is set to 1024 by default for the best results.
                    num_inference_steps (`int`, *optional*, defaults to 50):
                        The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                        expense of slower inference.
                    timesteps (`List[int]`, *optional*):
                        Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                        in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                        passed will be used. Must be in descending order.
                    guidance_scale (`float`, *optional*, defaults to 7.0):
                        Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                        `guidance_scale` is defined as `w` of equation 2. of [Imagen
                        Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                        1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                        usually at the expense of lower image quality.
                    negative_prompt (`str` or `List[str]`, *optional*):
                        The prompt or prompts not to guide the image generation. If not defined, one has to pass
                        `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                        less than `1`).
                    negative_prompt_2 (`str` or `List[str]`, *optional*):
                        The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                        `text_encoder_2`. If not defined, `negative_prompt` is used instead
                    negative_prompt_3 (`str` or `List[str]`, *optional*):
                        The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                        `text_encoder_3`. If not defined, `negative_prompt` is used instead
                    num_images_per_prompt (`int`, *optional*, defaults to 1):
                        The number of images to generate per prompt.
                    generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                        One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                        to make generation deterministic.
                    latents (`torch.FloatTensor`, *optional*):
                        Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                        generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                        tensor will ge generated by sampling using the supplied random `generator`.
                    prompt_embeds (`torch.FloatTensor`, *optional*):
                        Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                        provided, text embeddings will be generated from `prompt` input argument.
                    negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                        Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                        weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                        argument.
                    pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                        Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                        If not provided, pooled text embeddings will be generated from `prompt` input argument.
                    negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                        Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                        weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                        input argument.
                    output_type (`str`, *optional*, defaults to `"pil"`):
                        The output format of the generate image. Choose between
                        [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
                    return_dict (`bool`, *optional*, defaults to `True`):
                        Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                        of a plain tuple.
                    joint_attention_kwargs (`dict`, *optional*):
                        A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                        `self.processor` in
                        [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                    callback_on_step_end (`Callable`, *optional*):
                        A function that calls at the end of each denoising steps during the inference. The function is called
                        with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                        callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                        `callback_on_step_end_tensor_inputs`.
                    callback_on_step_end_tensor_inputs (`List`, *optional*):
                        The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                        will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                        `._callback_tensor_inputs` attribute of your pipeline class.
                    max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
        s
                Examples:

                Returns:
                    [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
                    [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
                    `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        if not self.has_init:
            self.has_init = True
            self.has_adapter()

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
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
        image = image.to(self.transformer.dtype).to(self.transformer.device)

        device = self._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
        )

        if len(self.scheduler.timesteps) != self.scheduler.config.num_train_timesteps:
            self.scheduler.set_timesteps(
                self.scheduler.config.num_train_timesteps,
                device=self.transformer.device,
            )

        # 4. Prepare timesteps
        x = np.linspace(0, 1, num_inference_steps + 1) ** 3
        x = ((x / x.max()) * self.scheduler.config.num_train_timesteps).astype(
            np.int32
        )[:-1]
        timesteps = self.scheduler.timesteps[x].to(self.transformer.device)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents_output = self.vae.encode(image)
        ori_latents = latents_output.latent_dist.sample()
        ori_latents = (
            ori_latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor

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
        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            t = t.unsqueeze(-1)

            sigmas = self.get_sigmas(t, n_dim=ori_latents.ndim, dtype=ori_latents.dtype)
            latents = (1.0 - sigmas) * ori_latents + sigmas * noise

            latent_model_input = latents
            timestep = t.expand(latent_model_input.shape[0])

            with torch.enable_grad():
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred, noise.unsqueeze(0))
                loss.backward()

                loss_grad = None
                if self.has_adapter:
                    loss_grad = torch.cat(
                        [
                            p.grad.detach().reshape(-1)
                            for name, p in self.transformer.named_parameters()
                            if p.grad is not None
                            and "transformer_blocks" in name
                            and "lora" in name
                        ]
                    )
                else:
                    loss_grad = torch.cat(
                        [
                            p.grad.detach().reshape(-1)
                            for name, p in self.transformer.named_parameters()
                            if p.grad is not None and "transformer_blocks" in name
                        ]
                    )
                if self.cpu_offload:
                    loss_grad = loss_grad.cpu()

                loss_grad_list.append(loss_grad)

                self.transformer.zero_grad()

        # Offload all models
        self.maybe_free_model_hooks()

        return torch.stack(loss_grad_list, dim=0)

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(timesteps.device)
        timesteps = timesteps.to(timesteps.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_sigmas_diff(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(timesteps.device)
        timesteps = timesteps.to(timesteps.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma_diff = (
            sigmas[[x + 1 for x in step_indices]].flatten()
            - sigmas[step_indices].flatten()
        )
        while len(sigma_diff.shape) < n_dim:
            sigma_diff = sigma_diff.unsqueeze(-1)
        return sigma_diff
