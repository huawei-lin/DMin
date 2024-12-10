from collections import OrderedDict
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipelines.auto_pipeline import _get_task_class
from DMIN.pipelines.pipeline_stable_diffusion_3 import IFStableDiffusion3Pipeline
from DMIN.pipelines.pipeline_stable_diffusion import IFStableDiffusionPipeline
from DMIN.pipelines.pipeline_ddpm import IFDDPMPipeline

AUTO_PIPELINES_MAPPING = OrderedDict(
    [
        ("StableDiffusion3Pipeline", IFStableDiffusion3Pipeline),
        ("StableDiffusionPipeline", IFStableDiffusionPipeline),
        ("DDPMPipeline", IFDDPMPipeline),
    ]
)


def _get_task_class(
    mapping, pipeline_class_name, throw_error_if_not_exist: bool = True
):
    task_class = mapping.get(pipeline_class_name, None)
    if task_class is not None:
        return task_class

    raise ValueError(
        f"AutoPipeline can't find a pipeline linked to {pipeline_class_name}"
    )


class IFAutoPipeline(ConfigMixin):
    config_name = "model_index.json"

    @classmethod
    def from_pretrained(cls, DMIN_config, pretrained_model_or_path, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        load_config_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "token": token,
            "local_files_only": local_files_only,
            "revision": revision,
        }

        config = cls.load_config(pretrained_model_or_path, **load_config_kwargs)
        orig_class_name = config["_class_name"]
        print(orig_class_name)

        text_2_image_cls = _get_task_class(AUTO_PIPELINES_MAPPING, orig_class_name)

        kwargs = {**load_config_kwargs, **kwargs}
        model = text_2_image_cls.from_pretrained(pretrained_model_or_path, **kwargs)
        model.set_DMIN_config(DMIN_config)
        return model
