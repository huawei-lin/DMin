from datasets import load_dataset
from torchvision import transforms
from diffusers import DPMSolverMultistepScheduler
from DMIN import DiTPipeline
import torch

resolution = 512


def preprocess_train(examples):
    global resolution
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomCrop(resolution),
            # transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    images = [image.convert("RGB") for image in examples["image"]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["class_id"] = [
        torch.tensor(x, dtype=torch.int16) for x in examples["label"]
    ]
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    class_ids = torch.stack([example["class_id"] for example in examples])
    return {"pixel_values": pixel_values, "class_ids": class_ids}


def get_data_loader(dataset_path_or_name):
    dataset = load_dataset(dataset_path_or_name)
    train_dataset = dataset["train"].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1
    )
    return train_dataloader


def get_pipeline(model_path_or_name, device=None, torch_dtype=None):
    pipe = DiTPipeline.from_pretrained(model_path_or_name, torch_dtype=torch_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe
