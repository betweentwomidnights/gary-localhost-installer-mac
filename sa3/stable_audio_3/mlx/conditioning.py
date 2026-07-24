from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import numpy as np

from stable_audio_3.mlx.dit_blocks import ExpoFourierFeatures
from stable_audio_3.mlx.runtime import import_mlx_core
from stable_audio_3.mlx.runtime import import_mlx_nn
from stable_audio_3.mlx.spec import extract_mlx_port_requirements

mx = import_mlx_core(required=True)
nn = import_mlx_nn(required=True)


@dataclass(frozen=True)
class NumberConditionerConversionReport:
    unexpected_keys: list[str]
    transposed_keys: list[str]


class MLXNumberConditioner(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        min_val: float = 0.0,
        max_val: float = 1.0,
        fourier_features_dim: int = 256,
        fourier_features_type: str = "expo",
        param_dtype=mx.float32,
    ):
        super().__init__()
        if fourier_features_type != "expo":
            raise NotImplementedError(
                "Only expo number conditioning is implemented for the SA3 MLX smoke path."
            )

        self.output_dim = int(output_dim)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.fourier_features_type = str(fourier_features_type)
        self.param_dtype = param_dtype
        self.features = ExpoFourierFeatures(dim=int(fourier_features_dim))
        self.proj = nn.Linear(int(fourier_features_dim), self.output_dim)

    @classmethod
    def from_torch_conditioner(
        cls,
        torch_conditioner,
        *,
        mlx_dtype=mx.float32,
    ) -> tuple["MLXNumberConditioner", NumberConditionerConversionReport]:
        state = torch_conditioner.state_dict()
        weight_key = "embedder.embedding.1.weight"
        bias_key = "embedder.embedding.1.bias"
        if weight_key not in state or bias_key not in state:
            raise ValueError("Torch NumberConditioner does not expose the expected linear state keys.")

        weight = state[weight_key].detach().cpu().float().numpy()
        bias = state[bias_key].detach().cpu().float().numpy()
        output_dim, fourier_features_dim = weight.shape
        torch_features_type = str(torch_conditioner.embedder.embedding[0].__class__.__name__).lower()
        if torch_features_type == "expofourierfeatures":
            torch_features_type = "expo"
        conditioner = cls(
            output_dim=int(output_dim),
            min_val=float(torch_conditioner.min_val),
            max_val=float(torch_conditioner.max_val),
            fourier_features_dim=int(fourier_features_dim),
            fourier_features_type=torch_features_type,
            param_dtype=mlx_dtype,
        )

        conditioner.proj.weight = mx.array(weight.astype(np.float32, copy=False)).astype(mlx_dtype)
        conditioner.proj.bias = mx.array(bias.astype(np.float32, copy=False)).astype(mlx_dtype)
        mx.eval(conditioner.parameters())

        unexpected = sorted(k for k in state if k not in {weight_key, bias_key})
        return (
            conditioner,
            NumberConditionerConversionReport(
                unexpected_keys=unexpected,
                transposed_keys=[],
            ),
        )

    def __call__(self, values: list[float] | tp.Any):
        arr = mx.array(values, dtype=mx.float32)
        arr = mx.clip(arr, self.min_val, self.max_val)
        arr = (arr - self.min_val) / (self.max_val - self.min_val)
        embeddings = self.proj(self.features(arr).astype(self.param_dtype))[:, None, :]
        mask = mx.ones((int(embeddings.shape[0]), 1), dtype=mx.bool_)
        return embeddings, mask


def _to_numpy(value: tp.Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _to_mlx_array(value: tp.Any, *, dtype_name: str = "float32", bool_mask: bool = False):
    if value is None:
        return None
    if isinstance(value, mx.array):
        out = value
    else:
        arr = _to_numpy(value)
        out = mx.array(arr)
    if bool_mask:
        return out.astype(mx.bool_)
    return out.astype(getattr(mx, dtype_name))


def _split_conditioning_entry(entry: tp.Any):
    if isinstance(entry, (list, tuple)):
        if len(entry) == 0:
            raise ValueError("Conditioning entries must not be empty.")
        if len(entry) == 1:
            return entry[0], None
        return entry[0], entry[1]
    return entry, None


def normalize_conditioning_tensors(
    conditioning_tensors: dict[str, tp.Any],
    *,
    dtype_name: str = "float32",
) -> dict[str, tuple[tp.Any, tp.Any | None]]:
    normalized: dict[str, tuple[tp.Any, tp.Any | None]] = {}
    for key, value in conditioning_tensors.items():
        tensor, mask = _split_conditioning_entry(value)
        normalized[key] = (
            _to_mlx_array(tensor, dtype_name=dtype_name, bool_mask=False),
            _to_mlx_array(mask, dtype_name=dtype_name, bool_mask=True),
        )
    return normalized


def with_default_inpaint_tensors(
    model_config: dict[str, tp.Any],
    conditioning_tensors: dict[str, tuple[tp.Any, tp.Any | None]],
    *,
    latent_length: int,
    dtype_name: str = "float32",
    default_inpaint_mode: str = "inference",
) -> dict[str, tuple[tp.Any, tp.Any | None]]:
    """Fill missing local inpaint inputs for pure generation.

    Stable Audio 3 mask semantics are 0=generate and 1=provided context, so
    omitted inpaint conditioning is an all-zero mask plus zero context in both
    inference and full-generation training. Trainers that mix inpainting modes
    supply explicit masks instead of relying on this default.
    """
    requirements = extract_mlx_port_requirements(model_config)
    normalized = dict(conditioning_tensors)

    if not requirements.diffusion.local_add_cond_ids:
        return normalized

    if "inpaint_mask" in normalized and "inpaint_masked_input" in normalized:
        return normalized

    if not normalized:
        raise ValueError(
            "Cannot infer batch size for default inpaint tensors from an empty conditioning_tensors dict."
        )

    first_tensor = next(iter(normalized.values()))[0]
    batch_size = int(first_tensor.shape[0])
    dtype = getattr(mx, dtype_name)
    io_channels = int(model_config["model"]["diffusion"]["config"]["io_channels"])
    if default_inpaint_mode in {"inference", "training"}:
        default_mask = mx.zeros((batch_size, 1, latent_length), dtype=dtype)
    else:
        raise ValueError(
            "default_inpaint_mode must be 'inference' or 'training', "
            f"got {default_inpaint_mode!r}."
        )

    normalized.setdefault(
        "inpaint_mask",
        (default_mask, None),
    )
    normalized.setdefault(
        "inpaint_masked_input",
        (mx.zeros((batch_size, io_channels, latent_length), dtype=dtype), None),
    )
    return normalized


def assemble_conditioning_inputs_from_tensors(
    model_config: dict[str, tp.Any],
    conditioning_tensors: dict[str, tp.Any],
    *,
    negative: bool = False,
    latent_length: int | None = None,
    dtype_name: str = "float32",
    default_inpaint_mode: str = "inference",
) -> dict[str, tp.Any]:
    requirements = extract_mlx_port_requirements(model_config)
    normalized = normalize_conditioning_tensors(conditioning_tensors, dtype_name=dtype_name)

    if latent_length is not None:
        normalized = with_default_inpaint_tensors(
            model_config,
            normalized,
            latent_length=latent_length,
            dtype_name=dtype_name,
            default_inpaint_mode=default_inpaint_mode,
        )

    cross_attention_input = None
    cross_attention_mask = None
    global_embed = None
    input_concat_cond = None
    local_add_cond = None
    prepend_cond = None
    prepend_cond_mask = None

    if requirements.diffusion.cross_attention_cond_ids:
        cross_inputs = []
        cross_masks = []
        for key in requirements.diffusion.cross_attention_cond_ids:
            cross_in, cross_mask = normalized[key]
            if cross_in.ndim == 2:
                cross_in = cross_in[:, None, :]
            if cross_mask is None:
                cross_mask = mx.ones((cross_in.shape[0], cross_in.shape[1]), dtype=mx.bool_)
            elif cross_mask.ndim == 1:
                cross_mask = cross_mask[:, None]
            cross_inputs.append(cross_in)
            cross_masks.append(cross_mask.astype(mx.bool_))

        cross_attention_input = mx.concatenate(cross_inputs, axis=1)
        cross_attention_mask = mx.concatenate(cross_masks, axis=1)

    if requirements.diffusion.global_cond_ids:
        global_parts = [normalized[key][0] for key in requirements.diffusion.global_cond_ids]
        global_embed = mx.concatenate(global_parts, axis=-1)
        if global_embed.ndim == 3 and global_embed.shape[1] == 1:
            global_embed = mx.squeeze(global_embed, axis=1)

    if requirements.diffusion.input_concat_ids:
        input_concat_cond = mx.concatenate(
            [normalized[key][0] for key in requirements.diffusion.input_concat_ids],
            axis=1,
        )

    if requirements.diffusion.local_add_cond_ids:
        missing = [key for key in requirements.diffusion.local_add_cond_ids if key not in normalized]
        if missing:
            raise ValueError(
                "Missing local_add conditioning tensors. "
                f"Required keys: {list(requirements.diffusion.local_add_cond_ids)}, missing: {missing}."
            )
        local_add_cond = mx.concatenate(
            [normalized[key][0] for key in requirements.diffusion.local_add_cond_ids],
            axis=1,
        )

    if requirements.diffusion.prepend_cond_ids:
        prepend_inputs = []
        prepend_masks = []
        for key in requirements.diffusion.prepend_cond_ids:
            prepend_in, prepend_mask = normalized[key]
            prepend_inputs.append(prepend_in)
            if prepend_mask is None:
                prepend_mask = mx.ones(
                    (prepend_in.shape[0], prepend_in.shape[1]),
                    dtype=mx.bool_,
                )
            prepend_masks.append(prepend_mask.astype(mx.bool_))
        prepend_cond = mx.concatenate(prepend_inputs, axis=1)
        prepend_cond_mask = mx.concatenate(prepend_masks, axis=1)

    if negative:
        return {
            "negative_cross_attn_cond": cross_attention_input,
            "negative_cross_attn_mask": cross_attention_mask,
            "negative_global_embed": global_embed,
            "negative_input_concat_cond": input_concat_cond,
        }

    return {
        "cross_attn_cond": cross_attention_input,
        "cross_attn_cond_mask": cross_attention_mask,
        "global_embed": global_embed,
        "input_concat_cond": input_concat_cond,
        "local_add_cond": local_add_cond,
        "prepend_cond": prepend_cond,
        "prepend_cond_mask": prepend_cond_mask,
    }


def build_mlx_conditioning_inputs_from_torch_model(
    torch_model,
    model_config: dict[str, tp.Any],
    conditioning: list[dict[str, tp.Any]],
    *,
    negative_conditioning: list[dict[str, tp.Any]] | None = None,
    device: str = "cpu",
    latent_length: int | None = None,
    dtype_name: str = "float32",
) -> dict[str, tp.Any]:
    if not hasattr(torch_model, "conditioner"):
        raise ValueError("torch_model must expose a `.conditioner` attribute.")

    conditioning_tensors = torch_model.conditioner(conditioning, device)
    cond_inputs = assemble_conditioning_inputs_from_tensors(
        model_config,
        conditioning_tensors,
        negative=False,
        latent_length=latent_length,
        dtype_name=dtype_name,
    )

    neg_inputs: dict[str, tp.Any] = {}
    if negative_conditioning is not None:
        negative_tensors = torch_model.conditioner(negative_conditioning, device)
        neg_inputs = assemble_conditioning_inputs_from_tensors(
            model_config,
            negative_tensors,
            negative=True,
            latent_length=latent_length,
            dtype_name=dtype_name,
        )

    return {**cond_inputs, **neg_inputs}


def _conditioner_inputs_for_key(
    conditioning: list[dict[str, tp.Any]],
    key: str,
) -> list[tp.Any]:
    values = []
    for item in conditioning:
        if key not in item:
            raise ValueError(f"Conditioner key {key!r} not found in conditioning item.")
        value = item[key]
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        values.append(value)
    return values


def build_mlx_conditioning_tensors(
    model_config: dict[str, tp.Any],
    conditioning: list[dict[str, tp.Any]],
    *,
    text_conditioners: dict[str, tp.Any],
    number_conditioners: dict[str, MLXNumberConditioner],
) -> dict[str, tuple[tp.Any, tp.Any | None]]:
    requirements = extract_mlx_port_requirements(model_config)
    outputs: dict[str, tuple[tp.Any, tp.Any | None]] = {}

    for spec in requirements.conditioners:
        values = _conditioner_inputs_for_key(conditioning, spec.id)
        if spec.type == "t5gemma":
            if spec.id not in text_conditioners:
                raise ValueError(f"Missing MLX text conditioner for {spec.id!r}.")
            outputs[spec.id] = text_conditioners[spec.id]([str(value) for value in values])
        elif spec.type == "number":
            if spec.id not in number_conditioners:
                raise ValueError(f"Missing MLX number conditioner for {spec.id!r}.")
            outputs[spec.id] = number_conditioners[spec.id]([float(value) for value in values])
        else:
            raise NotImplementedError(
                f"MLX conditioner type {spec.type!r} is not implemented for {spec.id!r}."
            )

    return outputs


def build_mlx_conditioning_inputs(
    model_config: dict[str, tp.Any],
    conditioning: list[dict[str, tp.Any]],
    *,
    text_conditioners: dict[str, tp.Any],
    number_conditioners: dict[str, MLXNumberConditioner],
    negative_conditioning: list[dict[str, tp.Any]] | None = None,
    latent_length: int | None = None,
    dtype_name: str = "float32",
) -> dict[str, tp.Any]:
    conditioning_tensors = build_mlx_conditioning_tensors(
        model_config,
        conditioning,
        text_conditioners=text_conditioners,
        number_conditioners=number_conditioners,
    )
    cond_inputs = assemble_conditioning_inputs_from_tensors(
        model_config,
        conditioning_tensors,
        negative=False,
        latent_length=latent_length,
        dtype_name=dtype_name,
    )

    neg_inputs: dict[str, tp.Any] = {}
    if negative_conditioning is not None:
        negative_tensors = build_mlx_conditioning_tensors(
            model_config,
            negative_conditioning,
            text_conditioners=text_conditioners,
            number_conditioners=number_conditioners,
        )
        neg_inputs = assemble_conditioning_inputs_from_tensors(
            model_config,
            negative_tensors,
            negative=True,
            latent_length=latent_length,
            dtype_name=dtype_name,
        )

    return {**cond_inputs, **neg_inputs}
