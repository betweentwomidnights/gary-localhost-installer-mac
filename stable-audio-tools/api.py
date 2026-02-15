#!/usr/bin/env python3
"""
Stable Audio API - Enhanced with Style Transfer capabilities
Designed to be called alongside existing websockets backend
"""

from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
import torch
import torchaudio
import io
import base64
import uuid
import os
import time
import re
import threading
import gc
import sys
import types
import json
import hashlib
from einops import rearrange
from huggingface_hub import login, hf_hub_download
from contextlib import contextmanager
import contextlib
from pathlib import Path

# Compatibility shim:
# Some transitive dependencies (e.g. clip) still import
# "from pkg_resources import packaging". Newer setuptools releases may not
# ship pkg_resources, so provide a minimal replacement before loading
# stable_audio_tools and its dependencies.
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    try:
        from packaging import version as _packaging_version
    except ModuleNotFoundError:
        from setuptools._vendor.packaging import version as _packaging_version

    _packaging_module = types.SimpleNamespace(version=_packaging_version)
    _pkg_resources_module = types.SimpleNamespace(packaging=_packaging_module)
    sys.modules["pkg_resources"] = _pkg_resources_module

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.utils import load_ckpt_state_dict

from riff_manager import RiffManager

from model_loader_enhanced import model_manager, load_model

import soundfile as sf

from stable_audio_tools.inference.utils import prepare_audio
from stable_audio_tools.inference import generation as sa_generation
from stable_audio_tools.inference.sampling import sample_rf_guided

import numpy as np

import ctypes

# Load .env if present for HF_TOKEN and other config
load_dotenv()

def aggressive_cpu_cleanup():
    """More thorough cleanup of CPU (and CUDA allocator) memory.

    Safe to call after a request finishes. Does NOT unload models by itself,
    it only cleans up objects that are no longer referenced.
    """
    # Extra GC passes to really collect temporary tensors / arrays
    for _ in range(3):
        gc.collect()

    # On Linux, ask glibc to return free heap pages to the OS
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        # Non-glibc systems or failure: just ignore
        pass

    # Clean CUDA allocator bookkeeping
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # Clean MPS allocator bookkeeping
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            torch.mps.synchronize()
        except Exception:
            pass

def autocast_ctx(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if dev.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return contextlib.nullcontext()

def maybe_empty_cache(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

def make_fade_mask(T, bars, device, dtype, floor: float = 0.3):
    """
    strong on bar 1, gradually decaying to `floor` instead of 0.0
    """
    bars = max(int(bars), 1)
    bar_len = max(T // bars, 1)

    M = torch.zeros((1, 1, T), device=device, dtype=dtype)

    # First bar: full weight
    M[:, :, :bar_len] = 1.0

    # Remaining bars: linear decay from 1.0 -> floor
    if bars > 1:
        remaining = T - bar_len
        if remaining > 0:
            ramp = torch.linspace(1.0, floor, remaining, device=device, dtype=dtype)
            M[:, :, bar_len:] = ramp

    return M

def build_latent_guidance_from_audio(
    model,
    input_audio: torch.Tensor,  # [channels, samples]
    input_sr: int,
    sample_rate: int,
    sample_size: int,
    bars: int,
    strength: float = 1.0,
    t_min: float = 0.2,
    t_max: float = 0.999,
    mask_mode: str = "fade",          # "first_bar" | "full" | "fade"
    start_weight: float | None = None,
    end_weight: float | None = None,
):
    """
    Prepare Hawley-style latent-only inpainting guidance from an input riff.

    - Resamples + crops/pads the riff to model's sample_rate/sample_size
    - Encodes to latents if model.pretransform is present
    - Builds a time mask in latent space with an optional fade from
      start_weight -> end_weight over the clip.

      mask_mode="first_bar":   bar 1 full, rest 0 (then scaled by weights)
      mask_mode="full":        full clip = 1 (then scaled by weights)
      mask_mode="fade":        bar 1 strong, later bars decay toward a floor
    """

    device = next(model.parameters()).device

    # input_audio from process_input_audio: [channels, samples]
    audio = input_audio.to(device)

    # Ensure [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() > 2:
        # e.g. [batch, channels, samples] -> [channels, samples]
        audio = audio.squeeze(0)

    # Match generate_diffusion_cond's io_channels logic
    io_channels = model.io_channels
    if getattr(model, "pretransform", None) is not None:
        io_channels = model.pretransform.io_channels

    # Use stable-audio-tools' prepare_audio to match window length
    audio_prepared = prepare_audio(
        audio,
        in_sr=input_sr,
        target_sr=sample_rate,
        target_length=sample_size,
        target_channels=io_channels,
        device=device,
    )  # [channels, sample_size]

    # Encode to latents for SAOS (latent model)
    if getattr(model, "pretransform", None) is not None:
        z_y = model.pretransform.encode(audio_prepared)  # [B, C_latent, T_latent]
    else:
        z_y = audio_prepared.unsqueeze(0)  # [1, C, T]

    B, C, T = z_y.shape

    # Bar info for logging / fade shapes
    bars = max(int(bars), 1)
    bar_len_latent = max(T // bars, 1)
    first_bar_end = bar_len_latent

    # ---- base mask by mode (before steerable fading) ----
    mask_mode = (mask_mode or "fade").lower()

    if mask_mode == "full":
        base_M = torch.ones((1, 1, T), device=device, dtype=z_y.dtype)
        mask_desc = f"full clip (all {T} frames)"

    elif mask_mode == "fade":
        # decay to floor, can tune floor if you want
        base_M = make_fade_mask(T, bars, device=device, dtype=z_y.dtype, floor=0.3)
        mask_desc = f"fade: first bar_len={bar_len_latent}, then decay toward 0.3"

    else:  # "first_bar" (or unknown -> default)
        base_M = torch.zeros((1, 1, T), device=device, dtype=z_y.dtype)
        base_M[:, :, :first_bar_end] = 1.0
        mask_desc = f"first_bar only: first {first_bar_end}/{T} frames"

    # Expand to batch if needed
    if B > 1:
        base_M = base_M.expand(B, -1, -1).contiguous()

    # ---- steerable fade: start_weight -> end_weight over time ----
    # If not provided, default both to 1.0 (original behavior modulo strength).
    if start_weight is None:
        start_weight = 1.0
    if end_weight is None:
        end_weight = start_weight

    # ramp(t) in [start_weight, end_weight] over latent time axis
    if start_weight == end_weight:
        time_ramp = torch.full(
            (1, 1, T),
            fill_value=float(start_weight),
            device=device,
            dtype=z_y.dtype,
        )
        ramp_desc = f"const={start_weight}"
    else:
        ramp = torch.linspace(
            float(start_weight),
            float(end_weight),
            T,
            device=device,
            dtype=z_y.dtype,
        )  # [T]
        time_ramp = ramp.view(1, 1, T)
        ramp_desc = f"{start_weight}→{end_weight}"

    if B > 1:
        time_ramp = time_ramp.expand(B, -1, -1).contiguous()

    # Final mask: base shape (bars/decay) * time_ramp (steerable level)
    M = base_M * time_ramp
    M_sq = M ** 2

    guidance = {
        "mode": "latent_inpaint",
        "z_y": z_y,       # encoded riff latents
        "M_sq": M_sq,     # [B,1,T] mask in latent coords
        "strength": strength,
        "t_min": t_min,
        "t_max": t_max,
    }

    print(
        f"🎛 Latent guidance: z_y shape={z_y.shape}, "
        f"mask_mode={mask_mode}, {mask_desc}, "
        f"fade={ramp_desc}, strength={strength}"
    )
    return guidance


def generate_diffusion_cond_guided(
    model,
    steps: int,
    cfg_scale: float,
    conditioning: dict,
    negative_conditioning: dict,
    sample_size: int,
    sample_rate: int,
    seed: int,
    device: str,
    sampler_kwargs: dict,
    guidance: dict,
):
    """
    Minimal rectified_flow / rf_denoiser generator that uses sample_rf_guided.

    Mirrors stable_audio_tools.inference.generation.generate_diffusion_cond
    for RF objectives, but with an extra 'guidance' dict.
    """

    batch_size = 1  # loop use-case

    # --- audio vs latent sizes (THIS WAS THE MISSING BIT) ---
    audio_sample_size = sample_size  # length in audio samples

    if getattr(model, "pretransform", None) is not None:
        # Downsample sample_size to latent length
        sample_size = sample_size // model.pretransform.downsampling_ratio

    # --- seed / RNG ---
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    print(f"guided RF seed: {seed}")
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # --- initial noise in latent or audio space (matching upstream) ---
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    # --- conditioning tensors (same as generate_diffusion_cond) ---
    assert conditioning is not None, "conditioning dict is required"
    conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None:
        negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
        negative_conditioning_tensors = model.get_conditioning_inputs(
            negative_conditioning_tensors, negative=True
        )
    else:
        negative_conditioning_tensors = {}

    # --- dtype casting ---
    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {
        k: (v.type(model_dtype) if v is not None else v)
        for k, v in conditioning_inputs.items()
    }
    negative_conditioning_tensors = {
        k: (v.type(model_dtype) if v is not None else v)
        for k, v in negative_conditioning_tensors.items()
    }

    diff_objective = getattr(model, "diffusion_objective", None)
    if diff_objective not in ("rectified_flow", "rf_denoiser"):
        raise ValueError(
            f"generate_diffusion_cond_guided only supports RF objectives, got {diff_objective}"
        )

    sampler_kwargs = dict(sampler_kwargs or {})

    # --- guided RF sampling ---
    sampled = sample_rf_guided(
        model.model,
        noise,
        init_data=None,
        steps=steps,
        device=device,
        guidance=guidance,
        **sampler_kwargs,
        **conditioning_inputs,
        **negative_conditioning_tensors,
    )

    # --- decode latents back to audio if needed ---
    if getattr(model, "pretransform", None) is not None:
        sampled = model.pretransform.decode(sampled)  # -> [B, C, audio_len]

    return sampled



def save_audio(buffer, audio_tensor, sample_rate):
    """Save audio with soundfile backend (supports BytesIO)"""
    # Convert tensor to numpy and transpose for soundfile (expects [samples, channels])
    audio_np = audio_tensor.cpu().numpy().T  # Shape: (samples, channels)
    sf.write(buffer, audio_np, sample_rate, format='WAV', subtype='PCM_16')

# Replace your existing load_model() function with this:
def get_model(model_type="standard", finetune_repo=None, finetune_checkpoint=None, base_repo=None):
    """Get model using the enhanced model manager with caching"""
    return load_model(model_type, finetune_repo, finetune_checkpoint, base_repo)

app = Flask(__name__)

# Initialize riff manager globally (add this near the top with other globals)
riff_manager = RiffManager()

# Global model storage
model_cache = {}
model_lock = threading.Lock()

# Async Jerry generation status store (for JUCE polling)
jerry_generation_jobs = {}
jerry_generation_jobs_lock = threading.Lock()
jerry_generation_worker_lock = threading.Lock()
JERRY_JOB_TTL_SECONDS = 30 * 60

_MLX_GENERATE_FN = None
_MLX_FINETUNE_LOCK = threading.Lock()
_MLX_FINETUNE_MERGED_CACHE: dict[tuple[str, str, str], dict[str, str]] = {}
_MLX_CACHE_DIR = Path(__file__).resolve().parent / ".mlx_cache"
_MLX_MERGED_CHECKPOINT_DIR = _MLX_CACHE_DIR / "merged_checkpoints"
_MLX_STATUS_LOCK = threading.Lock()
_MLX_MODEL_STATUS: dict[str, dict] = {}


@contextmanager
def resource_cleanup():
    try:
        yield
    finally:
        # Drop any big locals in the current frame from references if you want
        aggressive_cpu_cleanup()

# def load_model():
#     """Load model if not already loaded."""
#     with model_lock:
#         if 'model' not in model_cache:
#             print("🔄 Loading stable-audio-open-small model...")
            
#             # Authenticate with HF
#             hf_token = os.getenv('HF_TOKEN')
#             if hf_token:
#                 login(token=hf_token)
#                 print(f"✅ HF authenticated ({hf_token[:10]}...)")
#             else:
#                 raise ValueError("HF_TOKEN environment variable required")
            
#             # Load model
#             model, config = get_pretrained_model("stabilityai/stable-audio-open-small")
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             model = model.to(device)
#             if device == "cuda":
#                 model = model.half()
            
#             model_cache['model'] = model
#             model_cache['config'] = config
#             model_cache['device'] = device
#             print(f"✅ Model loaded on {device}")
#             print(f"   Sample rate: {config['sample_rate']}")
#             print(f"   Sample size: {config['sample_size']}")
#             print(f"   Diffusion objective: {getattr(model, 'diffusion_objective', 'unknown')}")
        
#         return model_cache['model'], model_cache['config'], model_cache['device']

def extract_bpm(prompt):
    """Extract BPM from prompt for future loop processing."""
    # Look for patterns like "120bpm", "90 bpm", "140 BPM"
    bpm_match = re.search(r'(\d+)\s*bpm', prompt.lower())
    if bpm_match:
        return int(bpm_match.group(1))
    return None

def process_input_audio(audio_file, target_sr):
    """Process uploaded audio file into tensor format."""
    try:
        # Load audio file
        if hasattr(audio_file, 'read'):
            # File-like object from Flask
            audio_bytes = audio_file.read()
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buffer)
        else:
            # File path
            waveform, sample_rate = torchaudio.load(audio_file)
        
        # Convert to mono if stereo (take average of channels)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Ensure we have stereo output (duplicate mono to stereo)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        print(f"📁 Processed input audio: {waveform.shape} at {target_sr}Hz")
        return sample_rate, waveform
    
    except Exception as e:
        raise ValueError(f"Failed to process input audio: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        backend_engine = _default_backend_engine()
        if backend_engine == "mlx":
            standard_status = _ensure_mlx_standard_status()
            return jsonify({
                "status": "healthy",
                "model_loaded": bool(standard_status and standard_status.get("warmed")),
                "device": "mlx",
                "backend_engine": "mlx",
                "cuda_available": torch.cuda.is_available(),
                "model_info": {
                    "sample_rate": int(standard_status.get("sample_rate", 44100)) if standard_status else 44100,
                    "sample_size": int(standard_status.get("sample_size", 524288)) if standard_status else 524288,
                    "diffusion_objective": "mlx-runtime"
                }
            })

        model, config, device = load_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "device": device,
            "backend_engine": "mps",
            "cuda_available": torch.cuda.is_available(),
            "model_info": {
                "sample_rate": config["sample_rate"],
                "sample_size": config["sample_size"],
                "diffusion_objective": getattr(model, 'diffusion_objective', 'unknown')
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "error": str(e)
        }), 500
    
@app.route('/models/checkpoints', methods=['POST'])
def list_checkpoints():
    """
    List available checkpoints from a Hugging Face repository
    
    JSON Body:
    {
        "finetune_repo": "thepatch/jerry_grunge"
    }
    
    Returns:
    {
        "success": true,
        "repo": "thepatch/jerry_grunge",
        "checkpoints": [
            "jerry_un-encoded_epoch=32-step=2000.ckpt",
            "jerry_un-encoded_epoch=28-step=1800.ckpt",
            ...
        ]
    }
    """
    try:
        from huggingface_hub import list_repo_files
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        finetune_repo = data.get('finetune_repo', '').strip()
        if not finetune_repo:
            return jsonify({"error": "finetune_repo is required"}), 400
        
        # List all files in the repo
        try:
            all_files = list_repo_files(repo_id=finetune_repo, repo_type="model")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Could not access repository: {str(e)}",
                "hint": "Check that the repository exists and is public"
            }), 404
        
        # Filter for .ckpt files
        checkpoints = [f for f in all_files if f.endswith('.ckpt')]
        
        if not checkpoints:
            return jsonify({
                "success": False,
                "error": "No .ckpt checkpoint files found in repository",
                "repo": finetune_repo
            }), 404
        
        # Sort checkpoints (optional - by name or try to parse epoch/step)
        checkpoints.sort()
        
        return jsonify({
            "success": True,
            "repo": finetune_repo,
            "checkpoints": checkpoints,
            "count": len(checkpoints)
        })
        
    except Exception as e:
        print(f"Checkpoint listing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route('/models/switch', methods=['POST'])
def switch_model():
    """
    Switch between standard and finetune models
    
    JSON Body:
    {
        "model_type": "standard",  // or "finetune"
        "finetune_repo": "S3Sound/am_saos1",  // required if finetune
        "finetune_checkpoint": "am_saos1_e18_s4800.ckpt"  // required if finetune
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        model_type = data.get('model_type', 'standard')
        finetune_repo = data.get('finetune_repo')
        finetune_checkpoint = data.get('finetune_checkpoint')
        
        if model_type not in ['standard', 'finetune']:
            return jsonify({"error": "model_type must be 'standard' or 'finetune'"}), 400
        
        if model_type == 'finetune':
            if not finetune_repo or not finetune_checkpoint:
                return jsonify({
                    "error": "finetune_repo and finetune_checkpoint required for finetune models"
                }), 400
        
        backend_engine, default_backend, requested_backend = _resolve_backend_engine_from_request(data)
        print(
            "🎛 Backend selection (/models/switch):"
            f" request={requested_backend!r},"
            f" default={default_backend},"
            f" effective={backend_engine}"
        )
        warmup = _coerce_bool(data.get("warmup", True), default=True)

        if backend_engine == "mlx":
            print(f"Switching to {model_type} model (mlx)...")
            model_spec = _resolve_mlx_model_spec(
                model_type=model_type,
                finetune_repo=finetune_repo,
                finetune_checkpoint=finetune_checkpoint,
                base_repo=data.get("base_repo"),
                model_id=data.get("model_id"),
            )

            if warmup:
                try:
                    _prewarm_mlx_model_spec(model_spec, model_type)
                    warmed = True
                except Exception as warmup_error:
                    warmed = False
                    print(f"⚠️ MLX warmup failed during /models/switch: {warmup_error}")
            else:
                warmed = False

            source = model_spec.get("source", {})
            _record_mlx_model_status(model_type, source, model_spec["model_config"], warmed=warmed)
            mlx_snapshot = _snapshot_mlx_model_status()
            return jsonify({
                "success": True,
                "model_type": model_type,
                "device": "mlx",
                "backend_engine": "mlx",
                "config": {
                    "sample_rate": int(model_spec["model_config"].get("sample_rate", 44100)),
                    "sample_size": int(model_spec["model_config"].get("sample_size", 524288))
                },
                "cache_status": {
                    "loaded_models": list(mlx_snapshot.keys()),
                    "usage_order": list(mlx_snapshot.keys()),
                    "cache_utilization": f"{len(mlx_snapshot)}/mlx-runtime",
                },
                "message": (
                    f"Successfully switched to {model_type} model (mlx)"
                    + (" and warmed caches" if warmed else "")
                )
            })

        # Load the requested model (this will cache it)
        print(f"Switching to {model_type} model (mps)...")
        model, config, device = get_model(model_type, finetune_repo, finetune_checkpoint)

        # Get cache status
        cache_info = model_manager.list_loaded_models()

        return jsonify({
            "success": True,
            "model_type": model_type,
            "device": device,
            "backend_engine": "mps",
            "config": {
                "sample_rate": config["sample_rate"],
                "sample_size": config["sample_size"]
            },
            "cache_status": cache_info,
            "message": f"Successfully switched to {model_type} model"
        })
        
    except Exception as e:
        print(f"Model switch error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/models/status', methods=['GET'])
def models_status():
    """Get status of loaded models"""
    try:
        if _default_backend_engine() == "mlx":
            _ensure_mlx_standard_status()
            mlx_snapshot = _snapshot_mlx_model_status()
            usage_order = list(mlx_snapshot.keys())
            return jsonify({
                "cache_status": {
                    "loaded_models": usage_order,
                    "usage_order": usage_order,
                    "cache_utilization": f"{len(usage_order)}/mlx-runtime",
                },
                "model_details": mlx_snapshot,
                "max_models": "mlx-runtime",
                "backend_engine": "mlx",
                "cuda_available": torch.cuda.is_available()
            })

        cache_info = model_manager.list_loaded_models()
        
        # Get detailed info about each loaded model
        detailed_info = {}
        for model_key in cache_info['loaded_models']:
            if model_key in model_manager.model_cache:
                model_data = model_manager.model_cache[model_key]
                detailed_info[model_key] = {
                    "type": model_data["type"],
                    "source": model_data["source"],
                    "device": model_data["device"],
                    "sample_rate": model_data["config"]["sample_rate"],
                    "sample_size": model_data["config"]["sample_size"]
                }
        
        return jsonify({
            "cache_status": cache_info,
            "model_details": detailed_info,
            "max_models": model_manager.max_models,
            "backend_engine": "mps",
            "cuda_available": torch.cuda.is_available()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/models/prompts', methods=['GET', 'POST'])
def get_model_prompts():
    try:
        cache = model_manager.list_loaded_models()
        usage_order = cache.get("usage_order", [])
        # You may need direct access; if so, read model_manager.model_cache instead of list_loaded_models
        model_cache = getattr(model_manager, "model_cache", {})

        if request.method == "POST":
            payload = request.get_json(silent=True) or {}
            key = payload.get("key")
            # Support both repo/checkpoint and finetune_repo/finetune_checkpoint
            repo = payload.get("repo") or payload.get("finetune_repo")
            checkpoint = payload.get("checkpoint") or payload.get("finetune_checkpoint")
            prefer = (payload.get("prefer") or "active").lower()
        else:
            # --- selection inputs ---
            key = request.args.get("key")
            # Support both repo/checkpoint and finetune_repo/finetune_checkpoint
            repo = request.args.get("repo") or request.args.get("finetune_repo")
            checkpoint = request.args.get("checkpoint") or request.args.get("finetune_checkpoint")
            prefer = (request.args.get("prefer") or "active").lower()  # active | finetune | recent

        selected_key = None
        direct_prompts = None
        print(f"/models/prompts hit: method={request.method} repo={repo} checkpoint={checkpoint} prefer={prefer} key={key}")

        # Helper to resolve a cache_key by repo+checkpoint substrings
        def find_key_by_repo_ckpt(r, ck):
            for k in reversed(usage_order):  # prefer most recent
                entry = model_cache.get(k, {})
                if (r and r in str(entry.get("repo", ""))) and (ck and ck in str(entry.get("checkpoint", ""))):
                    return k
            return None

        # 1) explicit key
        if key and key in model_cache:
            selected_key = key
        # 2) repo+checkpoint
        elif repo and checkpoint:
            selected_key = find_key_by_repo_ckpt(repo, checkpoint)
        elif repo and not checkpoint:
            # Prefer most recent matching repo; if prefer=finetune, require type=finetune
            for k in reversed(usage_order):
                entry = model_cache.get(k, {})
                if repo in str(entry.get("repo", "")):
                    if prefer == "finetune" and entry.get("type") != "finetune":
                        continue
                    selected_key = k
                    break
        else:
            # 3) prefer active
            if prefer == "active":
                active_key = getattr(model_manager, "active_model_key", None)
                if active_key in model_cache:
                    selected_key = active_key
            # 4) prefer finetune
            if not selected_key and prefer in ("finetune",):
                for k in reversed(usage_order):
                    if model_cache.get(k, {}).get("type") == "finetune":
                        selected_key = k
                        break
            # 5) fallback to most recent
            if not selected_key and usage_order:
                selected_key = usage_order[-1]

        if not selected_key or selected_key not in model_cache:
            # If caller provided a repo but it's not in cache, fetch prompts directly from HF.
            if repo:
                try:
                    direct_prompts = model_manager._fetch_prompts_for_repo(
                        repo_id=repo,
                        checkpoint=checkpoint
                    )
                except Exception:
                    direct_prompts = None

            if direct_prompts:
                return jsonify({
                    "success": True,
                    "model_key": None,
                    "type": "finetune",
                    "source": repo,
                    "checkpoint": checkpoint,
                    "prompts": direct_prompts,
                }), 200

            return jsonify({
                "success": True,
                "prompts": None,
                "message": "no model selected"
            }), 200

        entry = model_cache[selected_key]
        prompts = entry.get("prompts")

        # Lazy load prompts.json if this is a finetune and prompts missing
        try:
            if entry.get("type") == "finetune" and not prompts:
                # You added this helper earlier; falls back to None on failure
                prompts = model_manager._fetch_prompts_for_repo(
                    repo_id=entry.get("repo"),
                    checkpoint=entry.get("checkpoint")
                )
                entry["prompts"] = prompts
        except Exception:
            pass  # non-fatal

        # Normalize payload so JUCE can rely on consistent shape
        if not prompts:
            prompts = {"version": 1, "dice": {"generic": [], "drums": [], "instrumental": []}}

        payload = {
            "success": True,
            "model_key": selected_key,
            "type": entry.get("type"),
            "source": entry.get("repo") or entry.get("source"),
            "checkpoint": entry.get("checkpoint"),
            "prompts": prompts,
        }
        return jsonify(payload), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/models/clear-cache', methods=['POST'])
def clear_model_cache():
    """Clear all cached models"""
    try:
        model_manager.clear_cache()
        return jsonify({
            "success": True,
            "message": "Model cache cleared successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
# Test endpoint for the finetune
@app.route('/test/finetune', methods=['GET'])
def test_finetune():
    """Test endpoint to verify finetune loading"""
    try:
        print("Testing finetune loading...")
        
        # Try to load the finetune
        model, config, device = get_model(
            model_type="finetune",
            finetune_repo="S3Sound/am_saos1",
            finetune_checkpoint="am_saos1_e18_s4800.ckpt"
        )
        
        cache_info = model_manager.list_loaded_models()
        
        return jsonify({
            "success": True,
            "message": "Finetune loaded successfully",
            "device": device,
            "config": {
                "sample_rate": config["sample_rate"],
                "sample_size": config["sample_size"]
            },
            "cache_status": cache_info
        })
        
    except Exception as e:
        print(f"Finetune test failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    

    
def detect_model_family(config):
    """Detect if this is SAO 1.0 vs SAOS"""
    # SAO 1.0 has seconds_start in conditioning
    cond_configs = config.get("model", {}).get("conditioning", {}).get("configs", [])
    has_seconds_start = any(c.get("id") == "seconds_start" for c in cond_configs)
    
    # Also check sample_size (SAOS max is 524288, SAO 1.0 finetunes are higher)
    sample_size = config.get("sample_size", 0)
    is_long_form = sample_size > 524288  # ← Changed from 1000000
    
    if has_seconds_start or is_long_form:
        return "sao1.0"
    return "saos"

def sampler_kwargs_for_objective(model, config, client_overrides=None):
    """Choose sampler kwargs based on model family and objective"""
    obj = getattr(model, "diffusion_objective", None)
    model_family = detect_model_family(config)
    kw = {}
    
    if client_overrides:
        kw.update(client_overrides)
    
    # SAO 1.0 specific parameters
    if model_family == "sao1.0":
        kw.setdefault("sampler_type", "dpmpp-3m-sde")
        kw.setdefault("sigma_min", 0.3)
        kw.setdefault("sigma_max", 500)
        return kw
    
    # SAOS parameters (your existing logic)
    if obj in ("rf_denoiser", "rectified_flow"):
        # For rf_* objectives, pingpong is the default for the post-adversarial rf_denoiser.
        # For rectified_flow, safest is to omit sampler_type and let stable-audio-tools choose defaults.
        if obj == "rf_denoiser":
            kw.setdefault("sampler_type", "pingpong")
        else:
            # Ensure we DON'T force pingpong on rectified_flow
            kw.pop("sampler_type", None)
    elif obj == "v":
        # v-objective typically uses k-diffusion samplers chosen inside generate_diffusion_cond
        kw.pop("sampler_type", None)
    else:
        # Unknown—be conservative
        kw.pop("sampler_type", None)

    return kw

def _report_generation_progress(progress_callback, progress, stage):
    if progress_callback is None:
        return
    try:
        progress_callback(max(0, min(100, int(progress))), stage)
    except Exception:
        pass

def _set_jerry_job_state(request_id, **updates):
    now = time.time()
    with jerry_generation_jobs_lock:
        state = dict(jerry_generation_jobs.get(request_id, {}))
        state.update(updates)
        state["updated_at"] = now
        jerry_generation_jobs[request_id] = state

def _get_jerry_job_state(request_id):
    with jerry_generation_jobs_lock:
        state = jerry_generation_jobs.get(request_id)
        return dict(state) if state is not None else None

def _cleanup_jerry_jobs():
    now = time.time()
    with jerry_generation_jobs_lock:
        stale_ids = []
        for request_id, state in jerry_generation_jobs.items():
            updated_at = float(state.get("updated_at", now))
            status = state.get("status", "")
            if status in ("completed", "failed"):
                if now - updated_at > JERRY_JOB_TTL_SECONDS:
                    stale_ids.append(request_id)
            elif now - updated_at > (2 * JERRY_JOB_TTL_SECONDS):
                stale_ids.append(request_id)

        for request_id in stale_ids:
            jerry_generation_jobs.pop(request_id, None)


def _normalize_backend_engine(raw_backend):
    if raw_backend is None:
        return "mps"
    backend = str(raw_backend).strip().lower()
    if backend in ("", "mps", "torch"):
        return "mps"
    if backend == "mlx":
        return "mlx"
    raise ValueError("backend_engine must be 'mps' or 'mlx'")


def _default_backend_engine():
    configured = os.getenv("STABLE_AUDIO_BACKEND_ENGINE", "mps")
    try:
        return _normalize_backend_engine(configured)
    except Exception:
        print(
            f"⚠️ Invalid STABLE_AUDIO_BACKEND_ENGINE='{configured}', falling back to 'mps'"
        )
        return "mps"


def _resolve_backend_engine_from_request(data):
    default_backend = _default_backend_engine()
    requested_backend = None

    if isinstance(data, dict) and "backend_engine" in data:
        requested_backend = data.get("backend_engine")
        if requested_backend is None:
            return default_backend, default_backend, requested_backend
        if isinstance(requested_backend, str) and requested_backend.strip() == "":
            return default_backend, default_backend, requested_backend
        return _normalize_backend_engine(requested_backend), default_backend, requested_backend

    return default_backend, default_backend, requested_backend


def _coerce_bool(value, default=True):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on"):
            return True
        if normalized in ("0", "false", "no", "n", "off"):
            return False
    return default


def _get_mlx_generate_fn():
    global _MLX_GENERATE_FN
    if _MLX_GENERATE_FN is not None:
        return _MLX_GENERATE_FN

    try:
        from saomlx.pipeline import generate_diffusion_cond_mlx
    except Exception as exc:
        raise RuntimeError(
            "MLX backend is unavailable. Ensure dependencies are installed and saomlx is importable."
        ) from exc

    _MLX_GENERATE_FN = generate_diffusion_cond_mlx
    return _MLX_GENERATE_FN


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _torch_load_any(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_mlx_finetune_assets(finetune_repo, finetune_checkpoint, base_repo):
    cache_key = (finetune_repo, finetune_checkpoint, base_repo)

    with _MLX_FINETUNE_LOCK:
        cached = _MLX_FINETUNE_MERGED_CACHE.get(cache_key)
        if cached:
            cached_ckpt = cached.get("merged_ckpt_path")
            cached_config = cached.get("config_path")
            if cached_ckpt and cached_config and os.path.exists(cached_ckpt) and os.path.exists(cached_config):
                print(
                    f"🎯 Using cached MLX finetune assets: {finetune_repo}/{finetune_checkpoint}"
                )
                model_config = _load_json(cached_config)
                return {
                    "model_id": None,
                    "model_config_path": cached_config,
                    "model_ckpt_path": cached_ckpt,
                    "model_config": model_config,
                    "source": {
                        "type": "finetune",
                        "repo": finetune_repo,
                        "checkpoint": finetune_checkpoint,
                        "base_repo": base_repo,
                    },
                }

        os.makedirs(_MLX_MERGED_CHECKPOINT_DIR, exist_ok=True)
        digest_input = f"{finetune_repo}|{finetune_checkpoint}|{base_repo}"
        digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:24]
        merged_ckpt_path = str((_MLX_MERGED_CHECKPOINT_DIR / f"{digest}.ckpt").resolve())

        config_path = model_manager._detect_config_from_checkpoint(finetune_checkpoint, finetune_repo)
        if config_path is None:
            config_path = hf_hub_download(repo_id=base_repo, filename="base_model_config.json", repo_type="model")

        base_ckpt_path = hf_hub_download(repo_id=base_repo, filename="base_model.ckpt", repo_type="model")
        finetune_ckpt_path = hf_hub_download(repo_id=finetune_repo, filename=finetune_checkpoint, repo_type="model")

        if not os.path.exists(merged_ckpt_path):
            print(f"🔧 Preparing merged MLX finetune checkpoint: {finetune_repo}/{finetune_checkpoint}")

            base_state_dict = load_ckpt_state_dict(base_ckpt_path)
            finetune_payload = _torch_load_any(finetune_ckpt_path)
            finetune_state_dict = finetune_payload.get("state_dict", finetune_payload)
            if not isinstance(finetune_state_dict, dict):
                raise ValueError("Unsupported finetune checkpoint format: expected state_dict dictionary")

            ema_prefix = "diffusion_ema.ema_model."
            ema_keys = [k for k in finetune_state_dict.keys() if str(k).startswith(ema_prefix)]

            if ema_keys:
                overlay = {}
                for key in ema_keys:
                    mapped_key = str(key).replace(ema_prefix, "diffusion.model.")
                    overlay[mapped_key] = finetune_state_dict[key]
                for key, value in finetune_state_dict.items():
                    if str(key).startswith("diffusion.pretransform"):
                        overlay[key] = value
            else:
                overlay = {
                    key: value
                    for key, value in finetune_state_dict.items()
                    if not str(key).startswith("pretransform.")
                }

            merged_state_dict = dict(base_state_dict)
            merged_state_dict.update(overlay)
            torch.save({"state_dict": merged_state_dict}, merged_ckpt_path)
        else:
            print(
                f"🎯 Reusing merged MLX finetune checkpoint: {finetune_repo}/{finetune_checkpoint}"
            )

        _MLX_FINETUNE_MERGED_CACHE[cache_key] = {
            "config_path": str(config_path),
            "merged_ckpt_path": merged_ckpt_path,
        }

        model_config = _load_json(config_path)
        return {
            "model_id": None,
            "model_config_path": str(config_path),
            "model_ckpt_path": merged_ckpt_path,
            "model_config": model_config,
            "source": {
                "type": "finetune",
                "repo": finetune_repo,
                "checkpoint": finetune_checkpoint,
                "base_repo": base_repo,
            },
        }


def _resolve_mlx_model_spec(model_type, finetune_repo, finetune_checkpoint, base_repo=None, model_id=None):
    if model_id:
        config_path = hf_hub_download(repo_id=model_id, filename="model_config.json", repo_type="model")
        return {
            "model_id": model_id,
            "model_config_path": str(config_path),
            "model_ckpt_path": None,
            "model_config": _load_json(config_path),
            "source": {"type": "pretrained", "model_id": model_id},
        }

    if model_type == "standard":
        pretrained_id = "stabilityai/stable-audio-open-small"
        config_path = hf_hub_download(repo_id=pretrained_id, filename="model_config.json", repo_type="model")
        return {
            "model_id": pretrained_id,
            "model_config_path": str(config_path),
            "model_ckpt_path": None,
            "model_config": _load_json(config_path),
            "source": {"type": "pretrained", "model_id": pretrained_id},
        }

    if model_type == "finetune":
        if not finetune_repo or not finetune_checkpoint:
            raise ValueError("finetune_repo and finetune_checkpoint required for finetune models")
        resolved_base_repo = base_repo or "stabilityai/stable-audio-open-small"
        return _resolve_mlx_finetune_assets(finetune_repo, finetune_checkpoint, resolved_base_repo)

    raise ValueError("model_type must be 'standard' or 'finetune'")


def _mlx_cache_key(model_type, source):
    if model_type == "standard":
        return "standard_saos"
    repo = str(source.get("repo", "")).replace("/", "_")
    checkpoint = str(source.get("checkpoint", "")).replace("/", "_").replace(".", "_")
    return f"finetune_{repo}_{checkpoint}"


def _record_mlx_model_status(model_type, source, config, *, warmed=False):
    key = _mlx_cache_key(model_type, source)
    entry = {
        "type": model_type,
        "source": source.get("model_id") if model_type == "standard" else f"{source.get('repo', '')}/{source.get('checkpoint', '')}",
        "device": "mlx",
        "sample_rate": int(config.get("sample_rate", 44100)),
        "sample_size": int(config.get("sample_size", 524288)),
        "warmed": bool(warmed),
    }
    with _MLX_STATUS_LOCK:
        previous = _MLX_MODEL_STATUS.get(key, {})
        if previous.get("warmed"):
            entry["warmed"] = True
        _MLX_MODEL_STATUS[key] = entry
    return key


def _snapshot_mlx_model_status():
    with _MLX_STATUS_LOCK:
        return dict(_MLX_MODEL_STATUS)


def _ensure_mlx_standard_status():
    snapshot = _snapshot_mlx_model_status()
    if "standard_saos" in snapshot:
        return snapshot["standard_saos"]

    spec = _resolve_mlx_model_spec("standard", None, None)
    source = spec.get("source", {"type": "pretrained", "model_id": "stabilityai/stable-audio-open-small"})
    _record_mlx_model_status("standard", source, spec["model_config"], warmed=False)
    snapshot = _snapshot_mlx_model_status()
    return snapshot.get("standard_saos")


def _prewarm_mlx_model_spec(spec, model_type):
    generate_diffusion_cond_mlx = _get_mlx_generate_fn()
    config = spec["model_config"]
    model_family = detect_model_family(config)
    extra_cond = {}
    if model_family == "sao1.0":
        extra_cond["seconds_start"] = 0.0

    print(f"🔥 MLX warmup start ({model_type})...")
    generate_diffusion_cond_mlx(
        model_id=spec["model_id"],
        model_config_path=spec["model_config_path"],
        model_ckpt_path=spec["model_ckpt_path"],
        prompt="warmup tone",
        negative_prompt="",
        seed=123,
        steps=1,
        seconds=1.0,
        cfg_scale=1.0,
        conditioning_backend="torch",
        out_dir=None,
        extra_cond=extra_cond,
    )
    print(f"✅ MLX warmup complete ({model_type})")


def _encode_audio_np(audio_tc, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, audio_tc, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = buffer.getvalue()
    audio_b64 = base64.b64encode(wav_bytes).decode()
    return wav_bytes, audio_b64


def _run_generate_request_mlx(data, progress_callback=None):
    if not isinstance(data, dict):
        raise ValueError("JSON body required")

    _report_generation_progress(progress_callback, 2, "validating")

    model_type = data.get("model_type", "standard")
    finetune_repo = data.get("finetune_repo")
    finetune_checkpoint = data.get("finetune_checkpoint")
    base_repo = data.get("base_repo")
    model_id = data.get("model_id")

    prompt = data.get("prompt")
    if not prompt:
        raise ValueError("prompt is required")

    _report_generation_progress(progress_callback, 10, "loading_model")
    mlx_model_spec = _resolve_mlx_model_spec(
        model_type=model_type,
        finetune_repo=finetune_repo,
        finetune_checkpoint=finetune_checkpoint,
        base_repo=base_repo,
        model_id=model_id,
    )
    config = mlx_model_spec["model_config"]
    model_family = detect_model_family(config)
    _record_mlx_model_status(model_type, mlx_model_spec.get("source", {}), config, warmed=True)
    print("🟢 Backend engine: mlx")
    print(f"   Model type: {model_type}")
    print(f"   Model source: {mlx_model_spec.get('source')}")

    sample_rate = int(config.get("sample_rate", 44100))
    model_sample_size = int(config.get("sample_size", 524288))
    model_seconds_max = max(1, model_sample_size // sample_rate)

    req_seconds_total = data.get("seconds_total")
    if req_seconds_total is None:
        seconds_total = model_seconds_max
    else:
        try:
            seconds_total = int(req_seconds_total)
        except Exception as exc:
            raise ValueError("seconds_total must be an integer number of seconds") from exc
        if seconds_total < 1:
            raise ValueError("seconds_total must be >= 1")
        if seconds_total > model_seconds_max:
            seconds_total = model_seconds_max

    steps = data.get("steps")
    if steps is None:
        steps = 100 if model_family == "sao1.0" else 8
    if not isinstance(steps, int) or steps < 1 or steps > 250:
        raise ValueError("steps must be integer between 1-250")

    cfg_default = 7.0 if model_family == "sao1.0" else 1.0
    cfg_scale = data.get("cfg_scale", cfg_default)
    if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0 or cfg_scale > 20:
        raise ValueError("cfg_scale must be number between 0-20")

    negative_prompt = data.get("negative_prompt", "")
    seed = data.get("seed", -1)
    try:
        seed = int(seed)
    except Exception as exc:
        raise ValueError("seed must be an integer") from exc

    sampler_type = data.get("sampler_type")
    if not sampler_type and model_family == "sao1.0":
        sampler_type = "k-heun"

    mlx_conditioning_backend = str(data.get("mlx_conditioning_backend", "torch")).strip().lower()
    if mlx_conditioning_backend not in ("torch", "mlx"):
        raise ValueError("mlx_conditioning_backend must be 'torch' or 'mlx'")

    _report_generation_progress(progress_callback, 25, "preparing_conditioning")
    sampling_progress_start = 22
    sampling_progress_end = 92
    sampling_state = {
        "last_progress": sampling_progress_start - 1,
        "last_completed_step": 0,
        "saw_zero_index": False,
    }
    if progress_callback is not None:
        _report_generation_progress(progress_callback, sampling_progress_start, "sampling")

    def _mlx_sampling_step_callback(payload):
        if progress_callback is None or not isinstance(payload, dict):
            return

        raw_index = payload.get("i")
        if not isinstance(raw_index, (int, float)):
            return

        step_index = int(raw_index)
        if step_index < 0:
            return

        if step_index == 0:
            sampling_state["saw_zero_index"] = True

        if sampling_state["saw_zero_index"]:
            completed_step = step_index + 1
        else:
            cand_from_zero_based = max(1, step_index + 1)
            cand_from_one_based = max(1, step_index)
            prev_completed = sampling_state["last_completed_step"]
            viable = [c for c in (cand_from_zero_based, cand_from_one_based) if c >= prev_completed]
            completed_step = min(viable) if viable else max(cand_from_zero_based, cand_from_one_based, prev_completed)

        completed_step = max(1, min(steps, completed_step))
        sampling_state["last_completed_step"] = completed_step

        fraction = completed_step / float(max(steps, 1))
        progress = sampling_progress_start + int(round((sampling_progress_end - sampling_progress_start) * fraction))
        progress = max(sampling_progress_start, min(sampling_progress_end, progress))

        if progress > sampling_state["last_progress"]:
            sampling_state["last_progress"] = progress
            _report_generation_progress(progress_callback, progress, "sampling")

    extra_cond = {}
    if model_family == "sao1.0":
        try:
            extra_cond["seconds_start"] = float(data.get("seconds_start", 0))
        except Exception:
            extra_cond["seconds_start"] = 0.0

    _report_generation_progress(progress_callback, 35, "configuring_sampler")
    generate_diffusion_cond_mlx = _get_mlx_generate_fn()
    result = generate_diffusion_cond_mlx(
        model_id=mlx_model_spec["model_id"],
        model_config_path=mlx_model_spec["model_config_path"],
        model_ckpt_path=mlx_model_spec["model_ckpt_path"],
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        seconds=float(seconds_total),
        cfg_scale=float(cfg_scale),
        sampler_type=sampler_type,
        conditioning_backend=mlx_conditioning_backend,
        out_dir=None,
        extra_cond=extra_cond,
        step_callback=_mlx_sampling_step_callback,
    )

    generation_time = float(result.get("stats", {}).get("timings_sec", {}).get("total", 0.0))
    cache_info = result.get("stats", {}).get("cache", {})
    if cache_info:
        print(
            "   MLX cache hits:"
            f" torch_model={cache_info.get('torch_model_cache_hit')},"
            f" conditioner={cache_info.get('conditioner_cache_hit')},"
            f" converted={cache_info.get('converted_model_cache_hit')}"
        )
    print(f"   MLX conditioning backend: {mlx_conditioning_backend}")

    _report_generation_progress(progress_callback, 95, "postprocessing")
    output = np.asarray(result["audio"][0], dtype=np.float32)  # [T, C]
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0:
        output = output / peak
    output = np.clip(output, -1.0, 1.0)

    requested_samples = seconds_total * sample_rate
    if output.shape[0] > requested_samples:
        output = output[:requested_samples, :]
        print(f"   ✂️  Trimmed from {result['audio'][0].shape[0]/sample_rate:.1f}s to {seconds_total}s")

    _report_generation_progress(progress_callback, 98, "encoding_audio")
    wav_bytes, audio_b64 = _encode_audio_np(output, sample_rate)

    detected_bpm = extract_bpm(prompt)
    sampler_used = result.get("stats", {}).get("sampler_type", sampler_type)
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sample_rate": sample_rate,
        "duration_seconds": seconds_total,
        "generation_time": round(generation_time, 2),
        "wall_time_sec": round(generation_time, 2),
        "realtime_factor": round(seconds_total / max(generation_time, 1e-6), 2),
        "detected_bpm": detected_bpm,
        "device": "mlx",
        "backend_engine": "mlx",
        "sampler_type": sampler_used,
        "model_type": model_type,
    }

    print(f"✅ MLX generated in {generation_time:.2f}s ({metadata['realtime_factor']:.1f}x RT)")
    _report_generation_progress(progress_callback, 100, "completed")
    return {
        "audio_base64": audio_b64,
        "wav_bytes": wav_bytes,
        "metadata": metadata,
    }


def _run_generate_loop_request_mlx(data, progress_callback=None):
    if not isinstance(data, dict):
        raise ValueError("JSON body required")

    _report_generation_progress(progress_callback, 2, "validating")

    model_type = data.get("model_type", "standard")
    finetune_repo = data.get("finetune_repo")
    finetune_checkpoint = data.get("finetune_checkpoint")
    base_repo = data.get("base_repo")
    model_id = data.get("model_id")

    prompt = data.get("prompt")
    if not prompt:
        raise ValueError("prompt is required")

    loop_type = str(data.get("loop_type", "auto")).strip().lower()
    if loop_type not in ("auto", "drums", "instruments"):
        raise ValueError("loop_type must be auto, drums, or instruments")

    bars = data.get("bars")
    steps = data.get("steps", 8)
    cfg_scale = data.get("cfg_scale", 6.0)
    seed = data.get("seed", -1)

    try:
        steps = int(steps)
    except Exception as exc:
        raise ValueError("steps must be an integer") from exc
    if steps < 1 or steps > 250:
        raise ValueError("steps must be integer between 1-250")

    try:
        cfg_scale = float(cfg_scale)
    except Exception as exc:
        raise ValueError("cfg_scale must be a number") from exc
    if cfg_scale < 0 or cfg_scale > 20:
        raise ValueError("cfg_scale must be number between 0-20")

    try:
        seed = int(seed)
    except Exception as exc:
        raise ValueError("seed must be an integer") from exc

    detected_bpm = extract_bpm(prompt)
    if not detected_bpm:
        raise ValueError("BPM must be specified in prompt (e.g., '120bpm')")

    _report_generation_progress(progress_callback, 10, "loading_model")
    mlx_model_spec = _resolve_mlx_model_spec(
        model_type=model_type,
        finetune_repo=finetune_repo,
        finetune_checkpoint=finetune_checkpoint,
        base_repo=base_repo,
        model_id=model_id,
    )
    config = mlx_model_spec["model_config"]
    model_family = detect_model_family(config)
    _record_mlx_model_status(model_type, mlx_model_spec.get("source", {}), config, warmed=True)
    print("🟢 Backend engine: mlx (loop)")
    print(f"   Model type: {model_type}")
    print(f"   Model source: {mlx_model_spec.get('source')}")

    sample_rate = int(config.get("sample_rate", 44100))
    model_sample_size = int(config.get("sample_size", 524288))
    max_duration = model_sample_size / sample_rate

    if bars is not None and str(bars).strip() != "" and str(bars).lower() != "auto":
        try:
            bars = int(bars)
        except Exception as exc:
            raise ValueError("bars must be 1, 2, 4, or 8") from exc
    else:
        seconds_per_beat = 60.0 / detected_bpm
        seconds_per_bar = seconds_per_beat * 4
        max_loop_duration = max_duration - 1.0
        possible_bars = [8, 4, 2, 1]
        bars = 1
        for bar_count in possible_bars:
            loop_duration = seconds_per_bar * bar_count
            if loop_duration <= max_loop_duration:
                bars = bar_count
                break

    if bars not in [1, 2, 4, 8]:
        raise ValueError("bars must be 1, 2, 4, or 8")

    seconds_per_beat = 60.0 / detected_bpm
    seconds_per_bar = seconds_per_beat * 4
    calculated_loop_duration = seconds_per_bar * bars
    if calculated_loop_duration > max_duration:
        if calculated_loop_duration > (max_duration + 1.0):
            bars = max(1, bars // 2)
            calculated_loop_duration = seconds_per_bar * bars

    enhanced_prompt = prompt
    negative_prompt = ""
    if loop_type == "drums":
        if "drum" not in prompt.lower():
            enhanced_prompt = f"{prompt} drum loop"
        negative_prompt = "melody, harmony, pitched instruments, vocals, singing"
    elif loop_type == "instruments":
        if "drum" in prompt.lower():
            enhanced_prompt = prompt.replace("drum", "").replace("drums", "").strip()
        negative_prompt = "drums, percussion, kick, snare, hi-hat"

    seconds_total = max(1.0, min(max_duration, float(calculated_loop_duration)))
    sampler_type = data.get("sampler_type")
    if not sampler_type and model_family == "sao1.0":
        sampler_type = "k-heun"

    mlx_conditioning_backend = str(data.get("mlx_conditioning_backend", "torch")).strip().lower()
    if mlx_conditioning_backend not in ("torch", "mlx"):
        raise ValueError("mlx_conditioning_backend must be 'torch' or 'mlx'")

    sampling_progress_start = 22
    sampling_progress_end = 92
    sampling_state = {
        "last_progress": sampling_progress_start - 1,
        "last_completed_step": 0,
        "saw_zero_index": False,
    }
    if progress_callback is not None:
        _report_generation_progress(progress_callback, sampling_progress_start, "sampling")

    def _mlx_sampling_step_callback(payload):
        if progress_callback is None or not isinstance(payload, dict):
            return
        raw_index = payload.get("i")
        if not isinstance(raw_index, (int, float)):
            return
        step_index = int(raw_index)
        if step_index < 0:
            return
        if step_index == 0:
            sampling_state["saw_zero_index"] = True
        if sampling_state["saw_zero_index"]:
            completed_step = step_index + 1
        else:
            cand_from_zero_based = max(1, step_index + 1)
            cand_from_one_based = max(1, step_index)
            prev_completed = sampling_state["last_completed_step"]
            viable = [c for c in (cand_from_zero_based, cand_from_one_based) if c >= prev_completed]
            completed_step = min(viable) if viable else max(cand_from_zero_based, cand_from_one_based, prev_completed)
        completed_step = max(1, min(steps, completed_step))
        sampling_state["last_completed_step"] = completed_step
        fraction = completed_step / float(max(steps, 1))
        progress = sampling_progress_start + int(round((sampling_progress_end - sampling_progress_start) * fraction))
        progress = max(sampling_progress_start, min(sampling_progress_end, progress))
        if progress > sampling_state["last_progress"]:
            sampling_state["last_progress"] = progress
            _report_generation_progress(progress_callback, progress, "sampling")

    extra_cond = {}
    if model_family == "sao1.0":
        try:
            extra_cond["seconds_start"] = float(data.get("seconds_start", 0))
        except Exception:
            extra_cond["seconds_start"] = 0.0

    _report_generation_progress(progress_callback, 35, "configuring_sampler")
    generate_diffusion_cond_mlx = _get_mlx_generate_fn()
    result = generate_diffusion_cond_mlx(
        model_id=mlx_model_spec["model_id"],
        model_config_path=mlx_model_spec["model_config_path"],
        model_ckpt_path=mlx_model_spec["model_ckpt_path"],
        prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        seconds=float(seconds_total),
        cfg_scale=float(cfg_scale),
        sampler_type=sampler_type,
        conditioning_backend=mlx_conditioning_backend,
        out_dir=None,
        extra_cond=extra_cond,
        step_callback=_mlx_sampling_step_callback,
    )

    generation_time = float(result.get("stats", {}).get("timings_sec", {}).get("total", 0.0))
    cache_info = result.get("stats", {}).get("cache", {})
    if cache_info:
        print(
            "   MLX cache hits:"
            f" torch_model={cache_info.get('torch_model_cache_hit')},"
            f" conditioner={cache_info.get('conditioner_cache_hit')},"
            f" converted={cache_info.get('converted_model_cache_hit')}"
        )
    print(f"   MLX conditioning backend: {mlx_conditioning_backend}")

    _report_generation_progress(progress_callback, 95, "postprocessing")
    output = np.asarray(result["audio"][0], dtype=np.float32)  # [T, C]
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0:
        output = output / peak
    output = np.clip(output, -1.0, 1.0)

    loop_samples = int(calculated_loop_duration * sample_rate)
    available_samples = output.shape[0]
    available_duration = available_samples / sample_rate
    loop_duration = calculated_loop_duration
    if loop_samples > available_samples:
        loop_samples = available_samples
        loop_duration = available_duration

    loop_output = output[:loop_samples, :]
    _report_generation_progress(progress_callback, 98, "encoding_audio")
    wav_bytes, audio_b64 = _encode_audio_np(loop_output, sample_rate)

    sampler_used = result.get("stats", {}).get("sampler_type", sampler_type)
    metadata = {
        "prompt": enhanced_prompt,
        "original_prompt": prompt,
        "negative_prompt": negative_prompt,
        "loop_type": loop_type,
        "detected_bpm": detected_bpm,
        "bars": bars,
        "loop_duration_seconds": round(loop_duration, 2),
        "calculated_duration_seconds": round(calculated_loop_duration, 2),
        "available_audio_seconds": round(available_duration, 2),
        "seconds_per_bar": round(seconds_per_bar, 2),
        "style_transfer": False,
        "style_strength": None,
        "model_type": model_type,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sample_rate": sample_rate,
        "generation_time": round(generation_time, 2),
        "wall_time_sec": round(generation_time, 2),
        "device": "mlx",
        "backend_engine": "mlx",
        "sampler_type": sampler_used,
    }

    _report_generation_progress(progress_callback, 100, "completed")
    return {
        "audio_base64": audio_b64,
        "wav_bytes": wav_bytes,
        "metadata": metadata,
    }

def _run_generate_request(data, progress_callback=None):
    if not isinstance(data, dict):
        raise ValueError("JSON body required")

    _report_generation_progress(progress_callback, 2, "validating")
    backend_engine, default_backend_engine, requested_backend_engine = _resolve_backend_engine_from_request(data)
    print(
        "🎛 Backend selection (/generate):"
        f" request={requested_backend_engine!r},"
        f" default={default_backend_engine},"
        f" effective={backend_engine}"
    )
    if backend_engine == "mlx":
        return _run_generate_request_mlx(data, progress_callback=progress_callback)
    print("🟠 Backend engine: mps")

    model_type = data.get("model_type", "standard")
    finetune_repo = data.get("finetune_repo")
    finetune_checkpoint = data.get("finetune_checkpoint")

    prompt = data.get("prompt")
    if not prompt:
        raise ValueError("prompt is required")

    _report_generation_progress(progress_callback, 10, "loading_model")
    model, config, device = get_model(model_type, finetune_repo, finetune_checkpoint)

    sample_rate = int(config.get("sample_rate", 44100))
    model_sample_size = int(config.get("sample_size", 524288))
    model_seconds_max = max(1, model_sample_size // sample_rate)
    model_family = detect_model_family(config)

    req_seconds_total = data.get("seconds_total")
    if req_seconds_total is None:
        seconds_total = model_seconds_max
    else:
        try:
            seconds_total = int(req_seconds_total)
        except Exception as exc:
            raise ValueError("seconds_total must be an integer number of seconds") from exc
        if seconds_total < 1:
            raise ValueError("seconds_total must be >= 1")
        if seconds_total > model_seconds_max:
            seconds_total = model_seconds_max

    diffusion_objective = getattr(model, "diffusion_objective", None)
    steps = data.get("steps")
    if steps is None:
        if model_family == "sao1.0":
            steps = 100
        elif diffusion_objective == "rectified_flow":
            steps = 50
        else:
            steps = 8
    if not isinstance(steps, int) or steps < 1 or steps > 250:
        raise ValueError("steps must be integer between 1-250")

    cfg_scale = data.get("cfg_scale", 1.0)
    if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0 or cfg_scale > 20:
        raise ValueError("cfg_scale must be number between 0-20")

    negative_prompt = data.get("negative_prompt")

    seed = data.get("seed", -1)
    if seed != -1:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
    else:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

    _report_generation_progress(progress_callback, 25, "preparing_conditioning")
    conditioning = [{
        "prompt": prompt,
        "seconds_total": seconds_total
    }]

    negative_conditioning = None
    if negative_prompt:
        negative_conditioning = [{
            "prompt": negative_prompt,
            "seconds_total": seconds_total
        }]

    if model_family == "sao1.0":
        seconds_start = data.get("seconds_start", 0)
        conditioning[0]["seconds_start"] = seconds_start
        if negative_conditioning:
            negative_conditioning[0]["seconds_start"] = seconds_start

    client_sampler_overrides = {}
    if "sampler_type" in data:
        client_sampler_overrides["sampler_type"] = data["sampler_type"]

    _report_generation_progress(progress_callback, 35, "configuring_sampler")
    skw = sampler_kwargs_for_objective(model, config, client_sampler_overrides)

    print(f"Generating with {model_type} model:")
    print(f"   Prompt: {prompt}")
    print(f"   Objective: {diffusion_objective}")
    print(f"   seconds_total: {seconds_total} (max {model_seconds_max})")
    print(f"   Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
    print(f"   Negative: {negative_prompt or 'None'}")

    sampling_progress_start = 22
    sampling_progress_end = 92
    if progress_callback is not None:
        _report_generation_progress(progress_callback, sampling_progress_start, "sampling")

        existing_sampler_callback = skw.get("callback")
        sampling_state = {
            "last_progress": sampling_progress_start - 1,
            "last_completed_step": 0,
            "saw_zero_index": False
        }

        def _sampling_step_callback(payload):
            if callable(existing_sampler_callback):
                try:
                    existing_sampler_callback(payload)
                except Exception:
                    pass

            if not isinstance(payload, dict):
                return

            raw_index = payload.get("i")
            if not isinstance(raw_index, (int, float)):
                return

            step_index = int(raw_index)
            if step_index < 0:
                return

            if step_index == 0:
                sampling_state["saw_zero_index"] = True

            if sampling_state["saw_zero_index"]:
                completed_step = step_index + 1
            else:
                # Some samplers report 1-based step numbers; prefer conservative mapping
                # unless we observe a zero index.
                cand_from_zero_based = max(1, step_index + 1)
                cand_from_one_based = max(1, step_index)
                prev_completed = sampling_state["last_completed_step"]
                viable = [c for c in (cand_from_zero_based, cand_from_one_based) if c >= prev_completed]
                completed_step = min(viable) if viable else max(cand_from_zero_based, cand_from_one_based, prev_completed)

            completed_step = max(1, min(steps, completed_step))
            sampling_state["last_completed_step"] = completed_step

            fraction = completed_step / float(max(steps, 1))
            progress = sampling_progress_start + int(round((sampling_progress_end - sampling_progress_start) * fraction))
            progress = max(sampling_progress_start, min(sampling_progress_end, progress))

            if progress > sampling_state["last_progress"]:
                sampling_state["last_progress"] = progress
                _report_generation_progress(progress_callback, progress, "sampling")

        skw["callback"] = _sampling_step_callback

    start_time = time.time()
    with resource_cleanup():
        maybe_empty_cache(device)
        with autocast_ctx(device):
            output = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                negative_conditioning=negative_conditioning,
                sample_size=model_sample_size,
                device=device,
                seed=seed,
                **skw
            )

    generation_time = time.time() - start_time

    _report_generation_progress(progress_callback, 95, "postprocessing")
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)

    requested_samples = seconds_total * sample_rate
    actual_samples = output.shape[1]
    if requested_samples < actual_samples:
        output = output[:, :requested_samples]
        print(f"   ✂️  Trimmed from {actual_samples/sample_rate:.1f}s to {seconds_total}s")

    output_int16 = output.mul(32767).to(torch.int16).cpu()

    detected_bpm = extract_bpm(prompt)
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sample_rate": sample_rate,
        "duration_seconds": seconds_total,
        "generation_time": round(generation_time, 2),
        "wall_time_sec": round(generation_time, 2),
        "realtime_factor": round(seconds_total / max(generation_time, 1e-6), 2),
        "detected_bpm": detected_bpm,
        "device": device,
        "backend_engine": "mps",
        "sampler_type": skw.get("sampler_type"),
        "model_type": model_type,
    }
    if device == "cuda":
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_peak = torch.cuda.max_memory_allocated() / 1e9
        metadata["gpu_memory_used"] = round(memory_used, 2)
        metadata["gpu_memory_peak"] = round(memory_peak, 2)
        torch.cuda.reset_peak_memory_stats()

    print(f"✅ Generated in {generation_time:.2f}s ({metadata['realtime_factor']:.1f}x RT)")

    _report_generation_progress(progress_callback, 98, "encoding_audio")
    buffer = io.BytesIO()
    save_audio(buffer, output_int16, sample_rate)
    wav_bytes = buffer.getvalue()
    audio_b64 = base64.b64encode(wav_bytes).decode()
    del output, output_int16

    _report_generation_progress(progress_callback, 100, "completed")
    return {
        "audio_base64": audio_b64,
        "wav_bytes": wav_bytes,
        "metadata": metadata
    }

def _run_generate_loop_request(data, progress_callback=None):
    if not isinstance(data, dict):
        raise ValueError("JSON body required")

    _report_generation_progress(progress_callback, 2, "validating")
    backend_engine, default_backend_engine, requested_backend_engine = _resolve_backend_engine_from_request(data)
    print(
        "🎛 Backend selection (/generate/loop):"
        f" request={requested_backend_engine!r},"
        f" default={default_backend_engine},"
        f" effective={backend_engine}"
    )
    if backend_engine == "mlx":
        return _run_generate_loop_request_mlx(data, progress_callback=progress_callback)
    print("🟠 Backend engine: mps (loop)")

    model_type = data.get("model_type", "standard")
    finetune_repo = data.get("finetune_repo")
    finetune_checkpoint = data.get("finetune_checkpoint")

    prompt = data.get("prompt")
    if not prompt:
        raise ValueError("prompt is required")

    loop_type = str(data.get("loop_type", "auto")).strip().lower()
    if loop_type not in ("auto", "drums", "instruments"):
        raise ValueError("loop_type must be auto, drums, or instruments")

    bars = data.get("bars")
    steps = data.get("steps", 8)
    cfg_scale = data.get("cfg_scale", 6.0)
    seed = data.get("seed", -1)

    try:
        steps = int(steps)
    except Exception as exc:
        raise ValueError("steps must be an integer") from exc
    if steps < 1 or steps > 250:
        raise ValueError("steps must be integer between 1-250")

    try:
        cfg_scale = float(cfg_scale)
    except Exception as exc:
        raise ValueError("cfg_scale must be a number") from exc
    if cfg_scale < 0 or cfg_scale > 20:
        raise ValueError("cfg_scale must be number between 0-20")

    try:
        seed = int(seed)
    except Exception as exc:
        raise ValueError("seed must be an integer") from exc

    detected_bpm = extract_bpm(prompt)
    if not detected_bpm:
        raise ValueError("BPM must be specified in prompt (e.g., '120bpm')")

    _report_generation_progress(progress_callback, 10, "loading_model")
    model, config, device = load_model(model_type, finetune_repo, finetune_checkpoint)

    sample_rate = int(config.get("sample_rate", 44100))
    model_sample_size = int(config.get("sample_size", 524288))
    max_duration = model_sample_size / sample_rate
    model_family = detect_model_family(config)

    if bars is not None and str(bars).strip() != "" and str(bars).lower() != "auto":
        try:
            bars = int(bars)
        except Exception as exc:
            raise ValueError("bars must be 1, 2, 4, or 8") from exc
    else:
        seconds_per_beat = 60.0 / detected_bpm
        seconds_per_bar = seconds_per_beat * 4
        max_loop_duration = max_duration - 1.0

        possible_bars = [8, 4, 2, 1]
        bars = 1
        for bar_count in possible_bars:
            loop_duration = seconds_per_bar * bar_count
            if loop_duration <= max_loop_duration:
                bars = bar_count
                break

    if bars not in [1, 2, 4, 8]:
        raise ValueError("bars must be 1, 2, 4, or 8")

    seconds_per_beat = 60.0 / detected_bpm
    seconds_per_bar = seconds_per_beat * 4
    calculated_loop_duration = seconds_per_bar * bars

    if calculated_loop_duration > max_duration:
        if calculated_loop_duration > (max_duration + 1.0):
            bars = max(1, bars // 2)
            calculated_loop_duration = seconds_per_bar * bars

    enhanced_prompt = prompt
    negative_prompt = ""
    if loop_type == "drums":
        if "drum" not in prompt.lower():
            enhanced_prompt = f"{prompt} drum loop"
        negative_prompt = "melody, harmony, pitched instruments, vocals, singing"
    elif loop_type == "instruments":
        if "drum" in prompt.lower():
            enhanced_prompt = prompt.replace("drum", "").replace("drums", "").strip()
        negative_prompt = "drums, percussion, kick, snare, hi-hat"

    _report_generation_progress(progress_callback, 25, "preparing_conditioning")
    generation_sample_size = model_sample_size
    seconds_total = max(1, model_sample_size // sample_rate)

    # SAO 1.0 can generate long windows; for loop mode we only need enough
    # samples for the requested loop duration, which significantly improves latency.
    if model_family == "sao1.0":
        requested_loop_seconds = max(1.0, min(max_duration, float(calculated_loop_duration)))
        requested_samples = int(np.ceil(requested_loop_seconds * sample_rate))

        downsampling_ratio = 1
        pretransform = getattr(model, "pretransform", None)
        if pretransform is not None:
            downsampling_ratio = int(getattr(pretransform, "downsampling_ratio", 1) or 1)
        downsampling_ratio = max(1, downsampling_ratio)

        aligned_samples = requested_samples
        if downsampling_ratio > 1:
            aligned_samples = ((aligned_samples + downsampling_ratio - 1) // downsampling_ratio) * downsampling_ratio

        generation_sample_size = int(max(1, min(model_sample_size, aligned_samples)))
        seconds_total = generation_sample_size / float(sample_rate)
    try:
        seconds_start = float(data.get("seconds_start", 0))
    except Exception:
        seconds_start = 0.0

    conditioning = [{
        "prompt": enhanced_prompt,
        "seconds_total": seconds_total
    }]

    negative_conditioning = None
    if negative_prompt:
        negative_conditioning = [{
            "prompt": negative_prompt,
            "seconds_total": seconds_total
        }]

    if model_family == "sao1.0":
        conditioning[0]["seconds_start"] = seconds_start
        if negative_conditioning:
            negative_conditioning[0]["seconds_start"] = seconds_start

    client_sampler_overrides = {}
    if "sampler_type" in data:
        client_sampler_overrides["sampler_type"] = data["sampler_type"]

    _report_generation_progress(progress_callback, 35, "configuring_sampler")
    skw = sampler_kwargs_for_objective(model, config, client_sampler_overrides)

    if seed != -1:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
    else:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)

    print("🔄 Loop generation:")
    print(f"   BPM: {detected_bpm}, Bars: {bars}")
    print(f"   Type: {loop_type}")
    print(f"   Model: {model_type}")
    print(f"   Enhanced prompt: {enhanced_prompt}")
    print(f"   Negative: {negative_prompt}")
    print(f"   seconds_total: {seconds_total}, seconds_start: {seconds_start}")
    print(f"   generation sample size: {generation_sample_size} (model max {model_sample_size})")
    print(f"   Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
    print(f"   Using sampler kwargs: {skw}")

    sampling_progress_start = 22
    sampling_progress_end = 92
    if progress_callback is not None:
        _report_generation_progress(progress_callback, sampling_progress_start, "sampling")

        existing_sampler_callback = skw.get("callback")
        sampling_state = {
            "last_progress": sampling_progress_start - 1,
            "last_completed_step": 0,
            "saw_zero_index": False
        }

        def _sampling_step_callback(payload):
            if callable(existing_sampler_callback):
                try:
                    existing_sampler_callback(payload)
                except Exception:
                    pass

            if not isinstance(payload, dict):
                return

            raw_index = payload.get("i")
            if not isinstance(raw_index, (int, float)):
                return

            step_index = int(raw_index)
            if step_index < 0:
                return

            if step_index == 0:
                sampling_state["saw_zero_index"] = True

            if sampling_state["saw_zero_index"]:
                completed_step = step_index + 1
            else:
                cand_from_zero_based = max(1, step_index + 1)
                cand_from_one_based = max(1, step_index)
                prev_completed = sampling_state["last_completed_step"]
                viable = [c for c in (cand_from_zero_based, cand_from_one_based) if c >= prev_completed]
                completed_step = min(viable) if viable else max(cand_from_zero_based, cand_from_one_based, prev_completed)

            completed_step = max(1, min(steps, completed_step))
            sampling_state["last_completed_step"] = completed_step

            fraction = completed_step / float(max(steps, 1))
            progress = sampling_progress_start + int(round((sampling_progress_end - sampling_progress_start) * fraction))
            progress = max(sampling_progress_start, min(sampling_progress_end, progress))

            if progress > sampling_state["last_progress"]:
                sampling_state["last_progress"] = progress
                _report_generation_progress(progress_callback, progress, "sampling")

        skw["callback"] = _sampling_step_callback

    start_time = time.time()
    with resource_cleanup():
        maybe_empty_cache(device)
        with autocast_ctx(device):
            output = generate_diffusion_cond(
                model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                negative_conditioning=negative_conditioning,
                sample_size=generation_sample_size,
                device=device,
                seed=seed,
                **skw
            )

    generation_time = time.time() - start_time

    _report_generation_progress(progress_callback, 95, "postprocessing")
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)

    loop_samples = int(calculated_loop_duration * sample_rate)
    available_samples = output.shape[1]
    available_duration = available_samples / sample_rate
    loop_duration = calculated_loop_duration

    if loop_samples > available_samples:
        loop_samples = available_samples
        loop_duration = available_duration

    loop_output = output[:, :loop_samples]
    loop_output_int16 = loop_output.mul(32767).to(torch.int16).cpu()

    metadata = {
        "prompt": enhanced_prompt,
        "original_prompt": prompt,
        "negative_prompt": negative_prompt,
        "loop_type": loop_type,
        "detected_bpm": detected_bpm,
        "bars": bars,
        "loop_duration_seconds": round(loop_duration, 2),
        "calculated_duration_seconds": round(calculated_loop_duration, 2),
        "available_audio_seconds": round(available_duration, 2),
        "seconds_per_bar": round(seconds_per_bar, 2),
        "style_transfer": False,
        "style_strength": None,
        "model_type": model_type,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "sample_rate": sample_rate,
        "generation_time": round(generation_time, 2),
        "wall_time_sec": round(generation_time, 2),
        "device": device,
        "backend_engine": "mps",
        "sampler_type": skw.get("sampler_type"),
    }

    print(f"✅ Loop generated: {loop_duration:.2f}s ({bars} bars at {detected_bpm}bpm)")

    _report_generation_progress(progress_callback, 98, "encoding_audio")
    buffer = io.BytesIO()
    save_audio(buffer, loop_output_int16, sample_rate)
    wav_bytes = buffer.getvalue()
    audio_b64 = base64.b64encode(wav_bytes).decode()
    del output, loop_output, loop_output_int16

    _report_generation_progress(progress_callback, 100, "completed")
    return {
        "audio_base64": audio_b64,
        "wav_bytes": wav_bytes,
        "metadata": metadata
    }

@app.route('/generate', methods=['POST'])
def generate_audio():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        return_format = data.get("return_format", "file")
        if return_format not in ["file", "base64"]:
            return jsonify({"error": "return_format must be 'file' or 'base64'"}), 400

        result = _run_generate_request(data)
        if return_format == "file":
            filename = f"stable_audio_{result['metadata'].get('seed', 'na')}_{int(time.time())}.wav"
            buffer = io.BytesIO(result["wav_bytes"])
            buffer.seek(0)
            return send_file(buffer, mimetype="audio/wav", as_attachment=True, download_name=filename)

        return jsonify({
            "success": True,
            "audio_base64": result["audio_base64"],
            "metadata": result["metadata"]
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        import traceback; traceback.print_exc()
        with resource_cleanup():
            pass
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/generate/async', methods=['POST'])
def generate_audio_async():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        request_id = str(data.get("request_id") or uuid.uuid4()).strip()
        if not request_id:
            request_id = str(uuid.uuid4())

        payload = dict(data)
        payload["return_format"] = "base64"

        _cleanup_jerry_jobs()
        _set_jerry_job_state(
            request_id,
            status="queued",
            stage="queued",
            progress=0,
            generation_in_progress=True,
            transform_in_progress=False,
            audio_data="",
            metadata={},
            error=""
        )

        def progress_callback(progress, stage):
            status = "queued" if stage == "queued" else "processing"
            _set_jerry_job_state(
                request_id,
                status=status,
                stage=stage,
                progress=progress,
                generation_in_progress=True,
                transform_in_progress=False
            )

        def worker():
            try:
                with jerry_generation_worker_lock:
                    _set_jerry_job_state(
                        request_id,
                        status="processing",
                        stage="starting",
                        progress=5,
                        generation_in_progress=True,
                        transform_in_progress=False
                    )

                    result = _run_generate_request(payload, progress_callback=progress_callback)

                _set_jerry_job_state(
                    request_id,
                    status="completed",
                    stage="done",
                    progress=100,
                    generation_in_progress=False,
                    transform_in_progress=False,
                    audio_data=result["audio_base64"],
                    metadata=result["metadata"],
                    error=""
                )
            except ValueError as e:
                _set_jerry_job_state(
                    request_id,
                    status="failed",
                    stage="failed",
                    generation_in_progress=False,
                    transform_in_progress=False,
                    error=str(e)
                )
            except Exception as e:
                print(f"❌ Async generation error [{request_id}]: {str(e)}")
                import traceback
                traceback.print_exc()
                _set_jerry_job_state(
                    request_id,
                    status="failed",
                    stage="failed",
                    generation_in_progress=False,
                    transform_in_progress=False,
                    error=str(e)
                )

        threading.Thread(target=worker, daemon=True).start()

        return jsonify({
            "success": True,
            "session_id": request_id,
            "status": "queued"
        })

    except Exception as e:
        print(f"❌ Async generate endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/generate/loop/async', methods=['POST'])
def generate_loop_async():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        request_id = str(data.get("request_id") or uuid.uuid4()).strip()
        if not request_id:
            request_id = str(uuid.uuid4())

        payload = dict(data)
        payload["return_format"] = "base64"

        _cleanup_jerry_jobs()
        _set_jerry_job_state(
            request_id,
            status="queued",
            stage="queued",
            progress=0,
            generation_in_progress=True,
            transform_in_progress=False,
            audio_data="",
            metadata={},
            error=""
        )

        def progress_callback(progress, stage):
            status = "queued" if stage == "queued" else "processing"
            _set_jerry_job_state(
                request_id,
                status=status,
                stage=stage,
                progress=progress,
                generation_in_progress=True,
                transform_in_progress=False
            )

        def worker():
            try:
                with jerry_generation_worker_lock:
                    _set_jerry_job_state(
                        request_id,
                        status="processing",
                        stage="starting",
                        progress=5,
                        generation_in_progress=True,
                        transform_in_progress=False
                    )

                    result = _run_generate_loop_request(payload, progress_callback=progress_callback)

                _set_jerry_job_state(
                    request_id,
                    status="completed",
                    stage="done",
                    progress=100,
                    generation_in_progress=False,
                    transform_in_progress=False,
                    audio_data=result["audio_base64"],
                    metadata=result["metadata"],
                    error=""
                )
            except ValueError as e:
                _set_jerry_job_state(
                    request_id,
                    status="failed",
                    stage="failed",
                    generation_in_progress=False,
                    transform_in_progress=False,
                    error=str(e)
                )
            except Exception as e:
                print(f"❌ Async loop generation error [{request_id}]: {str(e)}")
                import traceback
                traceback.print_exc()
                _set_jerry_job_state(
                    request_id,
                    status="failed",
                    stage="failed",
                    generation_in_progress=False,
                    transform_in_progress=False,
                    error=str(e)
                )

        threading.Thread(target=worker, daemon=True).start()

        return jsonify({
            "success": True,
            "session_id": request_id,
            "status": "queued"
        })

    except Exception as e:
        print(f"❌ Async loop endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/juce/poll_status/<request_id>', methods=['GET'])
def poll_jerry_generation_status(request_id):
    _cleanup_jerry_jobs()
    state = _get_jerry_job_state(request_id)
    if state is None:
        return jsonify({
            "success": False,
            "status": "failed",
            "error": "request_id not found",
            "generation_in_progress": False,
            "transform_in_progress": False,
            "progress": 0
        })

    status = state.get("status", "queued")
    stage = state.get("stage", "")
    progress = int(state.get("progress", 0))

    if status in ("queued", "processing"):
        queue_status = {
            "status": "queued" if status == "queued" else "processing",
            "message": stage or ("queued" if status == "queued" else "processing"),
            "position": 1 if status == "queued" else 0,
            "estimated_time": "",
            "estimated_seconds": 0
        }
        return jsonify({
            "success": True,
            "status": "processing",
            "generation_in_progress": True,
            "transform_in_progress": False,
            "progress": progress,
            "queue_status": queue_status
        })

    if status == "completed":
        return jsonify({
            "success": True,
            "status": "completed",
            "generation_in_progress": False,
            "transform_in_progress": False,
            "progress": 100,
            "audio_data": state.get("audio_data", ""),
            "metadata": state.get("metadata", {})
        })

    return jsonify({
        "success": False,
        "status": "failed",
        "error": state.get("error", "generation failed"),
        "generation_in_progress": False,
        "transform_in_progress": False,
        "progress": progress
    })


@app.route('/generate/style-transfer', methods=['POST'])
def generate_style_transfer():
    """
    Generate audio using style transfer from input audio.
    
    Form Data:
    - audio_file: Audio file (WAV, MP3, etc.)
    - prompt: Text prompt for style guidance (required)
    - negative_prompt: Negative text prompt (optional)
    - style_strength: Style transfer strength 0.1-1.0 (optional, default 0.8)
    - steps: Diffusion steps (optional, default 8)
    - cfg_scale: CFG scale (optional, default 6.0)
    - seed: Random seed (optional, -1 for random)
    - return_format: "file" or "base64" (optional, default "file")
    """
    try:
        # Check if audio file is provided
        if 'audio_file' not in request.files:
            return jsonify({"error": "audio_file is required"}), 400
        
        audio_file = request.files['audio_file']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Parse form parameters
        prompt = request.form.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        negative_prompt = request.form.get('negative_prompt')
        style_strength = float(request.form.get('style_strength', 0.8))
        steps = int(request.form.get('steps', 8))
        cfg_scale = float(request.form.get('cfg_scale', 6.0))
        seed = int(request.form.get('seed', -1))
        return_format = request.form.get('return_format', 'file')
        
        # Validate parameters
        if not (0.1 <= style_strength <= 1.0):
            return jsonify({"error": "style_strength must be between 0.1-1.0"}), 400
        
        if not isinstance(steps, int) or steps < 1 or steps > 100:
            return jsonify({"error": "steps must be integer between 1-100"}), 400
        
        if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0 or cfg_scale > 20:
            return jsonify({"error": "cfg_scale must be number between 0-20"}), 400
        
        if return_format not in ['file', 'base64']:
            return jsonify({"error": "return_format must be 'file' or 'base64'"}), 400
        
        # Load model
        model, config, device = load_model()
        
        # Process input audio
        input_sr, input_audio = process_input_audio(audio_file, config["sample_rate"])
        
        # Set seed for reproducibility
        if seed != -1:
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        else:
            # Generate random seed for logging
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        
        # Prepare conditioning
        conditioning = [{
            "prompt": prompt,
            "seconds_total": 12  # Fixed duration for this model
        }]
        
        # Prepare negative conditioning if provided
        negative_conditioning = None
        if negative_prompt:
            negative_conditioning = [{
                "prompt": negative_prompt,
                "seconds_total": 12
            }]
        
        print(f"🎨 Style transfer generation:")
        print(f"   Input audio: {input_audio.shape}")
        print(f"   Prompt: {prompt}")
        print(f"   Negative: {negative_prompt or 'None'}")
        print(f"   Style strength: {style_strength}")
        print(f"   Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
        
        # Generate audio with style transfer
        start_time = time.time()
        
        with resource_cleanup():
            # Clear GPU cache before generation
            maybe_empty_cache(device)
            
            with autocast_ctx(device):
                output = generate_diffusion_cond(
                    model,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    conditioning=conditioning,
                    negative_conditioning=negative_conditioning,
                    sample_size=config["sample_size"],
                    sampler_type="pingpong",
                    device=device,
                    seed=seed,
                    init_audio=(config["sample_rate"], input_audio),
                    init_noise_level=style_strength
                )
            
            generation_time = time.time() - start_time
            
            # Post-process audio
            output = rearrange(output, "b d n -> d (b n)")  # (2, N) stereo
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
            output_int16 = output.mul(32767).to(torch.int16).cpu()
            
            # Extract BPM for future use
            detected_bpm = extract_bpm(prompt)
            
            # Prepare response metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "style_strength": style_strength,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "sample_rate": config["sample_rate"],
                "duration_seconds": 12,
                "generation_time": round(generation_time, 2),
                "realtime_factor": round(12 / generation_time, 2),
                "detected_bpm": detected_bpm,
                "device": device,
                "input_audio_shape": list(input_audio.shape)
            }
            
            # Add memory info if CUDA
            if device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_peak = torch.cuda.max_memory_allocated() / 1e9
                metadata["gpu_memory_used"] = round(memory_used, 2)
                metadata["gpu_memory_peak"] = round(memory_peak, 2)
                # Reset peak stats for next generation
                torch.cuda.reset_peak_memory_stats()
            
            print(f"✅ Style transfer complete in {generation_time:.2f}s ({metadata['realtime_factor']:.1f}x RT)")
            
            if return_format == "file":
                # Return as WAV file download
                buffer = io.BytesIO()
                save_audio(buffer, output_int16, config["sample_rate"])
                buffer.seek(0)
                
                filename = f"style_transfer_{seed}_{int(time.time())}.wav"
                
                # Explicitly clean up tensors before returning
                del output
                del output_int16
                del input_audio
                
                return send_file(
                    buffer,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name=filename
                )
            
            else:  # base64 format
                # Return as JSON with base64 audio
                buffer = io.BytesIO()
                save_audio(buffer, output_int16, config["sample_rate"])
                audio_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Explicitly clean up tensors before returning
                del output
                del output_int16
                del input_audio
                
                return jsonify({
                    "success": True,
                    "audio_base64": audio_b64,
                    "metadata": metadata
                })
        
    except Exception as e:
        print(f"❌ Style transfer error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ensure cleanup on error as well
        with resource_cleanup():
            pass
            
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/generate/loop', methods=['POST'])
def generate_loop():
    try:
        # Detect content type and parse accordingly
        content_type = request.headers.get('Content-Type', '').lower()
        
        if 'application/json' in content_type:
            data = request.get_json()
            if not data:
                return jsonify({"error": "JSON body required"}), 400

            return_format = data.get('return_format', 'file')
            if return_format not in ['file', 'base64']:
                return jsonify({"error": "return_format must be 'file' or 'base64'"}), 400

            result = _run_generate_loop_request(data)
            if return_format == "file":
                metadata = result.get("metadata", {})
                loop_type = metadata.get("loop_type", "auto")
                detected_bpm = metadata.get("detected_bpm", "na")
                bars = metadata.get("bars", "na")
                seed = metadata.get("seed", "na")
                filename = f"loop_{loop_type}_{detected_bpm}bpm_{bars}bars_{seed}.wav"

                buffer = io.BytesIO(result["wav_bytes"])
                buffer.seek(0)
                return send_file(buffer, mimetype='audio/wav', as_attachment=True, download_name=filename)

            return jsonify({
                "success": True,
                "audio_base64": result["audio_base64"],
                "metadata": result["metadata"]
            })

        else:
            # Form data input
            # ---- NEW: Model selection from form ----
            model_type = request.form.get('model_type', 'standard')
            finetune_repo = request.form.get('finetune_repo')
            finetune_checkpoint = request.form.get('finetune_checkpoint')
            
            prompt = request.form.get('prompt')
            loop_type = request.form.get('loop_type', 'auto')
            bars = request.form.get('bars')
            style_strength = float(request.form.get('style_strength', 0.8))
            steps = int(request.form.get('steps', 8))
            cfg_scale = float(request.form.get('cfg_scale', 6.0))
            seed = int(request.form.get('seed', -1))
            return_format = request.form.get('return_format', 'file')
            
            audio_file = request.files.get('audio_file')
        
        # Check for input audio file (style transfer mode)
        input_audio = None
        if audio_file and audio_file.filename != '':
            # ---- NEW: Load model with parameters ----
            model, config, device = load_model(model_type, finetune_repo, finetune_checkpoint)
            input_sr, input_audio = process_input_audio(audio_file, config["sample_rate"])
        
        # Extract BPM from prompt
        detected_bpm = extract_bpm(prompt)
        if not detected_bpm:
            return jsonify({"error": "BPM must be specified in prompt (e.g., '120bpm')"}), 400
        
        # Calculate bars if not specified
        if bars:
            bars = int(bars)
        else:
            # ---- NEW: Get timing from model config ----
            if input_audio is None:
                model, config, device = load_model(model_type, finetune_repo, finetune_checkpoint)
            
            sample_rate = int(config.get("sample_rate", 44100))
            model_sample_size = int(config.get("sample_size", 524288))
            max_duration = model_sample_size / sample_rate
            
            seconds_per_beat = 60.0 / detected_bpm
            seconds_per_bar = seconds_per_beat * 4
            max_loop_duration = max_duration - 1.0  # Leave buffer
            
            possible_bars = [8, 4, 2, 1]
            bars = 1
            
            for bar_count in possible_bars:
                loop_duration = seconds_per_bar * bar_count
                if loop_duration <= max_loop_duration:
                    bars = bar_count
                    break
            
            print(f"🎵 Auto-selected {bars} bars ({bars * seconds_per_bar:.2f}s) for {detected_bpm} BPM")
        
        # Validate parameters
        if bars not in [1, 2, 4, 8]:
            return jsonify({"error": "bars must be 1, 2, 4, or 8"}), 400
        
        # Pre-calculate loop timing
        seconds_per_beat = 60.0 / detected_bpm
        seconds_per_bar = seconds_per_beat * 4
        calculated_loop_duration = seconds_per_bar * bars
        
        # Warn if loop might be too long
        if calculated_loop_duration > max_duration:
            print(f"⚠️  Warning: {bars} bars at {detected_bpm}bpm = {calculated_loop_duration:.2f}s (may exceed generated audio)")
            if calculated_loop_duration > max_duration + 1.0:
                bars = max(1, bars // 2)
                calculated_loop_duration = seconds_per_bar * bars
                print(f"🔧 Auto-reduced to {bars} bars ({calculated_loop_duration:.2f}s)")
        
        # Enhance prompt based on loop_type
        enhanced_prompt = prompt
        negative_prompt = ""
        
        if loop_type == "drums":
            if "drum" not in prompt.lower():
                enhanced_prompt = f"{prompt} drum loop"
            negative_prompt = "melody, harmony, pitched instruments, vocals, singing"
        elif loop_type == "instruments":
            if "drum" in prompt.lower():
                enhanced_prompt = prompt.replace("drum", "").replace("drums", "").strip()
            negative_prompt = "drums, percussion, kick, snare, hi-hat"
        
        print(f"🔄 Loop generation:")
        print(f"   BPM: {detected_bpm}, Bars: {bars}")
        print(f"   Type: {loop_type}")
        print(f"   Model: {model_type}")
        print(f"   Enhanced prompt: {enhanced_prompt}")
        print(f"   Negative: {negative_prompt}")
        print(f"   Input audio: {'Yes' if input_audio is not None else 'No'}")
        
        # ---- NEW: Load model if not already loaded ----
        if input_audio is None:
            model, config, device = load_model(model_type, finetune_repo, finetune_checkpoint)
        
        # Set seed
        if seed != -1:
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        
        # ---- NEW: Use per-model timing ----
        sample_rate = int(config.get("sample_rate", 44100))
        model_sample_size = int(config.get("sample_size", 524288))
        seconds_total = max(1, model_sample_size // sample_rate)
        model_family = detect_model_family(config)
        seconds_start = data.get("seconds_start", 0) if 'application/json' in content_type else request.form.get('seconds_start', 0)
        try:
            seconds_start = float(seconds_start)
        except Exception:
            seconds_start = 0.0
        
        # Prepare conditioning
        conditioning = [{
            "prompt": enhanced_prompt,
            "seconds_total": seconds_total
        }]
        
        negative_conditioning = None
        if negative_prompt:
            negative_conditioning = [{
                "prompt": negative_prompt,
                "seconds_total": seconds_total
            }]

        if model_family == "sao1.0":
            conditioning[0]["seconds_start"] = seconds_start
            if negative_conditioning:
                negative_conditioning[0]["seconds_start"] = seconds_start
        
        # ---- NEW: Dynamic sampler selection ----
        client_sampler_overrides = {}
        skw = sampler_kwargs_for_objective(model, config, client_sampler_overrides)
        
        print(f"   Using sampler kwargs: {skw}")
        
        # Generate audio
        start_time = time.time()
        
        with resource_cleanup():
            maybe_empty_cache(device)
            
            with autocast_ctx(device):
                if input_audio is not None:
                    # Style transfer mode
                    output = generate_diffusion_cond(
                        model,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        conditioning=conditioning,
                        negative_conditioning=negative_conditioning,
                        sample_size=model_sample_size,
                        device=device,
                        seed=seed,
                        init_audio=(sample_rate, input_audio),
                        init_noise_level=style_strength,
                        **skw  # Use dynamic sampler
                    )
                else:
                    # Text-to-audio mode
                    output = generate_diffusion_cond(
                        model,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        conditioning=conditioning,
                        negative_conditioning=negative_conditioning,
                        sample_size=model_sample_size,
                        device=device,
                        seed=seed,
                        **skw  # Use dynamic sampler
                    )
            
            generation_time = time.time() - start_time
            
            # Post-process audio
            output = rearrange(output, "b d n -> d (b n)")
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
            
            # Calculate loop slice
            loop_duration = calculated_loop_duration
            loop_samples = int(loop_duration * sample_rate)
            
            # Safety check
            available_samples = output.shape[1]
            available_duration = available_samples / sample_rate
            
            if loop_samples > available_samples:
                print(f"⚠️  Requested loop ({loop_duration:.2f}s) exceeds available audio ({available_duration:.2f}s)")
                print(f"   Using maximum available: {available_duration:.2f}s")
                loop_samples = available_samples
                loop_duration = available_duration
            
            # Extract loop
            loop_output = output[:, :loop_samples]
            loop_output_int16 = loop_output.mul(32767).to(torch.int16).cpu()
            
            # Prepare metadata
            metadata = {
                "prompt": enhanced_prompt,
                "original_prompt": prompt,
                "negative_prompt": negative_prompt,
                "loop_type": loop_type,
                "detected_bpm": detected_bpm,
                "bars": bars,
                "loop_duration_seconds": round(loop_duration, 2),
                "calculated_duration_seconds": round(calculated_loop_duration, 2),
                "available_audio_seconds": round(available_duration, 2),
                "seconds_per_bar": round(seconds_per_bar, 2),
                "style_transfer": input_audio is not None,
                "style_strength": style_strength if input_audio is not None else None,
                "model_type": model_type,  # NEW
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "sample_rate": sample_rate,
                "generation_time": round(generation_time, 2),
                "device": device
            }
            
            print(f"✅ Loop generated: {loop_duration:.2f}s ({bars} bars at {detected_bpm}bpm)")
            
            if return_format == "file":
                buffer = io.BytesIO()
                save_audio(buffer, loop_output_int16, sample_rate)
                buffer.seek(0)
                
                filename = f"loop_{loop_type}_{detected_bpm}bpm_{bars}bars_{seed}.wav"
                
                # Cleanup
                del output
                del loop_output
                del loop_output_int16
                if input_audio is not None:
                    del input_audio
                
                return send_file(
                    buffer,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name=filename
                )
            
            else:  # base64 format
                buffer = io.BytesIO()
                save_audio(buffer, loop_output_int16, sample_rate)
                audio_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Cleanup
                del output
                del loop_output
                del loop_output_int16
                if input_audio is not None:
                    del input_audio
                
                return jsonify({
                    "success": True,
                    "audio_base64": audio_b64,
                    "metadata": metadata
                })
    
    except Exception as e:
        print(f"❌ Loop generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        with resource_cleanup():
            pass
            
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route('/generate/loop-guided', methods=['POST'])
def generate_loop_guided():
    """
    Experimental: rectified-flow latent inpainting / “freeze first bar” loop gen.
    Uses Hawley-style latent guidance via sample_rf_guided.
    """
    try:
        content_type = request.headers.get('Content-Type', '').lower()

        # ----- Parse JSON + file like /generate/loop -----
        if 'application/json' in content_type:
            data = request.get_json()
            return jsonify({"error": "Use multipart/form-data with audio_file for this endpoint"}), 400
        elif 'multipart/form-data' in content_type:
            form = request.form
            files = request.files
            audio_file = files.get('audio_file')
            if not audio_file or audio_file.filename == '':
                return jsonify({"error": "audio_file is required for guided loop"}), 400

            # Model selection
            model_type = form.get('model_type', 'standard')
            finetune_repo = form.get('finetune_repo')
            finetune_checkpoint = form.get('finetune_checkpoint')

            prompt = form.get('prompt')
            if not prompt:
                return jsonify({"error": "prompt is required"}), 400

            negative_prompt = form.get('negative_prompt')
            cfg_scale = float(form.get('cfg_scale', 6.0))
            steps = int(form.get('steps', 50))
            seed = int(form.get('seed', -1))
            loop_type = form.get('loop_type', 'auto')  # reuse your loop typing
            bars = form.get('bars')  # can be None / "auto" / int as string

            # NEW: guidance options
            guidance_mask = form.get('guidance_mask', 'fade')
            guidance_strength = float(form.get('guidance_strength', 1.0))

            start_weight = form.get('guidance_start_weight')
            end_weight = form.get('guidance_end_weight')
            start_weight = float(start_weight) if start_weight is not None else None
            end_weight = float(end_weight) if end_weight is not None else None
        else:
            return jsonify({"error": "Unsupported Content-Type"}), 400

        # ----- Load model + config -----
        model, config, device = load_model(model_type, finetune_repo, finetune_checkpoint)
        sample_rate = config["sample_rate"]
        model_sample_size = config["sample_size"]

        # Only support rectified_flow / rf_denoiser for now
        diff_obj = getattr(model, "diffusion_objective", None)
        if diff_obj not in ("rectified_flow", "rf_denoiser"):
            return jsonify({"error": f"guided loop only supports rectified_flow / rf_denoiser (got {diff_obj})"}), 400

        # ----- Load input audio (your guitar riff) -----
        input_sr, input_audio = process_input_audio(audio_file, target_sr=sample_rate)
        # input_audio: shape [channels, samples] float32

        # ----- BPM + bars logic (simplified version of /generate/loop) -----
        detected_bpm = extract_bpm(prompt)
        if not detected_bpm:
            return jsonify({"error": "BPM must be in the prompt (e.g. '120bpm')"}), 400

        bpm = detected_bpm
        seconds_per_bar = 4.0 * 60.0 / bpm

        # Default to auto bars if not provided
        if bars and bars != "auto":
            bars = int(bars)
        else:
            # reuse your simple auto-bar heuristic: 4 or 8 bars depending on model window
            max_seconds = model_sample_size / sample_rate
            # try 8,4,2,1 bars
            for bar_count in [8, 4, 2, 1]:
                if bar_count * seconds_per_bar <= max_seconds:
                    bars = bar_count
                    break
            else:
                bars = 1

        seconds_total = bars * seconds_per_bar

        # ----- Build conditioning like /generate (just reuse structure) -----
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0.0,
            "seconds_total": seconds_total,
        }]

        negative_conditioning = None
        if negative_prompt:
            negative_conditioning = [{
                "prompt": negative_prompt,
                "seconds_start": 0.0,
                "seconds_total": seconds_total,
            }]

        # ----- Sampler kwargs (reuse helper) -----
        client_overrides = {}
        # Allow client to override sampler_type for experimentation
        if 'sampler_type' in form:
            client_overrides["sampler_type"] = form.get('sampler_type')

        sampler_kwargs = sampler_kwargs_for_objective(model, config, client_overrides)

        # ----- Build latent inpainting guidance from guitar riff -----
        guidance = build_latent_guidance_from_audio(
            model=model,
            input_audio=input_audio,
            input_sr=input_sr,
            sample_rate=sample_rate,
            sample_size=model_sample_size,
            bars=bars,
            strength=guidance_strength,
            t_min=0.2,
            t_max=0.999,
            mask_mode=guidance_mask,
            start_weight=start_weight,
            end_weight=end_weight,
        )

        # ----- Call guided generator -----
        with resource_cleanup():
            maybe_empty_cache(device)
            if seed == -1:
                seed = np.random.randint(0, 2**32 - 1)
            print(f"🎛 guided loop seed: {seed}")

            with autocast_ctx(device):
                output = generate_diffusion_cond_guided(
                    model=model,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    conditioning=conditioning,
                    negative_conditioning=negative_conditioning,
                    sample_size=model_sample_size,
                    sample_rate=sample_rate,
                    seed=seed,
                    device=device,
                    sampler_kwargs=sampler_kwargs,
                    guidance=guidance,
                )

        # ----- Post-process to stereo float32 -----
        # output: [batch, channels, samples]
        output = output[0]  # batch=1
        output = output.to(torch.float32)
        output = output / (torch.max(torch.abs(output)) + 1e-9)
        output = output.clamp(-1, 1)

        # Trim to requested loop length
        requested_samples = int(seconds_total * sample_rate)
        if output.shape[-1] > requested_samples:
            output = output[:, :requested_samples]

        # Encode to wav
        buf = io.BytesIO()
        
        save_audio(buf, output, sample_rate)
        buf.seek(0)

        return send_file(
            buf,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"guided_loop_{uuid.uuid4().hex}.wav",
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get detailed model information."""
    try:
        model, config, device = load_model()
        return jsonify({
            "model_name": "stabilityai/stable-audio-open-small",
            "device": device,
            "config": {
                "sample_rate": config["sample_rate"],
                "sample_size": config["sample_size"],
                "max_duration_seconds": 12,
                "diffusion_objective": getattr(model, 'diffusion_objective', 'unknown'),
                "io_channels": getattr(model, 'io_channels', 'unknown')
            },
            "supported_endpoints": {
                "/generate": "Text-to-audio generation",
                "/generate/style-transfer": "Audio-to-audio style transfer",
                "/generate/loop": "BPM-aware loop generation (text or style transfer)"
            },
            "supported_parameters": {
                "prompt": {"type": "string", "required": True},
                "steps": {"type": "int", "default": 8, "range": "1-100"},
                "cfg_scale": {"type": "float", "default": 6.0, "range": "0-20"},
                "negative_prompt": {"type": "string", "required": False},
                "seed": {"type": "int", "default": -1, "note": "-1 for random"},
                "return_format": {"type": "string", "default": "file", "options": ["file", "base64"]},
                "style_strength": {"type": "float", "default": 0.8, "range": "0.1-1.0", "note": "For style transfer"},
                "loop_type": {"type": "string", "default": "auto", "options": ["drums", "instruments", "auto"]},
                "bars": {"type": "int", "default": "auto", "options": [1, 2, 4, 8]}
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/generate/loop-with-riff', methods=['POST'])
def generate_loop_with_riff():
    """
    Generate loop using your personal riff library as style transfer input
    
    Form Data:
    - prompt: Text prompt with BPM (required, e.g., "aggressive techno 140bpm")
    - key: Musical key (required, e.g., "gsharp", "f", "csharp")
    - loop_type: "drums" or "instruments" (optional, default "instruments")
    - bars: Number of bars (optional, auto-calculated)
    - style_strength: Style transfer strength (optional, default 0.8)
    - steps: Diffusion steps (optional, default 8)
    - cfg_scale: CFG scale (optional, default 1.0)
    - seed: Random seed (optional, -1 for random)
    - return_format: "file" or "base64" (optional, default "file")
    """
    try:
        # Parse form parameters
        prompt = request.form.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        key = request.form.get('key')
        if not key:
            return jsonify({"error": "key is required (e.g., 'gsharp', 'f', 'csharp')"}), 400
        
        loop_type = request.form.get('loop_type', 'instruments')
        bars = request.form.get('bars')
        style_strength = float(request.form.get('style_strength', 0.8))
        steps = int(request.form.get('steps', 8))
        cfg_scale = float(request.form.get('cfg_scale', 1.0))
        seed = int(request.form.get('seed', -1))
        return_format = request.form.get('return_format', 'file')
        
        # Extract BPM from prompt
        detected_bpm = extract_bpm(prompt)
        if not detected_bpm:
            return jsonify({"error": "BPM must be specified in prompt (e.g., '120bpm')"}), 400
        
        print(f"🎸 Riff-based generation request:")
        print(f"   Key: {key}")
        print(f"   Target BPM: {detected_bpm}")
        print(f"   Prompt: {prompt}")
        print(f"   Loop type: {loop_type}")
        
        # Get riff from library
        riff_temp_path = riff_manager.get_riff_for_style_transfer(key, detected_bpm)
        if not riff_temp_path:
            available_keys = riff_manager.get_available_keys()
            return jsonify({
                "error": f"No riffs available for key '{key}'",
                "available_keys": available_keys
            }), 400
        
        try:
            # Load model
            model, config, device = load_model()
            
            # Process the riff audio
            input_sr, input_audio = process_input_audio_from_path(riff_temp_path, config["sample_rate"])
            
            # Calculate bars if not specified
            if bars:
                bars = int(bars)
            else:
                seconds_per_beat = 60.0 / detected_bpm
                seconds_per_bar = seconds_per_beat * 4
                max_loop_duration = 10.0
                
                possible_bars = [8, 4, 2, 1]
                bars = 1
                
                for bar_count in possible_bars:
                    loop_duration = seconds_per_bar * bar_count
                    if loop_duration <= max_loop_duration:
                        bars = bar_count
                        break
                
                print(f"🎵 Auto-selected {bars} bars for {detected_bpm} BPM")
            
            # Enhance prompt based on loop_type
            enhanced_prompt = prompt
            negative_prompt = ""
            
            if loop_type == "drums":
                if "drum" not in prompt.lower():
                    enhanced_prompt = f"{prompt} drum loop"
                negative_prompt = "melody, harmony, pitched instruments, vocals, singing"
            elif loop_type == "instruments":
                if "drum" in prompt.lower():
                    enhanced_prompt = prompt.replace("drum", "").replace("drums", "").strip()
                negative_prompt = "drums, percussion, kick, snare, hi-hat"
            
            # Set seed
            if seed != -1:
                torch.manual_seed(seed)
                if device == "cuda":
                    torch.cuda.manual_seed(seed)
            else:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
                torch.manual_seed(seed)
                if device == "cuda":
                    torch.cuda.manual_seed(seed)
            
            # Prepare conditioning
            conditioning = [{
                "prompt": enhanced_prompt,
                "seconds_total": 12
            }]
            
            negative_conditioning = None
            if negative_prompt:
                negative_conditioning = [{
                    "prompt": negative_prompt,
                    "seconds_total": 12
                }]
            
            print(f"🎨 Starting style transfer with {key} riff...")
            
            # Generate audio with style transfer
            start_time = time.time()
            
            with resource_cleanup():
                maybe_empty_cache(device)
                
            with autocast_ctx(device):
                output = generate_diffusion_cond(
                        model,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        conditioning=conditioning,
                        negative_conditioning=negative_conditioning,
                        sample_size=config["sample_size"],
                        sampler_type="pingpong",
                        device=device,
                        seed=seed,
                        init_audio=(config["sample_rate"], input_audio),
                        init_noise_level=style_strength
                    )
                
                generation_time = time.time() - start_time
                
                # Post-process audio (same as existing endpoint)
                output = rearrange(output, "b d n -> d (b n)")
                output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
                
                # Calculate loop slice
                sample_rate = config["sample_rate"]
                seconds_per_beat = 60.0 / detected_bpm
                seconds_per_bar = seconds_per_beat * 4
                loop_duration = seconds_per_bar * bars
                loop_samples = int(loop_duration * sample_rate)
                
                # Safety check
                available_samples = output.shape[1]
                available_duration = available_samples / sample_rate
                
                if loop_samples > available_samples:
                    loop_samples = available_samples
                    loop_duration = available_duration
                
                # Extract loop
                loop_output = output[:, :loop_samples]
                loop_output_int16 = loop_output.mul(32767).to(torch.int16).cpu()
                
                # Prepare metadata
                metadata = {
                    "prompt": enhanced_prompt,
                    "original_prompt": prompt,
                    "key": key,
                    "negative_prompt": negative_prompt,
                    "loop_type": loop_type,
                    "detected_bpm": detected_bpm,
                    "bars": bars,
                    "loop_duration_seconds": round(loop_duration, 2),
                    "style_transfer": True,
                    "style_strength": style_strength,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                    "sample_rate": sample_rate,
                    "generation_time": round(generation_time, 2),
                    "device": device,
                    "source": "personal_riff_library"
                }
                
                print(f"✅ Riff-based loop generated: {loop_duration:.2f}s ({bars} bars at {detected_bpm}bpm)")
                
                if return_format == "file":
                    buffer = io.BytesIO()
                    save_audio(buffer, loop_output_int16, sample_rate)
                    buffer.seek(0)
                    
                    filename = f"riff_{key}_{loop_type}_{detected_bpm}bpm_{bars}bars_{seed}.wav"
                    
                    # Cleanup
                    del output
                    del loop_output
                    del loop_output_int16
                    del input_audio
                    
                    return send_file(
                        buffer,
                        mimetype='audio/wav',
                        as_attachment=True,
                        download_name=filename
                    )
                
                else:  # base64 format
                    buffer = io.BytesIO()
                    save_audio(buffer, loop_output_int16, sample_rate)
                    audio_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Cleanup
                    del output
                    del loop_output
                    del loop_output_int16
                    del input_audio
                    
                    return jsonify({
                        "success": True,
                        "audio_base64": audio_b64,
                        "metadata": metadata
                    })
        
        finally:
            # Always clean up the temp riff file
            if os.path.exists(riff_temp_path):
                os.unlink(riff_temp_path)
                print(f"🧹 Cleaned up temp riff file")
    
    except Exception as e:
        print(f"❌ Riff generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/riffs/available', methods=['GET'])
def get_available_riffs():
    """Get information about available riffs"""
    try:
        keys = riff_manager.get_available_keys()
        riff_info = {}
        
        for key in keys:
            riffs = riff_manager.get_riffs_for_key(key)
            riff_info[key] = [
                {
                    "filename": riff["filename"],
                    "original_bpm": riff["original_bpm"],
                    "description": riff["description"]
                }
                for riff in riffs
            ]
        
        return jsonify({
            "available_keys": keys,
            "total_keys": len(keys),
            "riff_details": riff_info
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helper function for loading audio from file path (add this too)
def process_input_audio_from_path(file_path, target_sr):
    """Process audio file from path into tensor format."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Ensure stereo output
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        print(f"📁 Processed riff audio: {waveform.shape} at {target_sr}Hz")
        return sample_rate, waveform
    
    except Exception as e:
        raise ValueError(f"Failed to process riff audio: {str(e)}")
    


@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """Manually trigger GPU cleanup."""
    try:
        with resource_cleanup():
            pass
        return jsonify({"message": "GPU cleanup completed successfully"})
    except Exception as e:
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500
    

@app.route('/debug/checkpoint', methods=['GET'])
def debug_checkpoint_structure():
    """Debug endpoint to analyze checkpoint structure mismatch"""
    try:
        from huggingface_hub import hf_hub_download, login
        from stable_audio_tools.models import create_model_from_config
        import json
        import torch
        import os
        
        results = {"debug_info": []}
        
        def add_log(message):
            results["debug_info"].append(message)
            print(message)  # Also log to console
        
        add_log("🔍 Starting checkpoint structure analysis...")
        
        # Authenticate
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
            add_log(f"✅ HF authenticated")
        
        # Download files
        add_log("📥 Downloading base_model_config.json...")
        config_path = hf_hub_download(
            repo_id="stabilityai/stable-audio-open-small",
            filename="base_model_config.json"
        )
        
        add_log("📥 Downloading finetune checkpoint...")
        ckpt_path = hf_hub_download(
            repo_id="S3Sound/am_saos1",
            filename="am_saos1_e18_s4800.ckpt"
        )
        
        # Load config and create model
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        add_log("🔧 Creating model from base config...")
        model = create_model_from_config(config)
        
        # Get expected model keys
        model_keys = set(model.state_dict().keys())
        add_log(f"📊 Model expects {len(model_keys)} keys")
        
        model_keys_sample = sorted(model_keys)[:10]
        results["model_keys_sample"] = model_keys_sample
        add_log("📝 First 10 expected keys:")
        for key in model_keys_sample:
            add_log(f"   {key}")
        
        # Load and analyze checkpoint
        add_log("🎯 Loading checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        checkpoint_structure = list(checkpoint.keys())
        results["checkpoint_structure"] = checkpoint_structure
        add_log(f"🔍 Checkpoint top-level keys: {checkpoint_structure}")
        
        # Find the actual state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict_key = "state_dict"
            add_log("   Using checkpoint['state_dict']")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            state_dict_key = "model"
            add_log("   Using checkpoint['model']")
        else:
            state_dict = checkpoint
            state_dict_key = "root"
            add_log("   Using checkpoint directly")
        
        results["state_dict_location"] = state_dict_key
        
        checkpoint_keys = set(state_dict.keys())
        add_log(f"📊 Checkpoint has {len(checkpoint_keys)} keys")
        
        checkpoint_keys_sample = sorted(checkpoint_keys)[:10]
        results["checkpoint_keys_sample"] = checkpoint_keys_sample
        add_log("📝 First 10 checkpoint keys:")
        for key in checkpoint_keys_sample:
            add_log(f"   {key}")
        
        # Analyze key patterns
        add_log("🔍 Key Pattern Analysis:")
        
        # Check for common prefixes in checkpoint keys
        checkpoint_prefixes = set()
        for key in checkpoint_keys:
            parts = key.split('.')
            if len(parts) > 1:
                checkpoint_prefixes.add(parts[0])
        
        checkpoint_prefixes_list = sorted(checkpoint_prefixes)
        results["checkpoint_prefixes"] = checkpoint_prefixes_list
        add_log(f"📝 Checkpoint key prefixes: {checkpoint_prefixes_list}")
        
        # Check for common prefixes in model keys  
        model_prefixes = set()
        for key in model_keys:
            parts = key.split('.')
            if len(parts) > 1:
                model_prefixes.add(parts[0])
        
        model_prefixes_list = sorted(model_prefixes)
        results["model_prefixes"] = model_prefixes_list
        add_log(f"📝 Model key prefixes: {model_prefixes_list}")
        
        # Try different key cleaning strategies
        add_log("🧪 Testing key cleaning strategies:")
        
        strategies = [
            ("no_cleaning", lambda k: k),
            ("remove_model_prefix", lambda k: k[6:] if k.startswith('model.') else k),
            ("remove_ema_model_prefix", lambda k: k[10:] if k.startswith('ema_model.') else k),
            ("remove_module_prefix", lambda k: k[7:] if k.startswith('module.') else k),
            ("add_model_prefix", lambda k: f"model.{k}"),
            ("remove_first_prefix", lambda k: '.'.join(k.split('.')[1:]) if '.' in k else k),
        ]
        
        strategy_results = {}
        
        for strategy_name, strategy_func in strategies:
            cleaned_keys = set(strategy_func(k) for k in checkpoint_keys)
            
            missing = model_keys - cleaned_keys
            unexpected = cleaned_keys - model_keys
            matching = model_keys & cleaned_keys
            
            match_percentage = len(matching)/len(model_keys)*100
            
            strategy_info = {
                "matching": len(matching),
                "total_model_keys": len(model_keys),
                "match_percentage": round(match_percentage, 1),
                "missing": len(missing),
                "unexpected": len(unexpected)
            }
            
            add_log(f"📊 Strategy '{strategy_name}':")
            add_log(f"   Matching: {len(matching)}/{len(model_keys)} ({match_percentage:.1f}%)")
            add_log(f"   Missing: {len(missing)}")
            add_log(f"   Unexpected: {len(unexpected)}")
            
            if len(matching) > len(model_keys) * 0.8:  # If > 80% match
                add_log(f"   ✅ Good strategy! Sample matches:")
                sample_matches = []
                for i, key in enumerate(sorted(matching)):
                    if i < 5:
                        original = None
                        for orig_key in checkpoint_keys:
                            if strategy_func(orig_key) == key:
                                original = orig_key
                                break
                        match_pair = f"{original} -> {key}"
                        sample_matches.append(match_pair)
                        add_log(f"      {match_pair}")
                
                strategy_info["sample_matches"] = sample_matches
                
                if len(missing) > 0:
                    add_log(f"   ❌ Sample missing keys:")
                    sample_missing = sorted(missing)[:5]
                    strategy_info["sample_missing"] = sample_missing
                    for key in sample_missing:
                        add_log(f"      {key}")
            
            strategy_results[strategy_name] = strategy_info
        
        results["strategy_analysis"] = strategy_results
        
        # Check for exact matches
        if checkpoint_keys == model_keys:
            add_log("✅ Keys match exactly - this shouldn't be happening!")
            results["keys_match_exactly"] = True
        else:
            results["keys_match_exactly"] = False
        
        # Summary and recommendation
        add_log("🔍 Analysis Summary:")
        best_strategy = None
        best_match_rate = 0
        
        for strategy_name, info in strategy_results.items():
            if info["match_percentage"] > best_match_rate:
                best_match_rate = info["match_percentage"]
                best_strategy = strategy_name
        
        results["best_strategy"] = best_strategy
        results["best_match_rate"] = best_match_rate
        
        if best_match_rate > 80:
            add_log(f"🎯 RECOMMENDATION: Use '{best_strategy}' strategy ({best_match_rate:.1f}% match)")
            results["recommendation"] = f"Use '{best_strategy}' strategy"
        else:
            add_log("❌ No strategy achieved >80% match. This checkpoint may be incompatible.")
            results["recommendation"] = "No compatible strategy found"
        
        results["success"] = True
        return jsonify(results)
        
    except Exception as e:
        print(f"Debug checkpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "debug_info": results.get("debug_info", [])
        }), 500
    
@app.route('/debug/pretransform', methods=['GET'])
def debug_pretransform():
    """Debug endpoint to check if checkpoint contains pretransform weights"""
    try:
        from huggingface_hub import hf_hub_download, login
        from stable_audio_tools.models import create_model_from_config
        from stable_audio_tools import get_pretrained_model
        import json
        import torch
        import os
        
        results = {"debug_info": []}
        
        def add_log(message):
            results["debug_info"].append(message)
            print(message)
        
        add_log("🔍 Analyzing pretransform weights...")
        
        # Authenticate
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
        
        # Load the standard model for comparison
        add_log("📥 Loading standard SAOS model...")
        standard_model, standard_config = get_pretrained_model("stabilityai/stable-audio-open-small")
        standard_keys = set(standard_model.state_dict().keys())
        
        # Find pretransform keys in standard model
        pretransform_keys = {k for k in standard_keys if k.startswith('pretransform.')}
        non_pretransform_keys = standard_keys - pretransform_keys
        
        add_log(f"📊 Standard model analysis:")
        add_log(f"   Total keys: {len(standard_keys)}")
        add_log(f"   Pretransform keys: {len(pretransform_keys)}")
        add_log(f"   Non-pretransform keys: {len(non_pretransform_keys)}")
        
        results["standard_model"] = {
            "total_keys": len(standard_keys),
            "pretransform_keys": len(pretransform_keys),
            "non_pretransform_keys": len(non_pretransform_keys)
        }
        
        # Sample pretransform keys
        sample_pretransform = sorted(pretransform_keys)[:5]
        results["sample_pretransform_keys"] = sample_pretransform
        add_log("📝 Sample pretransform keys:")
        for key in sample_pretransform:
            add_log(f"   {key}")
        
        # Load finetune checkpoint
        add_log("📥 Loading finetune checkpoint...")
        ckpt_path = hf_hub_download(
            repo_id="S3Sound/am_saos1",
            filename="am_saos1_e18_s4800.ckpt"
        )
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            finetune_state_dict = checkpoint['state_dict']
        else:
            finetune_state_dict = checkpoint
        
        finetune_keys = set(finetune_state_dict.keys())
        finetune_pretransform_keys = {k for k in finetune_keys if k.startswith('pretransform.')}
        finetune_non_pretransform_keys = finetune_keys - finetune_pretransform_keys
        
        add_log(f"📊 Finetune checkpoint analysis:")
        add_log(f"   Total keys: {len(finetune_keys)}")
        add_log(f"   Pretransform keys: {len(finetune_pretransform_keys)}")
        add_log(f"   Non-pretransform keys: {len(finetune_non_pretransform_keys)}")
        
        results["finetune_checkpoint"] = {
            "total_keys": len(finetune_keys),
            "pretransform_keys": len(finetune_pretransform_keys),
            "non_pretransform_keys": len(finetune_non_pretransform_keys)
        }
        
        # Check if finetune has pretransform weights
        if len(finetune_pretransform_keys) == 0:
            add_log("❌ PROBLEM FOUND: Finetune checkpoint has NO pretransform weights!")
            add_log("   This explains the static drone - pretransform has random weights")
            results["has_pretransform_weights"] = False
            results["problem_identified"] = "Missing pretransform weights in finetune checkpoint"
        else:
            add_log("✅ Finetune checkpoint has pretransform weights")
            results["has_pretransform_weights"] = True
        
        # Compare key coverage
        missing_pretransform = pretransform_keys - finetune_pretransform_keys
        missing_main_model = non_pretransform_keys - finetune_non_pretransform_keys
        
        if missing_pretransform:
            add_log(f"❌ Missing {len(missing_pretransform)} pretransform keys from finetune")
            results["missing_pretransform_count"] = len(missing_pretransform)
        
        if missing_main_model:
            add_log(f"❌ Missing {len(missing_main_model)} main model keys from finetune")
            results["missing_main_model_count"] = len(missing_main_model)
        
        # Recommendation
        add_log("\n🎯 SOLUTION:")
        if len(finetune_pretransform_keys) == 0:
            add_log("1. Load pretransform weights from standard SAOS model")
            add_log("2. Load only main model weights from finetune checkpoint") 
            add_log("3. This gives us: finetuned diffusion + standard pretransform")
            results["recommended_approach"] = "hybrid_loading"
        else:
            add_log("   Both models have pretransform - investigate further")
            results["recommended_approach"] = "investigate_further"
        
        results["success"] = True
        return jsonify(results)
        
    except Exception as e:
        print(f"Debug pretransform error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "debug_info": results.get("debug_info", [])
        }), 500
    
@app.route('/debug/weights', methods=['GET'])
def debug_weight_comparison():
    """Compare actual weight values between standard and finetune models"""
    try:
        from huggingface_hub import hf_hub_download, login
        from stable_audio_tools import get_pretrained_model
        import torch
        import os
        
        results = {"debug_info": []}
        
        def add_log(message):
            results["debug_info"].append(message)
            print(message)
        
        add_log("🔍 Comparing pretransform weight values...")
        
        # Authenticate
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
        
        # Load standard model
        add_log("📥 Loading standard SAOS pretransform weights...")
        standard_model, _ = get_pretrained_model("stabilityai/stable-audio-open-small")
        standard_state = standard_model.state_dict()
        
        # Load finetune checkpoint
        add_log("📥 Loading finetune checkpoint...")
        ckpt_path = hf_hub_download(
            repo_id="S3Sound/am_saos1",
            filename="am_saos1_e18_s4800.ckpt"
        )
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        finetune_state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Compare a few key pretransform weights
        test_keys = [
            'pretransform.model.decoder.layers.0.weight_g',
            'pretransform.model.decoder.layers.0.bias',
            'pretransform.model.encoder.layers.0.weight_g'
        ]
        
        weight_comparisons = {}
        
        for key in test_keys:
            if key in standard_state and key in finetune_state:
                std_weight = standard_state[key]
                ft_weight = finetune_state[key]
                
                # Compare shapes
                shapes_match = std_weight.shape == ft_weight.shape
                
                # Compare values (check if they're identical)
                weights_identical = torch.allclose(std_weight, ft_weight, atol=1e-6)
                
                # Get some statistics
                std_mean = float(std_weight.mean())
                ft_mean = float(ft_weight.mean())
                std_std = float(std_weight.std())
                ft_std = float(ft_weight.std())
                
                weight_comparisons[key] = {
                    "shapes_match": shapes_match,
                    "weights_identical": weights_identical,
                    "standard_mean": round(std_mean, 6),
                    "finetune_mean": round(ft_mean, 6),
                    "standard_std": round(std_std, 6),
                    "finetune_std": round(ft_std, 6)
                }
                
                add_log(f"🔍 Key: {key}")
                add_log(f"   Shapes match: {shapes_match}")
                add_log(f"   Weights identical: {weights_identical}")
                add_log(f"   Standard: mean={std_mean:.6f}, std={std_std:.6f}")
                add_log(f"   Finetune: mean={ft_mean:.6f}, std={ft_std:.6f}")
                
        results["weight_comparisons"] = weight_comparisons
        
        # Check if ANY pretransform weights are identical
        identical_pretransform_weights = 0
        total_pretransform_weights = 0
        
        for key in standard_state.keys():
            if key.startswith('pretransform.') and key in finetune_state:
                total_pretransform_weights += 1
                if torch.allclose(standard_state[key], finetune_state[key], atol=1e-6):
                    identical_pretransform_weights += 1
        
        identical_percentage = (identical_pretransform_weights / total_pretransform_weights) * 100
        
        add_log(f"📊 Pretransform weight analysis:")
        add_log(f"   Identical weights: {identical_pretransform_weights}/{total_pretransform_weights}")
        add_log(f"   Identical percentage: {identical_percentage:.1f}%")
        
        results["pretransform_analysis"] = {
            "identical_weights": identical_pretransform_weights,
            "total_weights": total_pretransform_weights,
            "identical_percentage": round(identical_percentage, 1)
        }
        
        # Recommendation
        if identical_percentage > 95:
            add_log("✅ Pretransform weights are nearly identical - issue elsewhere")
            results["recommendation"] = "pretransform_weights_good"
        elif identical_percentage < 5:
            add_log("❌ Pretransform weights completely different - use hybrid loading")
            results["recommendation"] = "use_hybrid_loading"
        else:
            add_log("⚠️  Pretransform weights partially different - investigate further")
            results["recommendation"] = "investigate_partial_difference"
        
        results["success"] = True
        return jsonify(results)
        
    except Exception as e:
        print(f"Debug weights error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "debug_info": results.get("debug_info", [])
        }), 500
    
@app.route('/debug/loading', methods=['GET'])
def debug_loading_process():
    """Compare the loading process between standard and finetune models"""
    try:
        results = {"debug_info": []}
        
        def add_log(message):
            results["debug_info"].append(message)
            print(message)
        
        add_log("🔍 Analyzing model loading processes...")
        
        # Get both models from our cache
        if len(model_manager.model_cache) < 2:
            add_log("❌ Need both models loaded in cache first")
            return jsonify({
                "error": "Both models need to be loaded first. Call /test/finetune to load both.",
                "debug_info": results["debug_info"]
            }), 400
        
        # Find the two models in cache
        standard_key = "standard_saos"
        finetune_key = None
        
        for key in model_manager.model_cache.keys():
            if key != standard_key:
                finetune_key = key
                break
        
        if not finetune_key:
            add_log("❌ Finetune model not found in cache")
            return jsonify({"error": "Finetune model not in cache"}), 400
        
        standard_data = model_manager.model_cache[standard_key]
        finetune_data = model_manager.model_cache[finetune_key]
        
        add_log(f"📊 Comparing models:")
        add_log(f"   Standard: {standard_data['source']}")
        add_log(f"   Finetune: {finetune_data['source']}")
        
        # Compare model properties
        std_model = standard_data["model"]
        ft_model = finetune_data["model"]
        
        # Check model modes
        std_training = std_model.training
        ft_training = ft_model.training
        
        add_log(f"🔍 Model states:")
        add_log(f"   Standard training mode: {std_training}")
        add_log(f"   Finetune training mode: {ft_training}")
        
        results["model_states"] = {
            "standard_training": std_training,
            "finetune_training": ft_training
        }
        
        # Check model types
        std_type = type(std_model).__name__
        ft_type = type(ft_model).__name__
        
        add_log(f"🔍 Model types:")
        add_log(f"   Standard: {std_type}")
        add_log(f"   Finetune: {ft_type}")
        
        results["model_types"] = {
            "standard": std_type,
            "finetune": ft_type
        }
        
        # Check if models have same structure
        std_modules = list(std_model.named_modules())
        ft_modules = list(ft_model.named_modules())
        
        add_log(f"🔍 Model structure:")
        add_log(f"   Standard modules: {len(std_modules)}")
        add_log(f"   Finetune modules: {len(ft_modules)}")
        
        # Check specific attributes
        important_attrs = ['sample_rate', 'sample_size', 'model_type']
        attr_comparison = {}
        
        for attr in important_attrs:
            std_val = getattr(std_model, attr, "NOT_FOUND")
            ft_val = getattr(ft_model, attr, "NOT_FOUND")
            
            attr_comparison[attr] = {
                "standard": str(std_val),
                "finetune": str(ft_val),
                "match": std_val == ft_val
            }
            
            add_log(f"🔍 Attribute {attr}:")
            add_log(f"   Standard: {std_val}")
            add_log(f"   Finetune: {ft_val}")
            add_log(f"   Match: {std_val == ft_val}")
        
        results["attribute_comparison"] = attr_comparison
        
        # Check configs
        std_config = standard_data["config"]
        ft_config = finetune_data["config"]
        
        config_match = std_config == ft_config
        add_log(f"🔍 Configs identical: {config_match}")
        
        if not config_match:
            add_log("❌ Configs differ - this could be the issue!")
            # Show key differences
            for key in set(std_config.keys()) | set(ft_config.keys()):
                std_val = std_config.get(key, "MISSING")
                ft_val = ft_config.get(key, "MISSING")
                if std_val != ft_val:
                    add_log(f"   Config diff - {key}: std={std_val}, ft={ft_val}")
        
        results["configs_match"] = config_match
        
        # Key insight: Check if we're using the right loading method
        add_log("\n🎯 ANALYSIS:")
        add_log("   Standard model loaded via: get_pretrained_model()")
        add_log("   Finetune model loaded via: create_model_from_config() + manual checkpoint")
        add_log("\n💡 HYPOTHESIS:")
        add_log("   Different loading methods might initialize models differently")
        add_log("   Even with identical weights, initialization/setup could differ")
        
        # Recommendation
        add_log("\n🔧 RECOMMENDED TEST:")
        add_log("   Try loading finetune using get_pretrained_model() approach")
        add_log("   Or try loading standard using create_model_from_config() approach")
        add_log("   This will isolate whether the issue is loading method vs weights")
        
        results["recommendation"] = "test_consistent_loading_method"
        results["success"] = True
        return jsonify(results)
        
    except Exception as e:
        print(f"Debug loading error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "debug_info": results.get("debug_info", [])
        }), 500
    
@app.route('/debug/inference', methods=['GET'])
def debug_inference_params():
    """Debug what parameters the inference function expects for different diffusion objectives"""
    try:
        results = {"debug_info": []}
        
        def add_log(message):
            results["debug_info"].append(message)
            print(message)
        
        add_log("🔍 Analyzing inference parameters for different diffusion objectives...")
        
        # Load both models
        if len(model_manager.model_cache) < 2:
            return jsonify({
                "error": "Both models need to be loaded first. Call /test/finetune to load both.",
                "debug_info": results["debug_info"]
            }), 400
        
        standard_key = "standard_saos"
        finetune_key = None
        
        for key in model_manager.model_cache.keys():
            if key != standard_key:
                finetune_key = key
                break
        
        standard_data = model_manager.model_cache[standard_key]
        finetune_data = model_manager.model_cache[finetune_key]
        
        add_log(f"📊 Model comparison:")
        add_log(f"   Standard: {standard_data['config']['model']['diffusion']['diffusion_objective']}")
        add_log(f"   Finetune: {finetune_data['config']['model']['diffusion']['diffusion_objective']}")
        
        results["diffusion_objectives"] = {
            "standard": standard_data['config']['model']['diffusion']['diffusion_objective'],
            "finetune": finetune_data['config']['model']['diffusion']['diffusion_objective']
        }
        
        # Check if models have different attributes that inference might use
        std_model = standard_data["model"]
        ft_model = finetune_data["model"]
        
        # Look for diffusion_objective attribute on the model itself
        std_obj = getattr(std_model, 'diffusion_objective', 'NOT_FOUND')
        ft_obj = getattr(ft_model, 'diffusion_objective', 'NOT_FOUND')
        
        add_log(f"🔍 Model diffusion_objective attributes:")
        add_log(f"   Standard model.diffusion_objective: {std_obj}")
        add_log(f"   Finetune model.diffusion_objective: {ft_obj}")
        
        results["model_attributes"] = {
            "standard_diffusion_objective": str(std_obj),
            "finetune_diffusion_objective": str(ft_obj)
        }
        
        # Check what sampler types might be appropriate
        add_log("🎯 Recommended inference parameters:")
        
        if finetune_data['config']['model']['diffusion']['diffusion_objective'] == 'rectified_flow':
            add_log("   For rectified_flow model:")
            add_log("     - sampler_type: Try 'euler', 'rk4', or 'midpoint'")
            add_log("     - sigma_min/sigma_max: May need different noise schedule")
            add_log("     - steps: Rectified flow often works with fewer steps")
            
            results["rectified_flow_recommendations"] = {
                "sampler_types": ["euler", "rk4", "midpoint"],
                "note": "Avoid pingpong sampler for rectified flow",
                "fewer_steps": "Rectified flow often works with fewer steps than rf_denoiser"
            }
        
        if standard_data['config']['model']['diffusion']['diffusion_objective'] == 'rf_denoiser':
            add_log("   For rf_denoiser model:")
            add_log("     - sampler_type: 'pingpong' (current)")
            add_log("     - Works with adversarial training parameters")
            
            results["rf_denoiser_recommendations"] = {
                "sampler_type": "pingpong",
                "note": "Current inference likely optimized for this"
            }
        
        # Check generate_diffusion_cond signature
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        import inspect
        
        sig = inspect.signature(generate_diffusion_cond)
        params = list(sig.parameters.keys())
        
        add_log(f"🔧 generate_diffusion_cond parameters:")
        for param in params:
            add_log(f"   - {param}")
        
        results["generation_function_params"] = params
        
        # Key insight
        add_log("\n💡 KEY INSIGHT:")
        add_log("   The generate_diffusion_cond function might be hardcoded for rf_denoiser")
        add_log("   Even if model config says 'rectified_flow', inference uses rf_denoiser methods")
        add_log("   Need to check if function respects model.diffusion_objective")
        
        results["key_insight"] = "generate_diffusion_cond might not respect diffusion_objective from config"
        results["solution"] = "Need to ensure inference method matches training objective"
        
        results["success"] = True
        return jsonify(results)
        
    except Exception as e:
        print(f"Debug inference error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "debug_info": results.get("debug_info", [])
        }), 500

if __name__ == '__main__':
    # Pre-load model on startup
    print("🚀 Starting Enhanced Stable Audio API...")
    try:
        startup_backend = _default_backend_engine()
        print(f"🎛 Startup backend default: {startup_backend}")
        if startup_backend == "mlx":
            standard_spec = _resolve_mlx_model_spec("standard", None, None)
            _record_mlx_model_status("standard", standard_spec.get("source", {}), standard_spec["model_config"], warmed=False)
            _prewarm_mlx_model_spec(standard_spec, "standard")
            _record_mlx_model_status("standard", standard_spec.get("source", {}), standard_spec["model_config"], warmed=True)
            print("✅ MLX warmup pre-load completed")
        else:
            load_model()
            print("✅ Model pre-loaded successfully")
    except Exception as e:
        print(f"❌ Failed to pre-load model: {e}")
        print("Will attempt to load on first request...")
    
    # Run server
    app.run(host='0.0.0.0', port=8005, debug=False)
