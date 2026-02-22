#!/usr/bin/env python3
"""
Enhanced model loading system for Stable Audio API with finetune support
Handles both standard SAOS and custom finetunes with smart memory management
NOW SUPPORTS AUTO-DETECTION OF VERSIONED CONFIGS FROM CHECKPOINT NAMES
FIXED: Proper config priority (HF repo first) and EMA weight handling
"""

import os
import json
import re
import threading
import gc
import torch
from contextlib import contextmanager
from pathlib import Path
from huggingface_hub import login, hf_hub_download
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, copy_state_dict

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def _temporary_env(overrides):
    previous = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _hf_download_prefer_local(repo_id: str, filename: str, *, repo_type: str = "model", force_download: bool = False):
    if not force_download:
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_files_only=True,
            )
        except Exception:
            pass

    attempts = [
        {},
        {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"},
    ]
    errors = []
    for index, env_profile in enumerate(attempts, start=1):
        try:
            kwargs = {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "force_download": force_download,
            }
            if env_profile:
                with _temporary_env(env_profile):
                    return hf_hub_download(**kwargs)
            return hf_hub_download(**kwargs)
        except Exception as exc:
            errors.append(f"attempt {index} {env_profile or {'default': True}}: {exc}")

    raise RuntimeError(
        f"failed to download {repo_id}/{filename}: {' | '.join(errors)}"
    )


def _evict_cached_hf_file(path: str):
    path_obj = Path(path)
    candidates = [path_obj]
    try:
        if path_obj.is_symlink():
            candidates.append(path_obj.resolve())
    except Exception:
        pass

    seen = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            if candidate.is_symlink() or candidate.exists():
                candidate.unlink()
        except Exception:
            pass


def _load_finetune_state_dict(path: str):
    torch_error = None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(state_dict, dict):
            return state_dict
    except TypeError:
        try:
            payload = torch.load(path, map_location="cpu")
            state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            if isinstance(state_dict, dict):
                return state_dict
        except Exception as exc:
            torch_error = exc
    except Exception as exc:
        torch_error = exc

    ckpt_error = None
    try:
        payload = load_ckpt_state_dict(path)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(state_dict, dict):
            return state_dict
    except Exception as exc:
        ckpt_error = exc

    raise RuntimeError(
        "unable to parse finetune checkpoint"
        + (f" | torch.load: {torch_error}" if torch_error else "")
        + (f" | load_ckpt_state_dict: {ckpt_error}" if ckpt_error else "")
    )

class ModelManager:
    """Smart model cache with memory management for multiple model support"""
    
    def __init__(self, max_models=2):
        self.model_cache = {}
        self.model_lock = threading.Lock()
        self.max_models = max_models
        self.model_usage_order = []  # Track usage for LRU eviction
        self.active_model_key = None
        
    @contextmanager
    def resource_cleanup(self):
        """Context manager to ensure proper cleanup of GPU resources."""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                except Exception:
                    pass
            gc.collect()
    
    def _evict_lru_model(self):
        """Evict least recently used model to free memory"""
        if len(self.model_cache) < self.max_models:
            return
        
        # Find least recently used model
        for model_key in self.model_usage_order:
            if model_key in self.model_cache:
                print(f"🧹 Evicting LRU model: {model_key}")
                del self.model_cache[model_key]
                self.model_usage_order.remove(model_key)
                
                # Force cleanup
                with self.resource_cleanup():
                    pass
                break
    
    def _update_usage(self, model_key):
        """Update model usage order for LRU tracking"""
        if model_key in self.model_usage_order:
            self.model_usage_order.remove(model_key)
        self.model_usage_order.append(model_key)
    
    def _detect_config_from_checkpoint(self, checkpoint_name, finetune_repo):
        """
        Auto-detect the matching config file from checkpoint filename.
        
        Supports multiple patterns:
        - acid_v4_saos_e22_s16896.ckpt → acid_v4_saos_e22_s16896_model_config.json (v4 style)
        - acid_v1_saos_e21_s33000.ckpt → acid_v1_base_model_config.json (v1-v3 style)
        - acid_v2_*.ckpt → acid_v2_config.json (flexible fallback)
        - am_saos1_e18_s4800.ckpt → fallback to repo/base config
        
        FIXED: Now prioritizes HF repo over local file
        
        Returns: config_path (local or downloaded), or None if should use base_repo
        """
        print(f"🔍 Detecting config for checkpoint: {checkpoint_name}")
        
        # Extract version prefix for pattern matching
        version_match = re.match(r'(acid_v\d+)_', checkpoint_name)
        
        if version_match:
            version_prefix = version_match.group(1)
            print(f"   ✅ Detected version prefix: {version_prefix}")
            
            # Try multiple naming patterns in order of specificity
            checkpoint_base = checkpoint_name.replace('.ckpt', '')
            
            patterns_to_try = [
                # Pattern 1: Exact checkpoint name + _model_config.json (for v4)
                f"{checkpoint_base}_model_config.json",
                
                # Pattern 2: Version prefix + _base_model_config.json (for v1-v3)
                f"{version_prefix}_base_model_config.json",
                
                # Pattern 3: Version prefix + _model_config.json (variant)
                f"{version_prefix}_model_config.json",
                
                # Pattern 4: Just version + _config.json (most flexible)
                f"{version_prefix}_config.json",
            ]
            
            for i, pattern in enumerate(patterns_to_try, 1):
                try:
                    print(f"   🔍 Try {i}/{len(patterns_to_try)}: {pattern}")
                    config_path = _hf_download_prefer_local(
                        repo_id=finetune_repo,
                        filename=pattern,
                    )
                    print(f"   ✅ Config found and downloaded: {pattern}")
                    return config_path
                except Exception as e:
                    print(f"   ❌ Not found: {pattern}")
                    continue
            
            print(f"   ⚠️  No versioned config found for {version_prefix}")
        
        # FIXED: Fallback 1 - Try both naming conventions
        for config_name in ["base_model_config.json", "model_config.json"]:
            try:
                print(f"   🔍 Trying {config_name} from {finetune_repo}...")
                config_path = _hf_download_prefer_local(
                    repo_id=finetune_repo,
                    filename=config_name,
                )
                print(f"   ✅ Config downloaded from HF repo: {config_path}")
                return config_path
            except Exception as e:
                print(f"   ⚠️  No {config_name} in finetune repo")
                continue
        
        # Fallback 2: Check for local base_model_config.json (backward compat)
        local_config = os.path.abspath("./base_model_config.json")
        if os.path.exists(local_config):
            print(f"   ⚠️  Using local config as fallback: {local_config}")
            return local_config
        
        # Fallback 3: Return None to signal using base_repo config
        print(f"   ⚠️  No config detected, will use base_repo config")
        return None
    
    def _fetch_prompts_for_repo(self, repo_id: str, checkpoint: str | None = None):
        from huggingface_hub import hf_hub_download
        import json
        try:
            ppath = hf_hub_download(repo_id=repo_id, filename="prompts.json")
            with open(ppath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Optional: stamp what we loaded
            if checkpoint and isinstance(data, dict):
                data.setdefault("finetune_repo", repo_id)
                data.setdefault("checkpoint", checkpoint)
            return data
        except Exception:
            return None

    def get_model(self, model_spec="standard"):
        """Get model from cache or load it"""
        with self.model_lock:
            # Cache key
            if model_spec == "standard":
                cache_key = "standard_saos"
            else:
                base_repo_key = model_spec.get('base_repo', 'stabilityai/stable-audio-open-small').replace('/', '_')
                repo_key = model_spec['repo'].replace('/', '_')
                ckpt_key = model_spec['checkpoint'].replace('/', '_').replace('.', '_')
                cache_key = f"finetune_{repo_key}_{ckpt_key}"

            # Cached?
            if cache_key in self.model_cache:
                print(f"🎯 Using cached model: {cache_key}")
                self._update_usage(cache_key)
                self.active_model_key = cache_key
                return self.model_cache[cache_key]

            # Evict LRU if needed
            self._evict_lru_model()
            print(f"🔄 Loading model: {cache_key}")

            try:
                if model_spec == "standard":
                    model_data = self._load_standard_model()
                else:
                    model_data = self._load_finetune_model(model_spec)

                self.model_cache[cache_key] = model_data
                self._update_usage(cache_key)
                self.active_model_key = cache_key

                print(f"✅ Model loaded and cached: {cache_key}")
                print(f"📊 Cache status: {len(self.model_cache)}/{self.max_models} models loaded")
                return model_data

            except Exception as e:
                print(f"❌ Failed to load model {cache_key}: {e}")
                raise
    
    def _load_standard_model(self):
        """Load standard SAOS model (post-adversarial training)"""
        import os, torch
        from huggingface_hub import login

        # HF auth if provided
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            login(token=hf_token)

        model, config = get_pretrained_model("stabilityai/stable-audio-open-small")

        device = pick_device()
        model = model.to(device)
        if device in ("cuda", "mps"):
            model = model.half()
        model.eval()

        obj = getattr(model, "diffusion_objective", None)
        print(f"✅ Standard SAOS ready | objective: {obj} | device: {device}")
        print(f"   Sample rate: {config.get('sample_rate')}")
        print(f"   Sample size: {config.get('sample_size')}")

        return {
            "model": model,
            "config": config,
            "device": device,
            "type": "standard",
            "source": "stabilityai/stable-audio-open-small",
            "base_repo": "stabilityai/stable-audio-open-small",
        }
    
    def _load_finetune_model(self, model_spec):
        """
        Load a finetune with smart config detection and proper EMA handling:
        - Auto-detects versioned configs (acid_v1, acid_v2, etc.)
        - Prioritizes HF repo config over local config
        - Handles EMA weights properly (diffusion_ema.ema_model.* → diffusion.model.*)
        - Loads base weights first, then overlays finetune (excluding pretransform)
        
        FIXED: Added EMA weight remapping from test script
        """
        import os, json
        import torch
        from huggingface_hub import hf_hub_download, login

        repo = model_spec["repo"]
        checkpoint_name = model_spec["checkpoint"]
        base_repo = model_spec.get("base_repo", "stabilityai/stable-audio-open-small")

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        try:
            # Smart config detection (with fixed priority)
            config_path = self._detect_config_from_checkpoint(checkpoint_name, repo)
            
            if config_path is None:
                # Fallback: use base_repo config
                print(f"📥 Downloading config from base_repo: {base_repo}")
                config_path = _hf_download_prefer_local(
                    repo_id=base_repo,
                    filename="base_model_config.json",
                )

            print(f"📥 Downloading base_model.ckpt from {base_repo}")
            base_ckpt_path = _hf_download_prefer_local(
                repo_id=base_repo,
                filename="base_model.ckpt",
            )

            print(f"📥 Downloading finetune checkpoint {checkpoint_name} from {repo}")
            ft_ckpt_path = _hf_download_prefer_local(
                repo_id=repo,
                filename=checkpoint_name,
            )

            # Build model from config
            with open(config_path, "r") as f:
                config = json.load(f)

            print("🔧 Creating model from config…")
            print(f"   Config path: {config_path}")
            print(f"   Config keys: {list(config.keys())}")
            model = create_model_from_config(config)

            # Load base weights using copy_state_dict (like test script)
            print("🎯 Loading base_model.ckpt using copy_state_dict")
            try:
                base_sd = load_ckpt_state_dict(base_ckpt_path)
            except Exception as base_load_error:
                print(f"⚠️ base checkpoint parse failed, forcing re-download: {base_load_error}")
                _evict_cached_hf_file(base_ckpt_path)
                base_ckpt_path = _hf_download_prefer_local(
                    repo_id=base_repo,
                    filename="base_model.ckpt",
                    force_download=True,
                )
                base_sd = load_ckpt_state_dict(base_ckpt_path)
            copy_state_dict(model, base_sd)
            print(f"   ✅ Base weights loaded")

            # Load finetune checkpoint
            print("🎯 Loading finetune checkpoint...")
            try:
                state_dict = _load_finetune_state_dict(ft_ckpt_path)
            except Exception as ft_load_error:
                print(f"⚠️ finetune checkpoint parse failed, forcing re-download: {ft_load_error}")
                _evict_cached_hf_file(ft_ckpt_path)
                ft_ckpt_path = _hf_download_prefer_local(
                    repo_id=repo,
                    filename=checkpoint_name,
                    force_download=True,
                )
                state_dict = _load_finetune_state_dict(ft_ckpt_path)
            
            # FIXED: Handle EMA weights (from test script)
            ema_keys = [k for k in state_dict.keys() if k.startswith("diffusion_ema.ema_model.")]
            
            if ema_keys:
                print(f"   🔍 Found EMA weights ({len(ema_keys)} keys)")
                # Create new state dict with EMA weights, stripping the prefix
                ema_state = {}
                for key in ema_keys:
                    # Strip "diffusion_ema.ema_model." prefix to get "diffusion.model.*"
                    new_key = key.replace("diffusion_ema.ema_model.", "diffusion.model.")
                    ema_state[new_key] = state_dict[key]
                
                # Also include pretransform weights from base (not EMA'd)
                print(f"   🔍 Keeping pretransform weights from base model")
                for key in state_dict.keys():
                    if key.startswith("diffusion.pretransform"):
                        ema_state[key] = state_dict[key]
                
                # Use EMA state
                print(f"   ✅ Using {len(ema_state)} EMA-remapped weights")
                copy_state_dict(model, ema_state)
            else:
                print(f"   ℹ️  No EMA weights found, using regular state_dict")
                # Filter out pretransform to preserve base model's pretransform
                filtered_state = {k: v for k, v in state_dict.items() 
                                if not k.startswith("pretransform.")}
                print(f"   ✅ Using {len(filtered_state)} regular weights (excluding pretransform)")
                copy_state_dict(model, filtered_state)

            # Device, dtype, eval()
            device = pick_device()
            model = model.to(device)
            if device in ("cuda", "mps"):
                model = model.half()
            model.eval()

            obj = getattr(model, "diffusion_objective", None)
            print(f"✅ Finetune model ready | objective: {obj} | device: {device}")
            print(f"   Config: {os.path.basename(config_path)}")
            print(f"   Checkpoint: {checkpoint_name}")
            print(f"   Sample rate: {config.get('sample_rate')}")
            print(f"   Sample size: {config.get('sample_size')}")

            prompts = self._fetch_prompts_for_repo(repo_id=repo, checkpoint=checkpoint_name)

            return {
                "model": model,
                "config": config,
                "device": device,
                "type": "finetune",
                "source": f"{repo}/{checkpoint_name}",
                "config_source": config_path,
                "base_source": base_repo,
                "repo": repo,
                "checkpoint": checkpoint_name,
                "base_repo": base_repo,
                "prompts": prompts,  # <-- add this
            }

        except Exception as e:
            print(f"❌ Error loading finetune: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def list_loaded_models(self):
        """Get info about currently loaded models"""
        with self.model_lock:
            return {
                "loaded_models": list(self.model_cache.keys()),
                "usage_order": self.model_usage_order.copy(),
                "cache_utilization": f"{len(self.model_cache)}/{self.max_models}"
            }
    
    def clear_cache(self):
        """Clear all cached models"""
        with self.model_lock:
            print("🧹 Clearing all cached models...")
            self.model_cache.clear()
            self.model_usage_order.clear()
            self.active_model_key = None
            with self.resource_cleanup():
                pass
            print("✅ Model cache cleared")

# Global model manager instance
model_manager = ModelManager(max_models=2)

def load_model(model_type="standard", finetune_repo=None, finetune_checkpoint=None, base_repo=None):
    """
    Load model using the enhanced model manager with auto-config detection
    
    Args:
        model_type: "standard" or "finetune"
        finetune_repo: HF repo for finetune (e.g., "thepatch/jerry_grunge")
        finetune_checkpoint: Checkpoint filename (e.g., "jerry_encoded_bs128_HARD_epoch=19-step=60.ckpt")
        base_repo: Base repo for base weights (optional, defaults to "stabilityai/stable-audio-open-small")
    
    Returns:
        Tuple of (model, config, device)
    
    Note:
        Config is now auto-detected with proper priority:
        1. Versioned configs from checkpoint name (acid_v1_*, acid_v2_*, etc.)
        2. base_model_config.json from HuggingFace repo
        3. Local base_model_config.json (backward compat)
        4. Base repo config
        
        Also handles EMA weights properly (diffusion_ema.ema_model.* → diffusion.model.*)
    """
    if model_type == "standard":
        model_spec = "standard"
    elif model_type == "finetune":
        if not finetune_repo or not finetune_checkpoint:
            raise ValueError("finetune_repo and finetune_checkpoint required for finetune models")
        model_spec = {
            "type": "finetune",
            "repo": finetune_repo,
            "checkpoint": finetune_checkpoint,
            "base_repo": base_repo or "stabilityai/stable-audio-open-small"
        }
    else:
        raise ValueError("model_type must be 'standard' or 'finetune'")
    
    model_data = model_manager.get_model(model_spec)
    return model_data["model"], model_data["config"], model_data["device"]

# Test function
def test_versioned_checkpoints():
    """Test loading different versioned checkpoints"""
    print("🧪 Testing versioned checkpoint loading...\n")
    
    test_cases = [
        {
            "name": "jerry_grunge (NEW: Fixed config + EMA)",
            "repo": "thepatch/jerry_grunge",
            "checkpoint": "jerry_encoded_bs128_HARD_epoch=19-step=60.ckpt"
        },
        {
            "name": "acid_v1",
            "repo": "S3Sound/acid_saos",
            "checkpoint": "acid_v1_saos_e21_s33000.ckpt"
        },
        {
            "name": "acid_v4",
            "repo": "S3Sound/acid_saos",
            "checkpoint": "acid_v4_saos_e22_s16896.ckpt"
        }
    ]
    
    for test in test_cases:
        try:
            print(f"\n{'='*60}")
            print(f"Testing: {test['name']}")
            print(f"{'='*60}")
            
            model, config, device = load_model(
                model_type="finetune",
                finetune_repo=test["repo"],
                finetune_checkpoint=test["checkpoint"]
            )
            
            print(f"✅ {test['name']} loaded successfully!")
            print(f"   Device: {device}")
            print(f"   Sample rate: {config['sample_rate']}")
            print(f"   Sample size: {config['sample_size']}")
            print(f"   Max duration: {config['sample_size'] / config['sample_rate']:.1f}s")
            
        except Exception as e:
            print(f"❌ {test['name']} failed: {e}")
    
    # Show cache status
    print(f"\n{'='*60}")
    print("Final Cache Status:")
    print(f"{'='*60}")
    cache_info = model_manager.list_loaded_models()
    print(f"Loaded models: {cache_info['loaded_models']}")
    print(f"Cache utilization: {cache_info['cache_utilization']}")

if __name__ == "__main__":
    # Run test
    test_versioned_checkpoints()
