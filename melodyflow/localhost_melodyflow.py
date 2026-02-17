from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchaudio
import time
import io
import tempfile
import os
import json
import threading
from audiocraft.models import MelodyFlow
import gc
from variations import VARIATIONS
import psutil
import logging
from contextlib import contextmanager

try:
    import redis
except Exception:
    redis = None  # type: ignore

# In both g4l_localhost.py AND localhost_melodyflow.py
import tempfile

# Use a consistent shared temp directory
SHARED_TEMP_DIR = os.path.join(tempfile.gettempdir(), "gary4juce_shared")
os.makedirs(SHARED_TEMP_DIR, exist_ok=True)

_redis_disabled = os.environ.get("MELODYFLOW_DISABLE_REDIS", "0") == "1"
redis_client = None
if redis is not None and not _redis_disabled:
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

PROGRESS_FILE_PREFIX = "melodyflow_progress_"
PROGRESS_TTL_SECONDS = 3600
_progress_lock = threading.Lock()
_last_progress_percent = {}

def _progress_file_path(session_id: str) -> str:
    safe_session = "".join(ch for ch in str(session_id) if ch.isalnum() or ch in ("-", "_"))
    if not safe_session:
        safe_session = "unknown"
    return os.path.join(SHARED_TEMP_DIR, f"{PROGRESS_FILE_PREFIX}{safe_session}.json")

def _build_queue_status(status: str, message: str, progress_percent: int = None) -> dict:
    payload = {
        "status": status,
        "message": message,
        "position": 0,
        "total_queued": 0,
        "estimated_time": None,
        "estimated_seconds": 0,
        "source": "melodyflow-localhost",
        "phase": "transform",
    }
    if progress_percent is not None:
        payload["progress"] = progress_percent
    return payload

def _write_progress_snapshot(session_id: str, *, progress: int, status: str, message: str, error: str = None) -> None:
    if not session_id:
        return

    queue_status = _build_queue_status(status, message, progress)
    snapshot = {
        "session_id": session_id,
        "progress": progress,
        "status": status,
        "queue_status": queue_status,
        "updated_at": time.time(),
    }
    if error:
        snapshot["error"] = error
        queue_status["error"] = error

    # Always write to shared temp so g4l_localhost can relay progress even without Redis.
    progress_path = _progress_file_path(session_id)
    tmp_path = f"{progress_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        os.replace(tmp_path, progress_path)
    except Exception as e:
        print(f"Progress snapshot write failed: {e}")

    # Best-effort Redis writes for compatibility if Redis is installed/running.
    if redis_client is None:
        return

    try:
        redis_client.setex(f"progress:{session_id}", PROGRESS_TTL_SECONDS, str(progress))
        redis_client.setex(f"status:{session_id}", PROGRESS_TTL_SECONDS, json.dumps({"status": status, "error": error} if error else {"status": status}))
        redis_client.setex(f"queue_status:{session_id}", PROGRESS_TTL_SECONDS, json.dumps(queue_status))
    except Exception as e:
        print(f"Redis progress update failed: {e}")

def _set_progress(session_id: str, progress_percent: int, status: str = "processing", message: str = None, error: str = None) -> None:
    if not session_id:
        return
    progress_percent = max(0, min(100, int(progress_percent)))
    if message is None:
        message = f"transforming... {progress_percent}%"
    _write_progress_snapshot(
        session_id,
        progress=progress_percent,
        status=status,
        message=message,
        error=error,
    )

def init_progress_tracking(session_id: str) -> None:
    if not session_id:
        return
    with _progress_lock:
        _last_progress_percent[session_id] = -1
    _set_progress(session_id, 0, status="processing", message="transforming... 0%")

def finalize_progress_tracking(session_id: str) -> None:
    if not session_id:
        return
    with _progress_lock:
        _last_progress_percent[session_id] = 100
    _set_progress(session_id, 100, status="processing", message="transforming... 100%")

def fail_progress_tracking(session_id: str, error_message: str) -> None:
    if not session_id:
        return
    with _progress_lock:
        _last_progress_percent.pop(session_id, None)
    _write_progress_snapshot(
        session_id,
        progress=0,
        status="failed",
        message=str(error_message),
        error=str(error_message),
    )

# Add progress callback that writes to shared snapshot + Redis (best effort).
def redis_progress_callback(session_id, current, total):
    if not session_id or not total:
        return

    try:
        progress_percent = int((float(current) / float(total)) * 100)
    except Exception:
        return

    # Keep completion ownership in g4l_localhost; it marks completed after WAV is fully relayed.
    progress_percent = max(0, min(99, progress_percent))
    with _progress_lock:
        last_percent = _last_progress_percent.get(session_id)
        if last_percent == progress_percent:
            return
        _last_progress_percent[session_id] = progress_percent

    _set_progress(session_id, progress_percent, status="processing")

app = Flask(__name__)
CORS(app)

# Global variables
model = None
_mlx_runtime = None
_mlx_runtime_model_id = None
_mlx_runtime_module = None

BACKEND_MPS = "mps"
BACKEND_MLX_NATIVE_TORCH_CODEC = "mlx_native_torch_codec"
BACKEND_MLX_NATIVE_MLX_CODEC = "mlx_native_mlx_codec"
_VALID_BACKEND_ENGINES = {
    BACKEND_MPS,
    BACKEND_MLX_NATIVE_TORCH_CODEC,
    BACKEND_MLX_NATIVE_MLX_CODEC,
}


def _normalize_backend_engine(raw_backend) -> str:
    if raw_backend is None:
        return BACKEND_MPS
    backend = str(raw_backend).strip().lower()
    if backend in _VALID_BACKEND_ENGINES:
        return backend
    if backend in ("", "torch"):
        return BACKEND_MPS
    raise ValueError(
        "backend_engine must be one of 'mps', 'mlx_native_torch_codec', or 'mlx_native_mlx_codec'"
    )


def _default_backend_engine() -> str:
    configured = os.environ.get("MELODYFLOW_BACKEND_ENGINE", BACKEND_MPS)
    try:
        return _normalize_backend_engine(configured)
    except ValueError:
        logger.warning(
            "Invalid MELODYFLOW_BACKEND_ENGINE='%s'; falling back to '%s'",
            configured,
            BACKEND_MPS,
        )
        return BACKEND_MPS


def _resolve_backend_engine_from_request(raw_backend) -> str:
    if raw_backend is None:
        return _default_backend_engine()
    if isinstance(raw_backend, str) and raw_backend.strip() == "":
        return _default_backend_engine()
    return _normalize_backend_engine(raw_backend)


def _mlx_dtype_name() -> str:
    raw = os.environ.get("MELODYFLOW_MLX_DTYPE", "float32").strip().lower()
    if raw in {"float16", "bfloat16", "float32"}:
        return raw
    logger.warning("Invalid MELODYFLOW_MLX_DTYPE='%s'; falling back to 'float32'", raw)
    return "float32"


def _mlx_native_prompt_mode() -> str:
    raw = os.environ.get("MELODYFLOW_MLX_NATIVE_PROMPT_MODE", "mean").strip().lower()
    if raw in {"mean", "stochastic"}:
        return raw
    logger.warning(
        "Invalid MELODYFLOW_MLX_NATIVE_PROMPT_MODE='%s'; falling back to 'mean'",
        raw,
    )
    return "mean"


def _require_mlx_runtime_module():
    global _mlx_runtime_module
    if _mlx_runtime_module is None:
        try:
            import melodyflow_mlx_edit as mlx_runtime_module
        except Exception as exc:
            raise RuntimeError(
                "Failed to import MLX runtime support. Ensure MLX dependencies are installed."
            ) from exc
        _mlx_runtime_module = mlx_runtime_module
    return _mlx_runtime_module


def _get_mlx_runtime(current_model, *, need_native_codec: bool):
    global _mlx_runtime
    global _mlx_runtime_model_id

    mlx_module = _require_mlx_runtime_module()
    runtime_model_id = id(current_model)
    if _mlx_runtime is None or _mlx_runtime_model_id != runtime_model_id:
        _mlx_runtime = mlx_module.build_runtime(current_model, dtype_name=_mlx_dtype_name())
        _mlx_runtime_model_id = runtime_model_id

    if need_native_codec:
        mlx_module.ensure_native_codec(_mlx_runtime, current_model)

    return _mlx_runtime, mlx_module


def _pick_device() -> str:
    forced_device = os.environ.get("MELODYFLOW_DEVICE", "").strip().lower()
    require_mps = os.environ.get("MELODYFLOW_REQUIRE_MPS", "0") == "1"

    if forced_device:
        if forced_device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("MELODYFLOW_DEVICE=mps but MPS is not available.")
        if forced_device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("MELODYFLOW_DEVICE=cuda but CUDA is not available.")
        if forced_device == "cpu":
            return "cpu"
        raise RuntimeError(f"Unsupported MELODYFLOW_DEVICE='{forced_device}'.")

    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    if require_mps:
        raise RuntimeError("MELODYFLOW_REQUIRE_MPS=1 but no MPS device is available.")
    return 'cpu'

device = _pick_device()
DEVICE = device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('melodyflow')
logger.info("Default MelodyFlow backend engine: %s", _default_backend_engine())

from flask import after_this_request

def unload_model():
    global model
    global _mlx_runtime
    global _mlx_runtime_model_id
    if model is not None:
        try:
            del model
        except:
            pass
        model = None
    _mlx_runtime = None
    _mlx_runtime_model_id = None
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except:
            pass
        torch.cuda.empty_cache()
        # Optional extra cleanup on some systems:
        try:
            torch.cuda.ipc_collect()
        except:
            pass
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except:
            pass
        try:
            torch.mps.empty_cache()
        except:
            pass
    gc.collect()

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code

@contextmanager
def resource_cleanup():
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
            except:
                pass
            try:
                torch.mps.empty_cache()
            except:
                pass
        gc.collect()

def load_model():
    """Initialize the MelodyFlow model."""
    global model
    if model is None:
        print("Loading MelodyFlow model...")
        model = MelodyFlow.get_pretrained('facebook/melodyflow-t24-30secs', device=DEVICE)
    return model

def load_audio_from_file(file_path: str, target_sr: int = 32000) -> torch.Tensor:
    """Load and preprocess audio from file path."""
    try:
        if not os.path.exists(file_path):
            raise AudioProcessingError(f"Audio file not found: {file_path}")
        
        waveform, sr = torchaudio.load(file_path)

        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform.unsqueeze(0).to(device)

    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio from file: {str(e)}")

def find_max_duration(model: MelodyFlow, waveform: torch.Tensor, sr: int = 32000, max_token_length: int = 750) -> tuple:
    """Binary search to find maximum duration that produces tokens under the limit."""
    min_seconds = 1.0
    max_seconds = waveform.shape[-1] / sr
    best_duration = min_seconds
    best_tokens = None

    if max_seconds <= min_seconds:
        tokens = model.encode_audio(waveform)
        return max_seconds, tokens

    while max_seconds - min_seconds > 0.1:
        mid_seconds = (min_seconds + max_seconds) / 2
        samples = max(1, int(mid_seconds * sr))
        test_waveform = waveform[..., :samples]

        try:
            tokens = model.encode_audio(test_waveform)
            token_length = tokens.shape[-1]

            if token_length <= max_token_length:
                best_duration = mid_seconds
                best_tokens = tokens
                min_seconds = mid_seconds
            else:
                max_seconds = mid_seconds

        except Exception as e:
            max_seconds = mid_seconds

    if best_tokens is None:
        fallback_samples = max(1, int(min_seconds * sr))
        best_tokens = model.encode_audio(waveform[..., :fallback_samples])
        best_duration = min_seconds

    if int(best_tokens.shape[-1]) > max_token_length:
        raise RuntimeError(
            f"Prompt token length {int(best_tokens.shape[-1])} exceeds max_token_length={max_token_length}"
        )

    return best_duration, best_tokens


def _normalize_waveform_for_save(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor
    if x.dim() >= 3:
        x = x[0]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2:
        raise RuntimeError(f"Expected output waveform with 2 dims [C,T], got shape={tuple(x.shape)}")
    return x

def process_audio(
    waveform: torch.Tensor,
    variation_name: str,
    custom_flowstep: float = None,
    solver: str = "euler",
    custom_prompt: str = None,
    session_id: str = None,
    progress_callback=None,
    backend_engine: str = None,
) -> torch.Tensor:
    """Process audio with selected variation."""

    try:
        if variation_name not in VARIATIONS:
            raise AudioProcessingError(f"Unknown variation: {variation_name}")

        backend = _resolve_backend_engine_from_request(backend_engine)
        config = VARIATIONS[variation_name].copy()
        flowstep = custom_flowstep if custom_flowstep is not None else config['default_flowstep']

        if custom_prompt is not None:
            config['prompt'] = custom_prompt

        with resource_cleanup():
            current_model = load_model()

            codec_mode = "native" if backend == BACKEND_MLX_NATIVE_MLX_CODEC else "torch"

            if backend == BACKEND_MPS:
                max_valid_duration, tokens = find_max_duration(current_model, waveform)
            else:
                need_native_codec = codec_mode == "native"
                mlx_runtime, mlx_module = _get_mlx_runtime(
                    current_model,
                    need_native_codec=need_native_codec,
                )
                if codec_mode == "native":
                    max_valid_duration, tokens = mlx_module.find_max_duration_with_encoder(
                        waveform,
                        encode_audio_fn=lambda w: mlx_module.encode_audio_with_mlx_codec(
                            mlx_runtime,
                            w,
                            current_model.device,
                        ),
                        sr=current_model.sample_rate,
                        max_token_length=750,
                    )
                else:
                    max_valid_duration, tokens = mlx_module.find_max_duration_with_encoder(
                        waveform,
                        encode_audio_fn=lambda w: current_model.encode_audio(w.to(current_model.device)),
                        sr=current_model.sample_rate,
                        max_token_length=750,
                    )

            config['duration'] = max_valid_duration

            if solver.lower() == "midpoint":
                steps = 64
                use_regularize = False
            else:
                steps = config['steps']
                use_regularize = True

            current_model.set_generation_params(
                solver=solver.lower(),
                steps=steps,
                duration=config['duration'],
            )

            current_model.set_editing_params(
                solver=solver.lower(),
                steps=steps,
                target_flowstep=flowstep,
                regularize=use_regularize,
                regularize_iters=2 if use_regularize else 0,
                keep_last_k_iters=1 if use_regularize else 0,
                lambda_kl=0.2 if use_regularize else 0.0,
            )

            model_progress_callback = None
            if progress_callback and session_id:
                def model_progress_callback(elapsed_steps: int, total_steps: int):
                    progress_callback(session_id, elapsed_steps, total_steps)

            if backend == BACKEND_MPS:
                if model_progress_callback is not None:
                    current_model._progress_callback = model_progress_callback
                edited_audio = current_model.edit(
                    prompt_tokens=tokens,
                    descriptions=[config['prompt']],
                    src_descriptions=[""],
                    progress=True,
                    return_tokens=True
                )
                return _normalize_waveform_for_save(edited_audio[0][0])

            native_prompt_mode = _mlx_native_prompt_mode()
            edited_audio = mlx_module.edit_with_mlx(
                current_model,
                mlx_runtime,
                prompt_tokens=tokens,
                prompt_text=config['prompt'],
                src_prompt_text="",
                codec_mode=codec_mode,
                native_prompt_mode=native_prompt_mode,
                progress_callback=model_progress_callback,
            )
            return _normalize_waveform_for_save(edited_audio)

    except AudioProcessingError:
        raise
    except Exception as e:
        raise AudioProcessingError(f"Failed to process audio: {str(e)}", status_code=500)

@app.route('/transform', methods=['POST'])
def transform_audio():
    """Handle audio transformation requests - simplified for localhost."""
    output_file_path = None  # <-- declare early so cleanup hook can see it

    try:
        # Aggressive localhost cleanup: unload model after each request
        AGGRESSIVE_UNLOAD = os.environ.get("MELODYFLOW_UNLOAD_EACH_REQUEST", "1") == "1"

        if AGGRESSIVE_UNLOAD:
            @after_this_request
            def _cleanup(response):
                # Remove generated WAV
                try:
                    if output_file_path and os.path.exists(output_file_path):
                        os.remove(output_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete output wav: {e}")

                # Unload model + clear GPU
                unload_model()
                return response

        session_id = None
        request_backend_engine = None

        # ---------------------------------------------------------------------
        # Handle both multipart form-data and JSON
        # ---------------------------------------------------------------------
        if request.content_type and 'multipart/form-data' in request.content_type:
            # File upload mode
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No audio file selected'}), 400
            
            variation = request.form.get('transformation_type', request.form.get('variation'))
            custom_flowstep = request.form.get('flowstep')
            solver = request.form.get('solver', 'euler')
            custom_prompt = request.form.get('prompt', request.form.get('custom_prompt'))
            session_id = request.form.get('session_id')
            request_backend_engine = request.form.get('backend_engine')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_file.save(tmp_file.name)
                temp_file_path = tmp_file.name
            
            try:
                input_waveform = load_audio_from_file(temp_file_path)
            finally:
                os.unlink(temp_file_path)

        else:
            # JSON mode
            data = request.get_json()
            if not data or 'variation' not in data:
                return jsonify({'error': 'Missing required data'}), 400
            
            variation = data['variation']
            custom_flowstep = data.get('flowstep')
            solver = data.get('solver', 'euler')
            custom_prompt = data.get('custom_prompt')
            session_id = data.get('session_id')
            request_backend_engine = data.get('backend_engine')

            if 'audio_file_path' in data and data['audio_file_path']:
                audio_file_path = data['audio_file_path']
                input_waveform = load_audio_from_file(audio_file_path)
            else:
                return jsonify({'error': 'No audio data provided'}), 400

        # ---------------------------------------------------------------------
        # Validate parameters
        # ---------------------------------------------------------------------
        if custom_flowstep is not None:
            try:
                custom_flowstep = float(custom_flowstep)
                if custom_flowstep <= 0:
                    return jsonify({'error': 'Flowstep must be positive'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid flowstep value'}), 400
        
        if solver not in ['euler', 'midpoint']:
            return jsonify({'error': 'Invalid solver. Must be "euler" or "midpoint"'}), 400

        # Publish transform progress snapshots for g4l_localhost poll relay.
        if session_id:
            init_progress_tracking(session_id)

        # ---------------------------------------------------------------------
        # Process audio
        # ---------------------------------------------------------------------
        processed_waveform = process_audio(
            input_waveform,
            variation,
            custom_flowstep,
            solver,
            custom_prompt,
            session_id=session_id,
            progress_callback=redis_progress_callback,
            backend_engine=request_backend_engine,
        )

        import uuid
        output_filename = f"output_{session_id}_{uuid.uuid4().hex[:8]}.wav"
        output_file_path = os.path.join(SHARED_TEMP_DIR, output_filename)

        torchaudio.save(output_file_path, processed_waveform.cpu(), 32000)

        # Explicit tensor cleanup (before model unload)
        del input_waveform
        del processed_waveform

        if session_id:
            finalize_progress_tracking(session_id)

        return send_file(
            output_file_path,
            as_attachment=True,
            download_name='transformed_audio.wav',
            mimetype='audio/wav'
        )

    except AudioProcessingError as e:
        if session_id:
            fail_progress_tracking(session_id, str(e))
        return jsonify({'error': str(e)}), e.status_code

    except Exception as e:
        if session_id:
            fail_progress_tracking(session_id, str(e))
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/variations', methods=['GET'])
def get_variations():
    """Return list of available variations."""
    try:
        variations_list = list(VARIATIONS.keys())
        variations_with_details = {
            name: {
                'prompt': VARIATIONS[name]['prompt'],
                'flowstep': VARIATIONS[name]['default_flowstep']
            } for name in variations_list
        }
        return jsonify({
            'variations': variations_with_details
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch variations: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        model_loaded = model is not None
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        accelerator = 'cuda' if cuda_available else 'mps' if mps_available else 'cpu'
        backend_engine = _default_backend_engine()
        requires_mps = backend_engine == BACKEND_MPS
        accelerator_available = accelerator != 'cpu'
        backend_ready = mps_available if requires_mps else True
        
        status = {
            'status': 'healthy' if accelerator_available and backend_ready else 'degraded',
            'accelerator': accelerator,
            'cuda_available': cuda_available,
            'mps_available': mps_available,
            'model_loaded': model_loaded,
            'backend_engine': backend_engine,
            'backend_requires_mps': requires_mps,
        }
        
        return jsonify(status), 200 if status['status'] == 'healthy' else 503
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=False)
