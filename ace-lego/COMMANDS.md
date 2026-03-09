# ACE-Step Commands

## Container

```bash
# Text2music (turbo, default)
docker run --gpus all -p 8001:8001 \
  -v /home/kev/ace/checkpoints:/app/checkpoints \
  -v /home/kev/ace/data:/app/data \
  ace-step-spark:latest

# Lego / repaint / cover (base model required)
docker run --gpus all -p 8001:8001 \
  -v /home/kev/ace/checkpoints:/app/checkpoints \
  -v /home/kev/ace/data:/app/data \
  -e ACESTEP_CONFIG_PATH=acestep-v15-base \
  ace-step-spark:latest
```

> First boot downloads models to `/home/kev/ace/checkpoints/` — persisted after that.

---

## Health

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

---

## Text2Music

```bash
# Submit
curl -s -X POST http://localhost:8001/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "fingerpicked acoustic guitar, soft Rhodes piano, indie folk, warm",
    "bpm": 90,
    "key_scale": "A minor",
    "audio_duration": 30,
    "thinking": false,
    "audio_format": "mp3"
  }' | tee /tmp/task.json

# Poll (repeat until status=1)
TASK_ID=$(cat /tmp/task.json | jq -r '.data.task_id')
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" | jq '.data[0].status'

# Download
curl -o output.mp3 "http://localhost:8001/v1/audio/<filename>.mp3"
```

**Key parameters:**
| Field | Example | Notes |
|-------|---------|-------|
| `prompt` | `"lo-fi hip hop, Rhodes, dusty drums"` | Describe vibe + instrumentation, not key/BPM |
| `bpm` | `90` | Integer |
| `key_scale` | `"A minor"`, `"C major"`, `"A# major"` | Separate conditioning, not in prompt |
| `audio_duration` | `30` | Seconds |
| `thinking` | `false` | DiT-only, faster. `true` requires LLM (ACESTEP_INIT_LLM=true) |
| `audio_format` | `"mp3"` / `"wav"` / `"flac"` | |
| `batch_size` | `2` | Returns multiple candidates |

---

## Lego (add a stem to existing audio)

Requires `ACESTEP_CONFIG_PATH=acestep-v15-base`.
Audio is uploaded as multipart — no absolute paths in JSON.

**Important — always set these for lego:**
- `inference_steps=50` (default 8 is for turbo; base model needs 50)
- `thinking=false` — **critical**: when `thinking=true`, the LM generates audio semantic
  codes which force `is_cover_task=True` internally, conflicting with the lego repainting
  mask. The DiT receives contradictory conditioning and outputs garbled noise. Lego mode
  does not use LM codes by design; always disable it.
- `bpm`, `key_scale`, `time_signature` explicitly — without them there is no rhythm anchor.
- `audio_duration` — set this to match your source audio length. Without it the LM
  estimates duration and will truncate longer files (e.g. a 2:27 clip came back as 1:36).

```bash
# Submit
curl -s -X POST http://localhost:8001/release_task \
  -F "task_type=lego" \
  -F "ctx_audio=@/home/kev/ace/asharp_89bpm_1.wav" \
  -F "track_name=drums" \
  -F "caption=live acoustic drum kit, tight kick and snare, brushed hi-hats" \
  -F "bpm=89" \
  -F "key_scale=A# major" \
  -F "time_signature=4" \
  -F "inference_steps=50" \
  -F "thinking=false" \
  -F "audio_duration=20" \
  -F "repainting_start=0.0" \
  -F "repainting_end=-1" \
  -F "batch_size=2" | tee /tmp/task.json

# Poll
TASK_ID=$(cat /tmp/task.json | jq -r '.data.task_id')
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" | jq '.data[0].status'

# Download (saves each file as UUID.mp3 in current dir)
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" \
  | jq -r '.data[0].result | fromjson | .[].file' \
  | while IFS= read -r path; do
      fname=$(echo "$path" | awk -F'%2F' '{print $NF}')
      curl -o "$fname" "http://localhost:8001${path}"
    done
```

**Supported track names:** `drums` `bass` `guitar` `piano` `strings` `synth`
`keyboard` `percussion` `brass` `woodwinds` `vocals` `backing_vocals`

---

## Repaint (regenerate a time segment)

```bash
curl -s -X POST http://localhost:8001/release_task \
  -F "task_type=repaint" \
  -F "ctx_audio=@/path/to/source.wav" \
  -F "caption=your description" \
  -F "repainting_start=8.0" \
  -F "repainting_end=16.0" \
  -F "bpm=120" \
  -F "batch_size=2" | tee /tmp/task.json
```

---

## Cover

```bash
curl -s -X POST http://localhost:8001/release_task \
  -F "task_type=cover" \
  -F "ctx_audio=@/path/to/source.wav" \
  -F "caption=jazz piano trio arrangement" \
  -F "batch_size=2" | tee /tmp/task.json
```

---

## One-liner poll + download

```bash
# Wait for job and download all audio files
TASK_ID="paste-task-id-here"
while true; do
  STATUS=$(curl -s -X POST http://localhost:8001/query_result \
    -H "Content-Type: application/json" \
    -d "{\"task_id_list\": [\"$TASK_ID\"]}" | jq -r '.data[0].status')
  [ "$STATUS" = "1" ] && break
  [ "$STATUS" = "2" ] && echo "FAILED" && exit 1
  printf "."; sleep 3
done
echo " done"
curl -s -X POST http://localhost:8001/query_result \
  -H "Content-Type: application/json" \
  -d "{\"task_id_list\": [\"$TASK_ID\"]}" \
  | jq -r '.data[0].result | fromjson | .[].file' \
  | while IFS= read -r path; do
      fname=$(echo "$path" | awk -F'%2F' '{print $NF}')
      curl -o "$fname" "http://localhost:8001${path}"
    done
```
