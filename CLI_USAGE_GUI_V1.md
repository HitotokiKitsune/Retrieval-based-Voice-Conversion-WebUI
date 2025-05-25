# Real-Time RVC CLI Usage (gui_v1.py)

This document describes the command-line interface (CLI) usage for the real-time voice changer script `gui_v1.py`.

To run the real-time voice conversion in CLI mode, use the `--cli` flag:

```bash
python gui_v1.py --cli [OPTIONS]
```

## Command-Line Arguments

Below is a list of available command-line arguments, their types, descriptions, default values, and choices where applicable.

*   **`--cli`**
    *   **Type:** `boolean`
    *   **Description:** Run in command-line interface mode. This flag activates CLI operation.
    *   **Default:** `False` (GUI mode)

*   **`--pth-path PATH`**
    *   **Type:** `string`
    *   **Description:** Path to the RVC `.pth` model file.
    *   **Default:** `""` (empty string)

*   **`--index-path PATH`**
    *   **Type:** `string`
    *   **Description:** Path to the Faiss `.index` file associated with the model.
    *   **Default:** `""` (empty string)

*   **`--input-device DEVICE`**
    *   **Type:** `string`
    *   **Description:** CLI: Input audio device name or index. Use tools like `python -m sounddevice` to list available devices and their names/indices.
    *   **Default:** `None` (system default input device)

*   **`--output-device DEVICE`**
    *   **Type:** `string`
    *   **Description:** CLI: Output audio device name or index. Use tools like `python -m sounddevice` to list available devices and their names/indices.
    *   **Default:** `None` (system default output device)

*   **`--sr-type TYPE`**
    *   **Type:** `string`
    *   **Description:** Sample rate type. `sr_model` uses the model's target sample rate. `sr_device` attempts to use the device's default sample rate.
    *   **Default:** `sr_model`
    *   **Choices:** `sr_model`, `sr_device`

*   **`--threshold VALUE`** (Note: maps to `threhold` in the script)
    *   **Type:** `integer`
    *   **Description:** Response threshold in dB. Audio below this level might be gated or processed differently.
    *   **Default:** `-60`

*   **`--pitch SEMITONES`**
    *   **Type:** `integer`
    *   **Description:** Pitch adjustment in semitones (e.g., `12` for one octave up, `-12` for one octave down).
    *   **Default:** `0`

*   **`--formant FACTOR`**
    *   **Type:** `float`
    *   **Description:** Formant adjustment factor. Values around `0.0` are typical. Can affect timbre.
    *   **Default:** `0.0`

*   **`--index-rate RATE`**
    *   **Type:** `float`
    *   **Description:** Controls the influence of the Faiss index in feature retrieval (0 to 1). Higher values mean more reliance on the index.
    *   **Default:** `0.0`

*   **`--rms-mix-rate RATE`**
    *   **Type:** `float`
    *   **Description:** RMS mix rate for volume envelope matching (0 to 1). Controls how much of the target's volume envelope is applied to the converted audio.
    *   **Default:** `0.0`

*   **`--f0-method METHOD`**
    *   **Type:** `string`
    *   **Description:** F0 (fundamental frequency) detection method.
    *   **Default:** `fcpe`
    *   **Choices:** `pm`, `harvest`, `crepe`, `rmvpe`, `fcpe`

*   **`--block-time SECONDS`**
    *   **Type:** `float`
    *   **Description:** Processing block time in seconds. This is the duration of audio chunks processed at once.
    *   **Default:** `0.25`

*   **`--n-cpu CORES`**
    *   **Type:** `integer`
    *   **Description:** Number of CPU cores used for specific parallelizable tasks (e.g., Harvest F0 extraction, feature retrieval).
    *   **Default:** System-dependent (e.g., `4` if 8 cores available, then potentially further limited).

*   **`--crossfade-length SECONDS`** (Note: maps to `crossfade_time` in the script)
    *   **Type:** `float`
    *   **Description:** Crossfade length in seconds, used for smoothing transitions between processed audio blocks.
    *   **Default:** `0.05`

*   **`--extra-time SECONDS`**
    *   **Type:** `float`
    *   **Description:** Extra inference time in seconds. This provides additional context for the model.
    *   **Default:** `2.5`

*   **`--i-noise-reduce`**
    *   **Type:** `boolean`
    *   **Description:** Enable input noise reduction.
    *   **Default:** `False`

*   **`--o-noise-reduce`**
    *   **Type:** `boolean`
    *   **Description:** Enable output noise reduction (applied to the converted voice).
    *   **Default:** `False`

*   **`--use-pv`**
    *   **Type:** `boolean`
    *   **Description:** Enable phase vocoder for SOLA algorithm, potentially improving quality at the cost of performance.
    *   **Default:** `False`

*   **`--function MODE`**
    *   **Type:** `string`
    *   **Description:** Operational mode. `vc` for voice conversion, `im` for input monitoring (passthrough).
    *   **Default:** `vc`
    *   **Choices:** `im`, `vc`

*   **`--save-config`**
    *   **Type:** `boolean`
    *   **Description:** If set along with `--cli`, saves the current CLI-derived configuration to `configs/inuse/config.json`. The script will then proceed to run the audio stream unless explicitly exited in a future modification.
    *   **Default:** `False`

*   **`--sg-hostapi NAME`**
    *   **Type:** `string`
    *   **Description:** Host API for audio devices. Primarily relevant for GUI mode but can be set. (e.g., "MME", "DirectSound", "WASAPI", "Core Audio").
    *   **Default:** `""` (empty string, system default or first available usually)

*   **`--sg-wasapi-exclusive`**
    *   **Type:** `boolean`
    *   **Description:** Enable WASAPI exclusive mode for audio devices (Windows specific). Primarily relevant for GUI mode.
    *   **Default:** `False`

## Examples

**1. Start voice conversion with a specific model (RVC v2, 48k) and index file:**
```bash
python gui_v1.py --cli --pth-path "assets/weights/MyModel_48k.pth" --index-path "logs/MyModel/added_IVF123_Flat_nprobe_1_MyModel_v2.index" --f0-method rmvpe --function vc
```

**2. Start input monitoring (no voice conversion):**
```bash
python gui_v1.py --cli --function im
```

**3. Start voice conversion with custom pitch, harvest F0 method, specific input/output devices, and higher threshold:**
```bash
python gui_v1.py --cli --pth-path "path/to/your.pth" --index-path "path/to/your.index" --pitch 6 --f0-method harvest --input-device "Microphone (Realtek Audio)" --output-device "Speakers (Realtek Audio)" --threshold -45 --function vc
```
*(Note: For `--input-device` and `--output-device`, you should use the names or indices of audio devices as listed by your system. You can often find these by running `python -m sounddevice`.)*

**4. Save current CLI settings to `configs/inuse/config.json` and then start voice conversion:**
```bash
python gui_v1.py --cli --pth-path "assets/weights/MyModel_40k.pth" --index-path "logs/MyModel/added_MyModel.index" --pitch -2 --save-config --function vc
```

**5. Start voice conversion with input noise reduction and a specific block time:**
```bash
python gui_v1.py --cli --pth-path "model.pth" --index-path "model.index" --i-noise-reduce --block-time 0.5 --function vc
```
