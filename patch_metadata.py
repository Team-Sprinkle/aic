import json
import os

# Local path relative to your workspace
json_path = "outputs/act_policy_ts_080000_cpu.json"

# The path where these will live INSIDE the Docker image
container_checkpoint_dir = "/opt/aic_policy/act_080000/checkpoints/080000/pretrained_model"
normalizer_filename = "policy_preprocessor_step_3_normalizer_processor.safetensors"
full_normalizer_path = os.path.join(container_checkpoint_dir, normalizer_filename)

with open(json_path, 'r') as f:
    data = json.load(f)

# Update the main checkpoint directory
data['checkpoint_dir'] = container_checkpoint_dir

# Update the normalizer reference (targeting common ACT JSON keys)
data['stats_path'] = full_normalizer_path
data['preprocessor_path'] = full_normalizer_path
data['normalizer_path'] = full_normalizer_path # Explicitly adding/updating this

with open(json_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"--- Metadata Patched ---")
print(f"Checkpoint: {data['checkpoint_dir']}")
print(f"Normalizer: {data['normalizer_path']}")