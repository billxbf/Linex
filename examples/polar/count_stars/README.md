# Count Stars VLM smoke recipe

This is the smallest VLM optimizer-step recipe. Polar uploads the fixture into
the task runtime, sends it through the Molt-owned vLLM router, and stores the
media under the run's `--rollout.save_dir`. The returned trace carries only the
media path used by Molt to build multimodal training inputs.

Build the runtime image, then launch the Qwen VLM quick start:

```bash
python examples/polar/count_stars/build_image.py
MODEL_PATH=/models/Qwen3.6-35B-A3B \
  bash examples/molt/scripts/quick_start/rl_qwen3_6_35b.sh
```

That launch enables `--data.max_images_per_prompt 1`, so one optimizer step
exercises the complete VLM artifact path.
