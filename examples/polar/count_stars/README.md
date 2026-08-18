# Count Stars (deferred VLM recipe)

This task is retained as a VLM artifact fixture, but it is not a supported
training recipe in the text-only Molt–Polar integration. Passing a Polar task
specification together with `--data.max_images_per_prompt > 0` fails at CLI
validation with a message pointing to the deferred VLM integration.

VLM rollout needs a shared run-artifact path for images and multimodal
processor inputs before this recipe can move to the Molt CLI. The runtime image
and source asset remain here for that migration; there is intentionally no
standalone Polar launcher or hand-written topology.
