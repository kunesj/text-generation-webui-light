# Text Generation Web UI: Light

This is a fork of [Text Generation Web UI](https://github.com/oobabooga/text-generation-webui) that aims to turn it into just a frontend, 
that could be used with any existing installation of [llama.cpp](https://github.com/ggml-org/llama.cpp).

```bash
LLAMA_SERVER_PATH=/path/to/your/llama-server uv run python server.py
```

[Original README.md](./README_upstream.md)


### Changes

- **Use separately installed llama.cpp** specified by `LLAMA_SERVER_PATH` env variable. 
- Removed all other inference backends and simplified dependencies to not need `torch` etc. 
- Migrated project to uv. Added pre-commit config. Added ruff config.
- Removed all run scripts etc. Just directly use `uv run python server.py`.


### TODO

- Use llama.cpp models-presets.ini configs
- Use builtin llama.cpp model switcher
- Generalize to use any OpenAI-compatible API
- Chat folders
- Refactor, cleanup and modernize the code
- Remove any leftovers from other inference backends, image generation, etc.
