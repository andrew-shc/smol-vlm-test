# SmolVLM Test

Just testing SmolVLM in Rust via Candle and tokenizer.

Currently it has a working text interface (maybe not the current commit). Looking to finalize integrating the vision transformer.
After that, might try to see how well it works with robotics and real-time processing of images.

# Learning Experience

- Biggest difficulties:
    - Getting swapping the weight values in MLPGates
    - Setting the scaled attention product to the full hidden dim (2048) instead of per head dimension (32)
    - Installing `candle-fast-attention` which (for no reason) uses parallel build process for all the CUDA kernels, filling up my swap space and freezing my laptop (took a while to realize)
- Biggest surprise:
    - Before, mostly played around with Python's transformer API. Never understood how LLMs read the structured formats of assistants and user message until I encountered special tokens.

