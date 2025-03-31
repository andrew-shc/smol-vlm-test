# SmolVLM Test

Just testing SmolVLM in Rust via Candle and tokenizer.

Currently it has a working text interface (maybe not the current commit).

## How to use
`img> ` Paste in a URL. If invalid or empty, it will first say which and will continue as if there were no images.
`txt> ` Text prompt.


## Short-term goals
- Add KV Caching
- Experiment with real-time image processing (read images and undestand each frame)
- (Maybe) multiple images and interleaving images


# Learning Experience

- Biggest difficulties:
    - Getting swapping the weight values in MLPGates
    - Setting the scaled attention product to the full hidden dim (2048) instead of per head dimension (32)
    - Installing `candle-fast-attention` which (for no reason) uses parallel build process for all the CUDA kernels, filling up my swap space and freezing my laptop (took a while to realize)
- Biggest surprise:
    - Before, mostly played around with Python's transformer API. Never understood how LLMs read the structured formats of assistants and user message until I encountered special tokens.

