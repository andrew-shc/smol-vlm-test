# SmolVLM Test

Just testing SmolVLM in Rust via Candle and tokenizer.

It has a working text and image interface via CLI. Currently, it only takes in at most one image mostly as a proof of concept for Rust implementation.

## How to use
`img> ` Paste in a URL. If invalid or empty, it will continue as if there no images were inputted.
`txt> ` Text prompt.


## Short-term goals
-[ ] Add KV Caching
-[ ] Experiment with real-time image processing (read images and undestand each frame)
-[ ] (Maybe) multiple images and interleaving images


# Learning Experience

- Learned multimodal LLMs are conceptually easier than thought
    - Like SmolVLM, multimodal LLMs can interpret images via image encoder which transforms images into token embeddings which would just be
    directly merged with other tokens for the text model itself. No cross attention and other complicated shenanigans.
- Biggest difficulties:
    - Getting swapping the weight values in MLPGates
    - Setting the scaled attention product to the full hidden dim (2048) instead of per head dimension (32)
    - Installing `candle-fast-attention` which (for no reason) uses parallel build process for all the CUDA kernels, filling up my swap space and freezing my laptop (took a while to realize)
    - Images are not being properly interpreted and mapped to token embedding space, producing random gibberish when an image is given.
        - Had to transpose back the attention (took a while to find out)
            - Now it's producing legible natural language but completely misinterprets the images
        - Replacing `candle`'s GELU with custom GELU approximated with tanh (which the Python impl. uses)
            - No effect
        - Realized I had the entire attention layer skipped by referring `xs` instead of `x` in attention module.
- Biggest surprise:
    - LLMs are given a Jinja template to structure the raw message input with special tokens. Before, I did not realize all this was happening
    within Python's `transformer` library.
    - LLMs are really *really* just bunch of math operations. After implementing my first LLM in Rust (SmolVLM), it changes how you see LLM now that you know how each math operation works under the hood. They're just a bunch of embedding transformations (things you use in vector databases), layer norms (similiar to batch norms), and transformers (i.e., scaled dot products (and also MLPs))
