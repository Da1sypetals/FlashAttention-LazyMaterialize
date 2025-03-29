# Flash attention with lazily materialized attention mask

> This repo is a fork of https://github.com/tspeterkim/flash-attention-minimal.


Attention mask is constructed as $mask=is\_positive(MM^T)$, where $M$ has shape `(seq_len, mask_dim)`. 
- The $is\_positive$ is an element-wise operation returning True if the element is positive, otherwise False.
- `mask_dim` is often the same order of magnitude as `embed_dim` for $Q, K$ and $V$.