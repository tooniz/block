
# Block Unblock Lib

Blocking and tiling is often needed to achieve:
- Good cache hits in GPU/accelerator memory
- Better reuse of activations or weights for matrix multiplications

This is a simple yet much faster method to blocking and unblocking data using existing PyTorch functions.

Inspired by [previous work](https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7) using numpy

# The Slow Way

A nested for-loops approach would look something like the snippet below:

```
def block_tensor_element_by_element(tensor, block_dim=128, ublock_dim=64, tile_dim=32, face_dim=16):
    # Example of reshaping and permuting element by element
    blocks_r = tensor.shape[-2] // block_dim
    blocks_c = tensor.shape[-1] // block_dim
    ublocks_r = block_dim // ublock_dim
    ublocks_c = block_dim // ublock_dim
    tiles_r = ublock_dim // tile_dim
    tiles_c = ublock_dim // tile_dim
    faces_r = tile_dim // face_dim
    faces_c = tile_dim // face_dim

    # Manually reshape the tensor
    blocked_tensor = []
    for br in range(blocks_r):
        for ur in range(ublocks_r):
            for tr in range(tiles_r):
                for fr in range(faces_r):
                    for bc in range(blocks_c):
                        for uc in range(ublocks_c):
                            for tc in range(tiles_c):
                                for fc in range(faces_c):
                                    block = tensor[br*block_dim:(br+1)*block_dim, bc*block_dim:(bc+1)*block_dim]
                                    ublock = block[ur*ublock_dim:(ur+1)*ublock_dim, uc*ublock_dim:(uc+1)*ublock_dim]
                                    tile = ublock[tr*tile_dim:(tr+1)*tile_dim, tc*tile_dim:(tc+1)*tile_dim]
                                    face = tile[fr*face_dim:(fr+1)*face_dim, fc*face_dim:(fc+1)*face_dim]
                                    blocked_tensor.append(face)

    # Reshape back into 2D
    concatenated_tensor = torch.cat(blocked_tensor)
    result = concatenated_tensor.reshape(tensor.shape[-2], tensor.shape[-1])
    return result
```

Achieving 35 samples/s using my AMD EPYC CPU.

# The fast way

Let's use the provided 2D blocking and tiling function with the same default parameters

```
from block_unblock import block_tensor

block_act = block_tensor(tensor)

```

Achieving 6000 samples/s using the same CPU. This seraializes the batch dim which could be parallelized as well for further speed up.