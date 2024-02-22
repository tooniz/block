import torch

from timeit import default_timer as timer
from block_unblock import block_tensor, unblock_tensor

def main():

    # 2D activation
    activation = torch.arange(0, 384*1024, dtype=torch.bfloat16).reshape(384,1024)

    # For simplificy we will loop over the batch rather than performing 3D block/unblock
    microbatch_size = 128
    nbytes = activation.nelement() * activation.element_size()

    # Benchmark block/unblock operations on CPU
    # - A gibibyte is equal to 2^30 or 1,073,741,824 bytes.
    # - A gigabyte is equal to 10^9 or 1,000,000,000 bytes.

    start = timer()
    for _ in range(microbatch_size):
        block_act = block_tensor(activation)
    duration = timer() - start
    print(f"PERF: python tilize time = {duration:.2f}, {microbatch_size/(duration):.2f} samples/s, {microbatch_size*nbytes/(duration*10**9):.2f} GB/s")

    start = timer()
    for _ in range(microbatch_size):
        ublock_act = unblock_tensor(block_act)
    duration = timer() - start
    print(f"PERF: python untilize time = {duration:.2f}, {microbatch_size/(duration):.2f} samples/s, {microbatch_size*nbytes/(duration*10**9):.2f} GB/s")

    # Check that the unblocked tensor is the same as the original
    assert torch.allclose(activation, ublock_act)

if __name__ == "__main__":
    main()
