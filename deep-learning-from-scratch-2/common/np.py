from common.config import GPU


if GPU:
    pass
    import cupy as np
    # Set the current allocate for GPU memory
    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

    # Notification
    print("\033[92m" + "-" * 60 + "\033[0m")
    print(" " * 23 + "\033[92mGPU Mode (cupy)\033[0m")
    print("\033[92m" + "-" * 60 + "\033[0m")
else:
    import numpy as np
