import numpy as np

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def main():
    dW1 = np.random.rand(3, 3) * 10
    dW2 = np.random.rand(3, 3) * 10
    grads = [dW1, dW2]
    max_norm = 5.0

    print(dW1)
    print(dW2)

    clip_grads(grads, max_norm)

    print("clipped:")
    print(dW1)
    print(dW2)


if __name__ == "__main__":
    main()
