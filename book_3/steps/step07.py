import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None  # Function which this Variable is produced from

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator  # 1. get a function
        if f is not None:
            x = f.input  # 2. get an input of the function
            x.grad = f.backward(self.grad)  # 3. call backward of the function
            x.backward()  # Call backward of the previous Variable recursively


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # Store a creator of the output
        # "Define-by-Run"
        self.input = input
        self.output = output  # Save the output
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


####################

A = Square()
B = Exp()
C = Square()

# y = (e^(x^2))^2
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# Backward propagation
y.grad = np.array(1.0)
y.backward()
print(x.grad)
