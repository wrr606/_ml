from micrograd import Value

def f(x: Value, y: Value, z: Value) -> Value:
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

x = Value(1)
y = Value(1)
z = Value(1)

lr = 0.0001
for _ in range(100000):
    cur = f(x, y, z)
    x.grad, y.grad, z.grad = 0, 0, 0
    cur.backward()
    x.data -= x.grad * lr
    y.data -= y.grad * lr
    z.data -= z.grad * lr
print(x)
print(y)
print(z)