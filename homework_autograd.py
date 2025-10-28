import torch

# Создайте тензоры x, y, z с requires_grad=True
tensor_x = torch.tensor(1, requires_grad=True)
tensor_y = torch.tensor(2, requires_grad=True)
tensor_z = torch.tensor(3, requires_grad=True)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = tensor_x**2 + tensor_y**2 + tensor_z**2 + 2*tensor_x*tensor_y*tensor_z

# Найдите градиенты по всем переменным
grad_x = tensor_x.grad.item()
grad_y = tensor_y.grad.item()
grad_z = tensor_z.grad.item()

# Проверьте результат аналитически
analytical_grad_x = 2*tensor_x.item() + 2*tensor_y.item()*tensor_z.item()
analytical_grad_y = 2*tensor_y.item() + 2*tensor_x.item()*tensor_z.item()
analytical_grad_z = 2*tensor_z.item() + 2*tensor_x.item()*tensor_y.item()

print(f'analytical grad_x: {analytical_grad_x} and grad_x: {grad_x}')
print(f'analytical grad_y: {analytical_grad_x} and grad_y: {grad_x}')
print(f'analytical grad_z: {analytical_grad_x} and grad_z: {grad_x}')


# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
# Найдите градиенты по w и b
def mse_loss_and_grads(x, y_true, w, b):
    y_pred = w * x + b
    loss = torch.mean((y_pred - y_true) ** 2)
    loss.backward()
    grad_w = w.grad.item()
    grad_b = b.grad.item()
    return loss.item(), grad_w, grad_b


x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([2.1, 3.9, 6.0, 8.1])

w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

loss, grad_w, grad_b = mse_loss_and_grads(x, y_true, w, b)

# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
def composite_function(x):
    return torch.sin(x**2 + 1)
x = torch.tensor(2.0, requires_grad=True)
f = composite_function(x)

# Найдите градиент df/dx
grad_x = x.grad.item()

# Проверьте результат с помощью torch.autograd.grad
f.backward(retain_graph=True)
grad_autograd = torch.autograd.grad(f, x)[0].item()
if (grad_x == grad_autograd):
    print('True')