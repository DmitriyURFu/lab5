# Домашнее задание к уроку 1: Основы PyTorch

## Цель задания
Закрепить навыки работы с тензорами PyTorch, изучить основные операции и научиться решать практические задачи.

## Задание 1: Создание и манипуляции с тензорами (25 баллов)

Создайте файл `homework_tensors.py` и выполните следующие задачи:

### 1.1 Создание тензоров (7 баллов)
```python
# Создайте следующие тензоры:
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
rand_tensor = torch.rand(3,4)

# - Тензор размером 2x3x4, заполненный нулями
zero_tensor = torch.zero(2,3,4)

# - Тензор размером 5x5, заполненный единицами
ones_tensor = torch.ones(5,5)

# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
0_to_15_tensor = torch.arange(16).reshape(4,4)
```

### 1.2 Операции с тензорами (6 баллов)
```python
# Дано: тензор A размером 3x4 и тензор B размером 4x3
A = torch.arange(12).reshape(3,4)
B = torch.arange(12).reshape(4,3)

# Выполните:
# - Транспонирование тензора A
transp_tensor = A.T

# - Матричное умножение A и B
multiply_A_and_B = A@B

# - Поэлементное умножение A и транспонированного B
elemnts_multiply = A*B.T

# - Вычислите сумму всех элементов тензора A
sum_elements_A = A.sum()
```

### 1.3 Индексация и срезы (6 баллов)
```python
# Создайте тензор размером 5x5x5
cube_tensor = torch.arange(125).reshape(5,5,5)

# Извлеките:
# - Первую строку
first_str_tensor = cube_tensor[0,:,:]

# - Последний столбец
last_clumn_tensor = cube_tensor[:, -1 ,:]

# - Подматрицу размером 2x2 из центра тензора
matrix_2x2 = cube_tensor [2:4, 2:4, ;]

# - Все элементы с четными индексами
even_index_elements = cube_tensor[::2,::2,::2]
```

### 1.4 Работа с формами (6 баллов)
```python
# Создайте тензор размером 24 элемента
tensor_24 = torch.arange(24)

# Преобразуйте его в формы:
# - 2x12
tensor_2x12 = tensor_24.reshape(2, 12)

# - 3x8
tensor_3x8 = tensor_24.reshape(3, 8)

# - 4x6
tensor_4x6 = tensor_24.reshape(4, 6)

# - 2x3x4
tensor_2x3x4 = tensor_24.reshape(2, 3, 4)

# - 2x2x2x3
tensor_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)
```

## Задание 2: Автоматическое дифференцирование (25 баллов)

Создайте файл `homework_autograd.py`:

### 2.1 Простые вычисления с градиентами (8 баллов)
```python
# Создайте тензоры x, y, z с requires_grad=True
tensor_x = torch.tensor(1, requires_grad=True)
tensor_y = torch.tensor(2, requires_grad=True)
tensor_z = torch.tensor(3, requires_grad=True)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2*x*y*z

# Найдите градиенты по всем переменным
grad_x = tensor_x.grad.item()
grad_y = tensor_y.grad.item()
grad_z = tensor_z.grad.item()

# Проверьте результат аналитически
analytical_grad_x = 2*tensor_x.item() + 2*tensor_y.item()*tensor_z.item()
analytical_grad_y = 2*tensor_y.item() + 2*tensor_x.item()*tensor_z.item()
analytical_grad_z = 2*tensor_z.item() + 2*tensor_x.item()*tensor_y.item()

if (grad_x == analytical_grad_x):
    print('X true')
if (grad_y == analytical_grad_y):
    print('Y true')
if (grad_z == analytical_grad_z):
    print('Z true')
```

### 2.2 Градиент функции потерь (9 баллов)
```python
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
```

### 2.3 Цепное правило (8 баллов)
```python
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
```

## Задание 3: Сравнение производительности CPU vs CUDA (20 баллов)

Создайте файл `homework_performance.py`:

### 3.1 Подготовка данных (5 баллов)
```python
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
matrix_64x1024x1024 = torch.randint(0, 100, (64, 1024, 1024))

# - 128 x 512 x 512
matrix_128x512x512 = torch.randint(0, 100, (128, 512, 512))

# - 256 x 256 x 256
matrix_256x256x256 = torch.randint(0, 100, (256, 256, 256))

# Заполните их случайными числами
```

### 3.2 Функция измерения времени (5 баллов)
```python
# Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU
def measure_time(operation, *args, device='cpu', **kwargs):
    if isinstance(device, torch.device):
        device = str(device)
        
    if 'cuda' in device and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        operation(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000.0
    else:
        t0 = time.time()
        operation(*args, **kwargs)
        return time.time() - t0
```

### 3.3 Сравнение операций (10 баллов)
```python
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов
operations = [
    ("Матричное умножение", lambda a, b: torch.matmul(a, b)),
    ("Поэлементное сложение", lambda a, b: a + b),
    ("Поэлементное умножение", lambda a, b: a * b),
    ("Транспонирование", lambda a, b: torch.transpose(a, -1, -2)),
    ("Сумма всех элементов", lambda a, b: torch.sum(a)),
]

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде
print(f"{'Операция':<22} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
print("-" * 60)

A_cpu = matrix_64x1024x1024.clone()
B_cpu = matrix_64x1024x1024.clone()
cuda_available = torch.cuda.is_available()#Пример для 1 матрицы из 3
if cuda_available:
    A_gpu = A_cpu.cuda()
    B_gpu = B_cpu.cuda()
    
for name, op in operations:
    time_cpu_sec = measure_time(op, A_cpu, B_cpu, device='cpu')
    time_cpu_ms = time_cpu_sec * 1000

    if cuda_available:
        time_gpu_sec = measure_time(op, A_gpu, B_gpu, device='cuda')
        time_gpu_ms = time_gpu_sec * 1000
        speedup = time_cpu_ms / time_gpu_ms if time_gpu_ms > 0 else float('inf')
        speedup_str = f"{speedup:.1f}x"
    else:
        time_gpu_ms = float('nan')
        speedup_str = "N/A"

    if cuda_available:
        print(f"{name:<22} | {time_cpu_ms:<10.2f} | {time_gpu_ms:<10.2f} | {speedup_str:<10}")
    else:
        print(f"{name:<22} | {time_cpu_ms:<10.2f} | {'N/A':<10} | {speedup_str:<10}")
```

### Пример вывода:
```
Операция          | CPU (мс) | GPU (мс) | Ускорение
Матричное умножение|   150.2  |    12.3  |   12.2x
Сложение          |    45.1  |     3.2  |   14.1x
...
```

### 3.4 Анализ результатов (5 баллов)
```python
# Проанализируйте результаты:
# - Какие операции получают наибольшее ускорение на GPU?
GPU ускоряет лучше всего те операции, где много одинаковых вычислений, которые можно делать одновременно.
Сильно ускоряются на GPU матричное умножение.

# - Почему некоторые операции могут быть медленнее на GPU?
Время на чтение и запись информации тратиться больше, чем идет само вычисление. GPU выгоден для сложных вычислений

# - Как размер матриц влияет на ускорение?
Размер матриц сильно влияет на ускорение. На малых размерах накладные расходы (синхронизация, запуск ядер) доминируют,
поэтому ускорения может не быть. На крупных размерах вычислительная нагрузка растёт быстрее, чем накладные расходы,
и GPU раскрывает свой потенциал.

# - Что происходит при передаче данных между CPU и GPU?
Передача данных между CPU и GPU — очень медленная по сравнению с вычислениями на одном устройстве.
Копирование тензора через .cuda() или .cpu() проходит через PCIe-шину, пропускная способность которой на порядок ниже,
чем у памяти GPU. Поэтому частая передача данных сводит на нет всё ускорение. 
```

## Дополнительные требования

1. **Код должен быть читаемым** - используйте комментарии и понятные имена переменных
2. **Обработка ошибок** - добавьте проверки размерностей и типов данных
3. **Документация** - добавьте docstring к функциям
4. **Тестирование** - проверьте результаты на простых примерах или напишите тесты
5. **Адаптивность** - код должен работать как с GPU, так и без него

## Срок сдачи
Домашнее задание должно быть выполнено до начала занятия 3.

## Полезные ссылки
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [CUDA Performance](https://pytorch.org/docs/stable/notes/cuda.html)

Удачи в выполнении задания! 🚀 