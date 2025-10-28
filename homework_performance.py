import time
import torch

matrix_64x1024x1024 = torch.randint(0, 100, (64, 1024, 1024))
matrix_128x512x512 = torch.randint(0, 100, (128, 512, 512))
matrix_256x256x256 = torch.randint(0, 100, (256, 256, 256))


def measure_time(operation, *args, device='cpu', **kwargs):
    if isinstance(device, torch.device):
        device = str(device)

    try:
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
    except Exception as e:
        print(
            f"Ошибка при выполнении операции '{operation.__name__ if hasattr(operation, '__name__') else 'operation'}': {e}")
        return float('nan')


operations = [
    ("Матричное умножение", lambda a, b: torch.matmul(a, b)),
    ("Поэлементное сложение", lambda a, b: a + b),
    ("Поэлементное умножение", lambda a, b: a * b),
    ("Транспонирование", lambda a, b: a.transpose(-1, -2)),  # Исправлено - используем только первый аргумент
    ("Сумма всех элементов", lambda a, b: torch.sum(a)),
]

print(f"{'Операция':<22} | {'CPU (мс)':<10} | {'GPU (мс)':<10} | {'Ускорение':<10}")
print("-" * 60)

A_cpu = matrix_64x1024x1024.clone()
B_cpu = matrix_64x1024x1024.clone()

cuda_available = torch.cuda.is_available()
if cuda_available:
    A_gpu = A_cpu.cuda()
    B_gpu = B_cpu.cuda()

for name, op in operations:
    if name == "Транспонирование":
        time_cpu_sec = measure_time(op, A_cpu, None, device='cpu')
        if cuda_available:
            time_gpu_sec = measure_time(op, A_gpu, None, device='cuda')
    else:
        time_cpu_sec = measure_time(op, A_cpu, B_cpu, device='cpu')
        if cuda_available:
            time_gpu_sec = measure_time(op, A_gpu, B_gpu, device='cuda')

    time_cpu_ms = time_cpu_sec * 1000

    if cuda_available and not torch.isnan(torch.tensor(time_gpu_sec)):
        time_gpu_ms = time_gpu_sec * 1000
        speedup = time_cpu_ms / time_gpu_ms if time_gpu_ms > 0 else float('inf')
        speedup_str = f"{speedup:.1f}x"
        print(f"{name:<22} | {time_cpu_ms:<10.2f} | {time_gpu_ms:<10.2f} | {speedup_str:<10}")
    else:
        time_gpu_ms = "N/A"
        speedup_str = "N/A"
        print(f"{name:<22} | {time_cpu_ms:<10.2f} | {time_gpu_ms:<10} | {speedup_str:<10}")