import torch


# Создайте следующие тензоры:
rand_tensor = torch.rand(3,4)
zero_tensor = torch.zero(2,3,4)
ones_tensor = torch.ones(5,5)
tensor4x4 = torch.arange(16).reshape(4,4)

# Дано: тензор A размером 3x4 и тензор B размером 4x3
A = torch.arange(12).reshape(3,4)
B = torch.arange(12).reshape(4,3)
# Выполните:
transp_tensor = A.T
multiply_A_and_B = A@B
elemnts_multiply = A*B.T
sum_elements_A = A.sum()

# Создайте тензор размером 5x5x5
cube_tensor = torch.arange(125).reshape(5,5,5)
# Извлеките:
first_str_tensor = cube_tensor[0,:,:]
last_column_tensor = cube_tensor[:, -1 ,:]
matrix_2x2 = cube_tensor [2:4, 2:4, :]
even_index_elements = cube_tensor[::2,::2,::2]

# Создайте тензор размером 24 элемента
tensor_24 = torch.arange(24)
# Преобразуйте его в формы:
tensor_2x12 = tensor_24.reshape(2, 12)
tensor_3x8 = tensor_24.reshape(3, 8)
tensor_4x6 = tensor_24.reshape(4, 6)
tensor_2x3x4 = tensor_24.reshape(2, 3, 4)
tensor_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)