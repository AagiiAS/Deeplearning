import torch.nn as  nn
import torch
import numpy as np

# target labels are class indices, which are integers
# y_true = torch.tensor([0,1,2], dtype=torch.int64)
# y_true = torch.tensor([0,1,2], dtype=torch.long)
# One-Hot Encoded Target Labels
y_true = torch.tensor([[1,0,0], [0,1,0], [0,0,1]], dtype=torch.float32)
# y_true = torch.argmax(y_true, dim=1)
y_predict = torch.tensor([[8,9,10],[12,4,6],[3,8,6]], dtype=torch.float32)
#only expect class indices , integers
loss = nn.CrossEntropyLoss()
# print(f'loss: {loss( y_predict, y_true)}')
print(loss(y_predict, y_true))
m_l = loss(y_predict, y_true).numpy()
print(f'loss :{m_l}')

