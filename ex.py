# # import torch
# # ex = torch.rand([4, 2, 2, 100, 16])
# # print('ex:', ex.shape)
# # Z = [list_ for list_ in ex]
# # Z = torch.sum(Z, dim=2)  

# # Z_list = [value for value in Z]
# # Z = torch.cat(Z_list, -1)
# # Z = torch.matmul(Z, torch.rand([264, 64]))
# # print('Z: ',Z.shape)
import torch

# # Step 1: Initialize random tensors
# Z = torch.rand([4, 200, 200, 64, 16])  # (4, 2, 2, user/item, 16)
# # V = torch.rand([4, 2, 100, 16])       # (4, 2, user/item, 16)

# # # Step 2: Multiply att and V
# # Z = torch.mul(att, V)  # Element-wise multiplication
# # print("Shape of Z after multiplication: ", Z.shape)  # Should be (4, 2, 2, user/item, 16)

# # Step 3: Sum over dimension 2 (user/item dimension)
# Z = torch.sum(Z, dim=2)  # Sum along the 2nd dimension
# print("Shape of Z after summing along dim=2: ", Z.shape)  # Should be (4, 2, 16)

# # Step 4: Flatten the tensor list and concatenate them
# Z_list = [value for value in Z]
# Z = torch.cat(Z_list, -1)  # Concatenate along the last dimension
# print("Shape of Z after concatenation: ", Z.shape)  # Should be (4, 32)

# # Step 5: Perform matrix multiplication with a random weight matrix
# w_self_attention_cat = torch.rand([264, 64])  # Weight matrix of shape (32, 64)
# Z = torch.matmul(Z, w_self_attention_cat)  # Matrix multiplication
# print("Shape of Z after matrix multiplication: ", Z.shape)  # Should be (4, 64)
import torch
# ex = torch.rand([100, 10])
# print(ex)
# ex = ex.reshape(10, 1, 10, 10)
# print(ex.shape)
# print(ex)
# user = 100
# ex1 = torch.rand([4, 2*user, 2*user, 64, 1])
# ex2 = torch.rand([4, 1, 2*user, 64, 16])
# ex3 = torch.mul(ex1, ex2)
# print(ex3.shape)
# ex_list = [torch.rand([100, 64]), torch.rand([100, 64])]
# ex_list = torch.stack(ex_list, dim=0)
# print(ex_list.shape)
import torch.nn.functional as F
ex = torch.matmul(torch.rand([2, 100, 64]), torch.rand([64, 64]))
print(ex.shape)
ex = ex.reshape(2, 100, 4, 16).permute(2, 0, 1, 3)
print(ex.shape)
Q = torch.unsqueeze(ex, 2) 
K = V = torch.unsqueeze(ex, 1)
att = torch.mul(Q, K) / torch.sqrt(torch.tensor(16)) 
att = torch.sum(att, dim=-1) 
# (4, 2, 2, user/item, 1)
att = torch.unsqueeze(att, dim=-1)  
# # (4, 2, 2, user/item, 1)
att = F.softmax(att, dim=2)  

# (4, 2, 2, user/item, 16)
Z = torch.mul(att, V)  
# (4, 2, user/item, 16)
Z = torch.sum(Z, dim=2)  

Z_list = [value for value in Z]
Z = torch.cat(Z_list, -1)
print(Z.shape)
Z = torch.matmul(Z, torch.rand(64, 64))
print(Z.shape)


res = torch.rand([2, 100, 64])
res = res.mean(0)
print(res.shape)