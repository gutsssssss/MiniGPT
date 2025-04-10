import torch
import torch.nn.functional as F
import string

# # 定义字典
# char2indx = {s: i for i, s in enumerate(sorted(string.ascii_lowercase))}
#
# example = list('love')
#
# # 利用字典，对文本进行数字化
# idx = []
#
# for i in example:
#     idx.append(char2indx[i])
#
# idx = torch.tensor(idx)
#
# # 使用独热编码，将文本转换为二维张量
# num_claz = 26
# dims = 5
# x = F.one_hot(idx, num_classes=num_claz).float()
#
# # 文本嵌入其实就是张量乘法
# W = torch.randn((num_claz, dims)) # (26,  5)
# print(torch.matmul(x, W))
#
# # 与前面张量乘法一致，但更加友好的实现方式
# # 因为运算涉及的张量idx维度更少，而且不需要经过独热编码
# print(W[idx])
# print(W)

from datasets import load_dataset

dataset = load_dataset('code_search_net', 'python', trust_remote_code=True)
print(dataset.cache_files)