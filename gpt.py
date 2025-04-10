import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

torch.manual_seed(12046)

# 一些超参数
emb_size = 128
head_size = 8
n_layer = 12
sequence_len = 64
learning_rate = 1e-3
eval_iters = 20
batch_size = 100
# 如果有GPU，该脚本将使用GPU进行计算
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 计算设备为V100 16G
# 如果使用CPU，需要非常长的时间，建议减少模型规模来加快速度（比如n_layer）

def attention(query, key, value, dropout, mask=None):
    '''
    注意力机制
    参数
    ----
    query ：torch.FloatTensor，查询向量，形状为(B, T, H)
    key ：torch.FloatTensor，键向量，形状为(B, T, H)
    value ：torch.FloatTensor，数值向量，形状为(B, T, H)
    dropout ：随机失活
    mask ：torch.FloatTensor，掩码，形状为(T, T)
    返回
    ----
    out ：torch.FloatTensor，根据注意力机制得到的背景向量，形状为(B, T, H)
    w_att ：torch.FloatTensor，权重向量，形状为(B, T, T)
    '''
    # query, key, value都有相同的形状
    B, T, H = query.shape
    # (B, T, H) @ (B, H, T) --> (B, T, T)
    scores = query @ key.transpose(-2, -1) / (H ** 0.5)
    if mask is not None:
        # 如果没有mask，则表示词元可以使用左右两边的背景，也就是双向注意力
        # 如果mask是上三角矩阵，则表示自回归模式的单向注意力
        # mask的形状是(T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    scores = F.softmax(scores, dim=-1)  # (B, T, T)
    w_att = dropout(scores)  # (B, T, T)
    out = w_att @ value  # (B, T, H)
    return out, w_att


class MaskedAttention(nn.Module):

    def __init__(self, emb_size, head_size):
        '''
        单头单向注意力
        参数
        ----
        emb_size ：int，特征长度
        head_size ：int，背景向量长度
        '''
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        # 这个上三角矩阵不参与模型训练
        self.register_buffer(
            'tril', torch.tril(torch.ones(sequence_len, sequence_len)))
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        '''
        向前传播
        参数
        ----
        x ：torch.FloatTensor
            文本的特征向量，形状为(B, T, C)，其中B表示批量大小，T表示文本长度，C表示特征长度（emb_size）
        返回
        ----
        out ：torch.FloatTensor
            根据注意力机制得到的背景向量，形状为(B, T, H)，其中H表示背景向量长度（head_size）
        '''
        B, T, C = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)
        mask = self.tril[:T, :T]
        out, _ = attention(q, k, v, self.dropout, mask)
        return out  # (B, T, H)


class MaskedMultiHeadAttention(nn.Module):

    def __init__(self, emb_size, head_size):
        '''
        多头单向注意力
        参数
        ----
        emb_size ：int，特征长度
        head_size ：int，背景向量长度
        '''
        super().__init__()
        # 确保特征长度是背景向量长度的倍数
        assert (emb_size % head_size == 0)
        # 定义单头注意力的个数
        n_head = emb_size // head_size
        heads = [MaskedAttention(emb_size, head_size) for _ in range(n_head)]
        self.heads = nn.ModuleList(heads)
        # 线性变换
        self.proj = nn.Linear(emb_size, emb_size)
        # 随机失活
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        '''
        向前传播
        参数
        ----
        x ：torch.FloatTensor
            文本的特征向量，形状为(B, T, C)，其中B表示批量大小，T表示文本长度，C表示特征长度（emb_size）
        返回
        ----
        out ：torch.FloatTensor，根据注意力机制得到的背景向量，形状为(B, T, C)
        '''
        # 将多个单头注意力的结果做张量拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, emb_size):
        '''
        多层感知器
        '''
        super().__init__()
        self.l1 = nn.Linear(emb_size, 4 * emb_size)
        self.l2 = nn.Linear(4 * emb_size, emb_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.gelu(self.l1(x))
        out = self.dropout(self.l2(x))
        return out


class Block(nn.Module):

    def __init__(self, emb_size, head_size):
        '''
        解码块
        参数
        ----
        emb_size ：int，特征长度
        head_size ：int，单头注意力中的背景向量长度
        '''
        super().__init__()
        self.mha = MaskedMultiHeadAttention(emb_size, head_size)
        self.ff = FeedForward(emb_size)
        # 层归一化
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        '''
        向前传播
        参数
        ----
        x ：torch.FloatTensor，文本的特征向量，形状为(B, T, C)
        返回
        ----
        out ：torch.FloatTensor，解码块的输出，形状为(B, T, C)
        '''
        # 残差连接
        x = x + self.mha(self.ln1(x))  # (B, T, C)
        out = x + self.ff(self.ln2(x))  # (B, T, C)
        return out


class CharGPT(nn.Module):

    def __init__(self, vs):
        '''
        利用GPT-2进行自然语言的自回归学习
        参数
        ----
        vs ：int，字典大小
        '''
        super().__init__()
        # 文字嵌入层
        self.token_embedding = nn.Embedding(vs, emb_size)
        # 位置嵌入层
        self.position_embedding = nn.Embedding(sequence_len, emb_size)
        # 解码块
        blocks = [Block(emb_size, head_size) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*blocks)
        self.ln = nn.LayerNorm(emb_size)
        # 语言建模头
        self.lm_head = nn.Linear(emb_size, vs)

    def forward(self, x):
        '''
        向前传播
        参数
        ----
        x ：torch.LongTensor，当前字母在字典中的位置，形状为(B, T)
        返回
        ----
        logits ：torch.FloatTensor，预测结果的logits，形状为(B, T, vs)
        '''
        B, T = x.shape
        # 定义词元的位置，形状为(T)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        # 词元语义特征
        tok_emb = self.token_embedding(x)  # (B, T,  C)
        # 位置特征
        pos_emb = self.position_embedding(pos)  # (   T,  C)
        x = tok_emb + pos_emb  # (B, T,  C)
        x = self.blocks(x)  # (B, T,  C)
        x = self.ln(x)  # (B, T,  C)
        logits = self.lm_head(x)  # (B, T, vs)
        return logits


class char_tokenizer:

    def __init__(self, data):
        # 数据中出现的所有字符构成字典
        chars = sorted(list(set(''.join(data))))
        # 预留一个位置给结尾的特殊字符
        self.char2ind = {s: i + 1 for i, s in enumerate(chars)}
        self.char2ind['<|e|>'] = 0
        self.ind2char = {i: s for s, i in self.char2ind.items()}

    def encode(self, text):
        return [self.char2ind[c] for c in text]

    def decode(self, enc):
        if isinstance(enc, int):
            return self.ind2char[enc]
        return [self.ind2char[i] for i in enc]


@torch.no_grad()
def generate_batch(model, idx, max_new_tokens=300):
    '''
    利用模型生成文本（反复使用模型进行预测）
    参数
    ----
    model ：CharGPT，生成文本的模型
    idx ：torch.LongTensor，当前字母在字典中的位置，形状为(1, T)
    max_new_tokens ：int，生成文本的最大长度
    返回
    ----
    out ：list[int]，生成的文本
    '''
    # 将模型切换至评估模式
    model.eval()
    for _ in range(max_new_tokens):
        # 限制背景长度，否则会报错
        context = idx[:, -sequence_len:]
        # 在文本生成时，模型的计算效率很低，因为有很多重复计算
        logits = model(context)
        # 只使用最后一个预测结果
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        # 根据模型预测的概率，得到最终的预测结果（下一个字母）
        # 这一步运算有一定随机性
        ix = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, ix), dim=1)
        if ix.item() == 0:
            break
    # 将模型切换至训练模式
    model.train()
    return idx.tolist()[0]


def process(data, sequence_len=sequence_len):
    '''
    根据文本生成训练数据
    '''
    # text是字符串列表
    text = data['whole_func_string']
    inputs, labels = [], []
    for i in text:
        enc = tok.encode(i)
        # 0对应着文本结束
        enc += [0]
        # 将文本转换为多个训练数据
        for i in range(len(enc) - sequence_len):
            inputs.append(enc[i: i + sequence_len])
            # 预测标签是下一个字母，因此只需要挪动一个位置即可
            labels.append(enc[i + 1: i + 1 + sequence_len])
    return {'inputs': inputs, 'labels': labels}


raw_datasets = load_dataset('code_search_net', 'python', cache_dir='your/custom/path', trust_remote_code=True)
datasets = raw_datasets['train'].filter(lambda x: 'apache/spark' in x['repository_name'])
tok = char_tokenizer(datasets['whole_func_string'])
# 展示模型结构
model = CharGPT(len(tok.char2ind)).to(device)
# 统计模型的参数个数
print(f'{sum(p.numel() for p in model.parameters())} parameters')
print(model)

# 使用模型来生成文本
begin_text = torch.tensor(tok.encode('def'), device=device).unsqueeze(0)
print(''.join(tok.decode(generate_batch(model, begin_text))))

# 将数据分为训练集和测试集
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
# 将文本转换为训练数据，里面包含inputs和labels
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
tokenized.set_format(type='torch', device=device)

# 构建数据读取器
train_loader = DataLoader(tokenized['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tokenized['test'], batch_size=batch_size, shuffle=True)
# 获取一个批量的数据
print(next(iter(test_loader)))


def estimate_loss(model):
    re = {}
    # 将模型切换至评估模式
    model.eval()
    re['train'] = _loss(model, train_loader)
    re['test'] = _loss(model, test_loader)
    # 将模型切换至训练模式
    model.train()
    return re


@torch.no_grad()
def _loss(model, data_loader):
    '''
    计算模型在不同数据集下面的评估指标
    '''
    loss = []
    data_iter = iter(data_loader)
    # 随机使用多个批量数据来预估模型效果
    for k in range(eval_iters):
        data = next(data_iter, None)
        if data is None:
            data_iter = iter(data_loader)
            data = next(data_iter, None)
        inputs, labels = data['inputs'], data['labels']
        logits = model(inputs)
        # 根据cross_entropy的定义，需要对logits进行转置运算
        # 具体细节请参考cross_entropy的官方文档
        logits = logits.transpose(-2, -1)
        loss.append(F.cross_entropy(logits, labels).item())
    return torch.tensor(loss).mean().item()


estimate_loss(model)


def train_gpt(model, optimizer, data_loader, epochs=10):
    lossi = []
    l = len(data_loader)
    for epoch in range(epochs):
        for i, data in enumerate(data_loader, 0):
            print(str(epoch)+': '+str(i)+'/'+str(l))
            inputs, labels = data['inputs'], data['labels']
            optimizer.zero_grad()
            logits = model(inputs)
            # 根据cross_entropy的定义，需要对logits进行转置运算
            logits = logits.transpose(-2, -1)
            loss = F.cross_entropy(logits, labels)
            lossi.append(loss.item())
            loss.backward()
            optimizer.step()
        # 评估模型，并输出结果
        stats = estimate_loss(model)
        train_loss = f"train loss {stats['train']:.4f}"
        test_loss = f"test loss {stats['test']:.4f}"
        print(f'epoch {epoch:>2}: {train_loss}, {test_loss}')
    return lossi


# l = train_gpt(model, optim.AdamW(model.parameters(), lr=learning_rate), train_loader)

# plt.plot(torch.tensor(l).view(-1, 10).mean(1).numpy())

# torch.save(model.state_dict(), 'gpt.pth')
# # 读取模型参数
new_model = CharGPT(len(tok.char2ind)).to(device)
new_model.load_state_dict(torch.load('gpt.pth'))

# 使用模型来生成文本
begin_text = torch.tensor(tok.encode('def HelloWorld'), device=device).unsqueeze(0)
# print(''.join(tok.decode(generate_batch(model, begin_text))))
print(''.join(tok.decode(generate_batch(new_model, begin_text))))
