import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from BPE import BPETokenizer

# 模型超参数
emb_size = 128
head_size = 8
n_layer = 12
sequence_len = 64
learning_rate = 1e-3
eval_iters = 20
batch_size = 100
model_path = "gpt_model.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载数据集并训练BPE分词器
raw_datasets = load_dataset('code_search_net', 'python', cache_dir='your/custom/path', trust_remote_code=True)
dataset = raw_datasets['train'].filter(lambda x: 'apache/spark' in x['repository_name'])
texts = dataset['whole_func_string']
tok = BPETokenizer(texts)


# 模型定义
class GPTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(sequence_len, emb_size)
        self.blocks = nn.Sequential(*[Block(emb_size, head_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(emb_size)
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# 模型结构依赖的子模块
class MaskedAttention(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(sequence_len, sequence_len)))
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        mask = self.tril[:T, :T]
        scores = q @ k.transpose(-2, -1) / (C ** 0.5)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return weights @ v


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        assert emb_size % head_size == 0
        self.n_head = emb_size // head_size
        self.heads = nn.ModuleList([MaskedAttention(emb_size, head_size) for _ in range(self.n_head)])
        self.proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.l1 = nn.Linear(emb_size, 4 * emb_size)
        self.l2 = nn.Linear(4 * emb_size, emb_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        return self.dropout(self.l2(F.gelu(self.l1(x))))


class Block(nn.Module):
    def __init__(self, emb_size, head_size):
        super().__init__()
        self.mha = MaskedMultiHeadAttention(emb_size, head_size)
        self.ff = FeedForward(emb_size)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# 数据处理函数
def process(batch):
    texts = batch['whole_func_string']
    inputs, labels = [], []
    for text in texts:
        ids = tok.encode(text) + [tok.eos_id]
        if len(ids) <= sequence_len:
            continue
        for i in range(len(ids) - sequence_len):
            inputs.append(ids[i:i + sequence_len])
            labels.append(ids[i + 1:i + 1 + sequence_len])
    return {'inputs': inputs, 'labels': labels}


# 数据集分割与映射
data_split = dataset.train_test_split(test_size=0.1, seed=1024)
tokenized_ds = data_split.map(process, batched=True, remove_columns=dataset.column_names)
tokenized_ds.set_format(type='torch', device=device)
train_loader = DataLoader(tokenized_ds['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tokenized_ds['test'], batch_size=batch_size)


# 损失评估函数
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {'train': 0.0, 'test': 0.0}
    for split, loader in [('train', train_loader), ('test', test_loader)]:
        batch_losses = []
        for k, batch in enumerate(loader):
            if k == eval_iters:
                break
            inp, tgt = batch['inputs'], batch['labels']
            logits = model(inp)
            logits = logits.transpose(1, 2)
            loss = F.cross_entropy(logits, tgt)
            batch_losses.append(loss.item())
        losses[split] = float(torch.tensor(batch_losses).mean())
    model.train()
    return losses


# 训练函数
def train_gpt(model, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            inp, tgt = batch['inputs'], batch['labels']
            optimizer.zero_grad()
            logits = model(inp)
            logits = logits.transpose(1, 2)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            optimizer.step()
        stats = estimate_loss(model)
        train_loss = stats['train']
        test_loss = stats['test']
        print(f"epoch {epoch + 1:>2}: train loss {train_loss:.4f}, perplexity {math.exp(train_loss):.2f}; "
              f"test loss {test_loss:.4f}, perplexity {math.exp(test_loss):.2f}")


# 模型训练与保存
model = GPTModel(tok.tokenizer.get_vocab_size()).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_gpt(model, optimizer, epochs=5)
torch.save(model.state_dict(), model_path)


# 文本生成函数
def generate(model, prompt, max_new_tokens=100):
    model.eval()
    ids = tok.encode(prompt)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        input_cond = input_ids[:, -sequence_len:]
        logits = model(input_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == tok.eos_id:
            break
    return tok.decode(input_ids[0].tolist())


# # 加载模型进行文本生成
# model.load_state_dict(torch.load(model_path))
# prompt = "def hello_world():\n    print(\"Hello"
# print(generate(model, prompt))
