import torch
import torch.nn.functional as F
from BPE import BPETokenizer
from gpt2 import GPTModel, sequence_len, emb_size, head_size, n_layer

# 设备设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载分词器和模型
print("加载分词器...")
# 注意：请根据保存路径加载 BPE 分词器
texts_placeholder = ["def foo(): pass"]  # 仅用于构造 tokenizer 实例结构
tok = BPETokenizer(texts_placeholder)
tok = BPETokenizer.load("bpe_tokenizer.json")

print("加载模型...")
model = GPTModel(tok.tokenizer.get_vocab_size()).to(device)
model.load_state_dict(torch.load("gpt_model.pth", map_location=device))
model.eval()


# 文本生成函数
def generate(prompt, max_new_tokens=100):
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


# 示例使用
if __name__ == "__main__":
    prompt = "def hello_world():\n    print(\"Hello"
    print("生成结果:")
    print(generate(prompt))
