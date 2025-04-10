from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPETokenizer:
    '''
    初始化参数
    --------
    texts : list[str]
        用于训练分词器的文本列表（必须提供）
    vocab_size : int
        分词器词表大小，默认10000
    special_tokens : list[str]
        特殊token列表，默认包含一个结束标记 <EOS>
    '''

    def __init__(self, texts=None, vocab_size=10000, special_tokens=["<EOS>"]):
        if texts is not None:
            # 初始化 BPE 分词器模型
            self.tokenizer = Tokenizer(models.BPE())
            # 使用 ByteLevel 预处理器和解码器，保留缩进、换行等格式
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
            self.tokenizer.decoder = decoders.ByteLevel()

            # 配置 BPE 训练器
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
            )
            # 基于传入的文本训练分词器
            self.tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
        else:
            raise ValueError("You must provide texts to train the tokenizer.")

        # 保存 <EOS> 的 token ID，便于判断生成终止
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

    def encode(self, text):
        '''
        编码函数：将文本转换为 token ID 列表
        '''
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        '''
        解码函数：将 token ID 列表还原为字符串
        '''
        return self.tokenizer.decode(ids)

    def save(self, path):
        '''
        保存训练好的分词器到指定路径
        '''
        self.tokenizer.save(path)

    @staticmethod
    def load(path):
        '''
        加载已有分词器文件并返回 BPETokenizer 实例
        参数
        ----
        path : str
            .json 格式的 tokenizer 文件路径
        返回
        ----
        instance : BPETokenizer
            加载后的分词器对象
        '''
        tokenizer = Tokenizer.from_file(path)
        instance = BPETokenizer.__new__(BPETokenizer)
        instance.tokenizer = tokenizer
        instance.eos_id = tokenizer.token_to_id("<EOS>")
        return instance
