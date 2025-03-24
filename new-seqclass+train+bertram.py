import os
import torch
import argparse
import pandas as pd
import json
from tqdm import tqdm

import transformers as tfr
from transformers import get_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from modeling_vgcn_bert import VGCNBertConfig, VGCNBertForSequenceClassification
from transformers import BertModel
from bertram import BertramWrapper


# 参数配置
parser = argparse.ArgumentParser(description="Train VGCN-BERT on URL dataset")
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--max_norm', type=float, default=1.0, help='Clip grad norm')
parser.add_argument('--num_workers', type=int, default=4, help='num_workers')



# parser.add_argument('--save_dir', type=str, default='runs/-bertram_50%-distill', help='Directory to save model and logs')
# parser.add_argument('--train_data_dir', type=str, default='data/Train_Split_0.5.csv', help='Training data file')
# parser.add_argument('--test_data_dir', type=str, default='data/Test_Split_0.5.csv', help='Testing data file')

parser.add_argument('--save_dir', type=str, default='runs/-bertram_10%-distill', help='Directory to save model and logs')
parser.add_argument('--train_data_dir', type=str, default='data/Train_Split_0.1.csv', help='Training data file')
parser.add_argument('--test_data_dir', type=str, default='data/Test_Split_0.1.csv', help='Testing data file')
# parser.add_argument('--json_path', type=str, default='data/processed_rare_words_with_contexts_train0.1_th=20_maxctx=5.json'
#                     , help='Rare words with contexts')

parser.add_argument('--json_path', type=str, default='data/processed_rare_words_with_contexts_train0.5_th=20_maxctx=5.json'
                    , help='Rare words with contexts')

parser.add_argument('--cuda', type=str, default='cuda:0', help='Training GPU')
args = parser.parse_args()

# 设置保存目录和日志文件
os.makedirs(args.save_dir, exist_ok=True)
log_file = os.path.join(args.save_dir, 'training_log+bertram.txt')
train_data_dir = args.train_data_dir
test_data_dir = args.test_data_dir
device = torch.device(args.cuda)

# 日志记录函数
def log_message(message, log_file=log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

class URLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, rare_words_with_contexts, add_bertram):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.rare_words_with_contexts = set(rare_words_with_contexts.keys())
        self.slash_token = self.tokenizer.tokenize('/')[0]
        self.add_bertram = add_bertram

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 获取原始文本和标签
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.add_bertram is True:
            # 替换稀有词并分词
            tokens = self.replace_with_bertram(text)
            text = "".join(tokens)
            text = self.selective_replace(text)

        # 将替换后的文本转为模型所需的输入格式
        encoded_input = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        )
        # # 获取 input_ids 并打印
        # input_ids = encoded_input['input_ids'].squeeze()  # 获取 input_ids
        # decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)  # 解码为文本
        # print("Decoded text:", decoded_text)  # 输出解码后的文本

        # 返回模型所需的输入
        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def replace_with_bertram(self, text):
        """
        对文本进行分词，替换稀有词及其子词为带有BERTRAM标记的形式。
        """
        tokens = self.tokenizer.tokenize(text)
        print('Before replace (tokens):', tokens)
        replaced_tokens = []
        i = 0

        while i < len(tokens):
            token = tokens[i]
            replaced_tokens.append(token)
            
            if token in self.rare_words_with_contexts:
                replaced_tokens.append(self.slash_token)
                replaced_tokens.append(f"<BERTRAM:{token}>")

            i += 1

        return replaced_tokens

    def selective_replace(self, text):
        result = []
        in_bertram_tag = False  # 标记是否在 <BERTRAM:##...> 中
        i = 0
        while i < len(text):
            # 进入 <BERTRAM:## 的情况
            if text[i:i+11] == '<BERTRAM:##' and not in_bertram_tag:
                in_bertram_tag = True
                result.append(text[i:i+11])  # 添加 `<BERTRAM:##`
                i += 11  # 跳过 `<BERTRAM:##`

            # 退出 <BERTRAM:## 的情况
            elif in_bertram_tag and text[i] == '>':
                in_bertram_tag = False
                result.append(text[i])  # 添加 `>`
                i += 1

            # 删除其他地方的 ##
            elif text[i:i+2] == '##' and not in_bertram_tag:
                i += 2  # 跳过 ##

            else:
                result.append(text[i])
                i += 1

        return ''.join(result)

# 使用 BERT 的分词器
tokenizer = tfr.AutoTokenizer.from_pretrained("./models/distilbert-base-uncased")

# 加载配置
config = VGCNBertConfig.from_pretrained(
    "./models/vgcn-bert-distilbert-base-uncased",
    num_labels=2,
)

# 初始化模型
classifier_model = VGCNBertForSequenceClassification(config=config)

# 加载 BERTRAM 模型
bertram = BertramWrapper('./models/bertram-add-for-bert-base-uncased', device=args.cuda)

# 读取包含稀有词和上下文的 JSON 文件
def load_rare_words(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

file_path = args.json_path
words_with_contexts = load_rare_words(file_path)
bertram.add_word_vectors_to_model(words_with_contexts, tokenizer, classifier_model)

# 生成词图并移动到指定设备
def generate_wgraph(model, train_texts, tokenizer, device):
    wgraph, wgraph_id_to_tokenizer_id_map = model.wgraph_builder(rows=train_texts, tokenizer=tokenizer)
    wgraph = wgraph.to(device)
    model.set_wgraphs([wgraph], [wgraph_id_to_tokenizer_id_map])

# 早停机制
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode="max"):
        """
        Early stopping to terminate training when validation performance stops improving.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change to qualify as an improvement.
            mode (str): "max" to maximize the metric (e.g., accuracy), "min" to minimize (e.g., loss).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_score = float('-inf') if mode == "max" else float('inf')
        self.mode = mode

    def __call__(self, current_score):
        """
        Checks whether early stopping should be triggered.

        Args:
            current_score (float): The current validation metric (accuracy or loss).
        """
        if self.mode == "max":
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == "min":
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

    def reset(self):
        """Reset early stopping state for reuse."""
        self.counter = 0
        self.early_stop = False
        self.best_score = float('-inf') if self.mode == "max" else float('inf')
        
# 初始化模型、损失、优化器、设备
classifier_model.to(device)
optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
early_stopping = EarlyStopping(patience=args.patience, min_delta=0, mode="max")

train_data = pd.read_csv(train_data_dir)
test_data = pd.read_csv(test_data_dir)
test_df, val_df = train_test_split(test_data, test_size=0.05, random_state=42)

# With bertram
train_loader = DataLoader(URLDataset(train_data['url'].tolist(), train_data['label'].tolist(), tokenizer, words_with_contexts, add_bertram=False),batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(URLDataset(val_df['url'].tolist(), val_df['label'].tolist(), tokenizer, words_with_contexts, add_bertram=False),batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
# test_loader = DataLoader(URLDataset(test_df['url'].tolist(), test_df['label'].tolist(), tokenizer, words_with_contexts),
#                          batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

print(train_data['label'].value_counts())
print(val_df['label'].value_counts())

# 生成词图
generate_wgraph(classifier_model.vgcn_bert, train_data['url'].tolist(), tokenizer, device)

# 训练
# best_val_loss = float('inf')

num_training_steps = len(train_loader) * args.num_epochs
num_warmup_steps = int(0.1 * num_training_steps)  # Warmup 步骤占总步骤的 10%

# 创建学习率调度器
lr_scheduler = get_scheduler(
    # name="linear",
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

def train_epoch(model, dataloader, optimizer, lr_scheduler, device):
    model.train()
    total_loss, correct_predictions = 0, 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        assert input_ids.size(1) <= 512, \
            f"Input sequence length {input_ids.size(1)} exceeds max_position_embeddings {512}..."

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        logits = output.logits

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)

        optimizer.step()

        # 更新学习率
        lr_scheduler.step()

        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / len(dataloader.dataset)

    return avg_loss, accuracy


def eval_model(model, dataloader, device):
    model.eval()
    total_loss, correct_predictions = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = (batch['input_ids'].to(device),
                                                 batch['attention_mask'].to(device),
                                                 batch['labels'].to(device))
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            logits = output.logits
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return avg_loss, accuracy

for epoch in range(args.num_epochs):
    current_lr = optimizer.param_groups[0]['lr']
    log_message(f"Epoch {epoch + 1}/{args.num_epochs} | Current Learning Rate: {current_lr:.8f}")
    
    train_loss, train_accuracy = train_epoch(classifier_model, train_loader, optimizer, lr_scheduler, device)
    val_loss, val_accuracy = eval_model(classifier_model, val_loader, device)

    log_message(f"Epoch {epoch + 1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    
    torch.save(classifier_model.state_dict(), os.path.join(args.save_dir, f"vgcnbert_epoch:{epoch + 1}_train_l:{train_loss:.4f}_train_acc:{train_accuracy:.4f}_val_l:{val_loss:.4f}_val_acc{val_accuracy:.4f}.pth"))
    log_message(f"Model checkpoint saved at epoch {epoch + 1}")

    early_stopping(val_accuracy)
    if early_stopping.early_stop:
        log_message(f"Early stopping triggered at epoch {epoch + 1}")
        break
