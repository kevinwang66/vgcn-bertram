import os
import torch
import argparse
import transformers as tfr
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from modeling_vgcn_bert import VGCNBertConfig, VGCNBertForSequenceClassification

from bertram import BertramWrapper
import json

# 参数配置
parser = argparse.ArgumentParser(description="Train VGCN-BERT on URL dataset")
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
parser.add_argument('--save_dir', type=str, default='exp_seqclas_bertram+vgcnbert_50%', help='Directory to save model and logs')
parser.add_argument('--cuda', type=str, default='cuda:0', help='Training GPU')
parser.add_argument('--json_path', type=str, default='exp_bertram+vgcnbert_50%/data-50%r/processed_rare_words_with_contexts_400k_th=20_maxctx=5.json'
                    , help='Rare words with contexts')
args = parser.parse_args()

# 设置保存目录和日志文件
os.makedirs(args.save_dir, exist_ok=True)
log_file = os.path.join(args.save_dir, 'training_log_seqclas_vgcn+bertram.txt')

device = torch.device(args.cuda)
# 日志记录函数
def log_message(message, log_file=log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# +bertram
class URLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, rare_words_with_contexts):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.rare_words_with_contexts = rare_words_with_contexts  # 稀有词词典

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 获取原始文本和标签
        text = self.texts[idx]
        label = self.labels[idx]

        # 替换稀有词为 slash 形式
        text = self.replace_with_bertram(text)

        # 对文本进行编码
        encoded_input = self.tokenizer(
            text, return_tensors='pt', truncation=True, padding='max_length', max_length=128
        )

        # 返回模型所需的输入
        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def replace_with_bertram(self, text):
        """
        替换文本中的稀有词为slash形式。
        """
        for word in self.rare_words_with_contexts:
            # 如果稀有词在文本中出现，替换为slash形式
            slash_word = f"{word} / <BERTRAM:{word}>"
            text = text.replace(word, slash_word)
        return text


# 使用 DistilBERT 的分词器
tokenizer = tfr.AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 加载配置
config = VGCNBertConfig.from_pretrained(
    "./vgcn-bert-distilbert-base-uncased",
    num_labels=2
)

# 初始化模型（暂不指定词图）
classifier_model = VGCNBertForSequenceClassification.from_pretrained(
    "./vgcn-bert-distilbert-base-uncased",
    config=config
).to(device)

# 加载 BERTRAM 模型
bertram = BertramWrapper('./models/bertram-add-for-bert-base-uncased', device=args.cuda)

# 读取包含稀有词和上下文的 JSON 文件
def load_rare_words(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

file_path = args.json_path
words_with_contexts = load_rare_words(file_path)
bertram.add_word_vectors_to_model(words_with_contexts, tokenizer, classifier_model.vgcn_bert)

# 生成词图并移动到指定设备
def generate_wgraph(model, train_texts, tokenizer, device):
    wgraph, wgraph_id_to_tokenizer_id_map = model.wgraph_builder(rows=train_texts, tokenizer=tokenizer)
    wgraph = wgraph.to(device)
    model.set_wgraphs([wgraph], [wgraph_id_to_tokenizer_id_map])

# 训练与评估函数
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = (batch['input_ids'].to(device),
                                             batch['attention_mask'].to(device),
                                             batch['labels'].to(device))
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

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

# 早停机制
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 初始化模型、损失、优化器、设备
optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(patience=args.patience)

# 加载数据
train_data = pd.read_csv('exp_bertram+vgcnbert_50%/data-50%r/Train_sampled_50.csv').head(5000)
test_data = pd.read_csv('exp_bertram+vgcnbert_50%/data-50%r/Test_sampled_50.csv').head(5000)
df = pd.concat([train_data, test_data], ignore_index=True)
df['label'] = df['label'].map({'malicious': 1, 'benign': 0})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

# With bertram
train_loader = DataLoader(URLDataset(train_df['url'].tolist(), train_df['label'].tolist(), tokenizer, words_with_contexts),
                          batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(URLDataset(val_df['url'].tolist(), val_df['label'].tolist(), tokenizer, words_with_contexts),
                        batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(URLDataset(test_df['url'].tolist(), test_df['label'].tolist(), tokenizer, words_with_contexts),
                         batch_size=args.batch_size, shuffle=False)

# 生成词图
generate_wgraph(classifier_model.vgcn_bert, train_df['url'].tolist(), tokenizer, device)

# 开始训练
best_val_loss = float('inf')

for epoch in range(args.num_epochs):
    train_loss = train_epoch(classifier_model, train_loader, optimizer, device)
    val_loss, val_accuracy = eval_model(classifier_model, val_loader, device)

    log_message(f"Epoch {epoch + 1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        classifier_model.save_pretrained(args.save_dir)
        log_message(f"Model checkpoint saved at epoch {epoch + 1}")

    early_stopping(val_loss)
    if early_stopping.early_stop:
        log_message("Early stopping triggered")
        break