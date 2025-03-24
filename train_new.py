import os
import torch
import argparse
import transformers as tfr
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# 参数配置
parser = argparse.ArgumentParser(description="Train VGCN-BERT on URL dataset")
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
parser.add_argument('--save_dir', type=str, default='exp_bertram+vgcnbert_50%', help='Directory to save model and logs')
parser.add_argument('--cuda', type=str, default='cuda:0', help='Training GPU')
parser.add_argument('--early_stop_metric', type=str, default='val_loss', choices=['val_loss', 'val_accuracy'],
                    help='Metric to monitor for early stopping (val_loss or val_accuracy)')
args = parser.parse_args()

# 设置保存目录和日志文件
os.makedirs(args.save_dir, exist_ok=True)
log_file = os.path.join(args.save_dir, 'training_log_vgcn+best_acc.txt')

device = torch.device(args.cuda)

# 日志记录函数
def log_message(message, log_file=log_file):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# 数据集类
class URLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded_input = self.tokenizer(
            text, return_tensors='pt', truncation=True, padding='max_length', max_length=128
        )
        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 使用 DistilBERT 的分词器
tokenizer = tfr.AutoTokenizer.from_pretrained("distilbert-base-uncased")
vgcn_bert_instance = tfr.AutoModel.from_pretrained("./vgcn-bert-distilbert-base-uncased", trust_remote_code=True)

# 定义分类器模型类
class VGCNClassifier(nn.Module):
    def __init__(self, vgcn_model, num_labels):
        super(VGCNClassifier, self).__init__()
        self.vgcn_model = vgcn_model
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.vgcn_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # 假设 outputs[0] 是隐藏状态
        logits = self.classifier(hidden_states[:, 0, :])
        return logits

# 生成词图并移动到指定设备
def generate_wgraph(model, train_texts, tokenizer, device):
    wgraph, wgraph_id_to_tokenizer_id_map = model.wgraph_builder(rows=train_texts, tokenizer=tokenizer)
    wgraph = wgraph.to(device)
    model.set_wgraphs([wgraph], [wgraph_id_to_tokenizer_id_map])

# 训练和评估函数
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = (batch['input_ids'].to(device),
                                             batch['attention_mask'].to(device),
                                             batch['labels'].to(device))

        assert input_ids.size(1) <= 512, \
            f"Input sequence length {input_ids.size(1)} exceeds max_position_embeddings {512}"

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_model(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, correct_predictions = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = (batch['input_ids'].to(device),
                                                 batch['attention_mask'].to(device),
                                                 batch['labels'].to(device))
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return avg_loss, accuracy

# 早停机制
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, metric='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = float('inf') if metric == 'val_loss' else -float('inf')
        self.early_stop = False
        self.metric = metric

    def __call__(self, val_metric):
        if self.metric == 'val_loss':
            if val_metric < self.best_metric - self.min_delta:
                self.best_metric = val_metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:  # 'val_accuracy'
            if val_metric > self.best_metric + self.min_delta:
                self.best_metric = val_metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

# 初始化模型、损失、优化器、设备
classifier_model = VGCNClassifier(vgcn_model=vgcn_bert_instance, num_labels=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(patience=args.patience, metric=args.early_stop_metric)

# 加载数据
train_data = pd.read_csv('exp_bertram+vgcnbert_50%/data-50%r/Train_sampled_50.csv')
test_data = pd.read_csv('exp_bertram+vgcnbert_50%/data-50%r/Test_sampled_50.csv')
df = pd.concat([train_data, test_data], ignore_index=True)
df['label'] = df['label'].map({'malicious': 1, 'benign': 0})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

# Without bertram
train_loader = DataLoader(URLDataset(train_df['url'].tolist(), train_df['label'].tolist(), tokenizer),
                          batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(URLDataset(val_df['url'].tolist(), val_df['label'].tolist(), tokenizer),
                        batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(URLDataset(test_df['url'].tolist(), test_df['label'].tolist(), tokenizer),
                         batch_size=args.batch_size, shuffle=False)

# 生成词图
generate_wgraph(classifier_model.vgcn_model, train_df['url'].tolist(), tokenizer, device)

# 开始训练
best_val_metric = -float('inf') if args.early_stop_metric == 'val_accuracy' else float('inf')

for epoch in range(args.num_epochs):
    train_loss = train_epoch(classifier_model, train_loader, optimizer, loss_fn, device)
    val_loss, val_accuracy = eval_model(classifier_model, val_loader, loss_fn, device)

    if args.early_stop_metric == 'val_loss':
        val_metric = val_loss
    else:
        val_metric = val_accuracy

    log_message(f"Epoch {epoch + 1}/{args.num_epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    if args.early_stop_metric == 'val_loss' and val_loss < best_val_metric:
        best_val_metric = val_loss
        torch.save(classifier_model.state_dict(), os.path.join(args.save_dir, f"vgcn_classifier+best_acc.pth"))
        log_message(f"Model checkpoint saved at epoch {epoch + 1}")

    elif args.early_stop_metric == 'val_accuracy' and val_accuracy > best_val_metric:
        best_val_metric = val_accuracy
        torch.save(classifier_model.state_dict(), os.path.join(args.save_dir, f"vgcn_classifier+best_acc.pth"))
        log_message(f"Model checkpoint saved at epoch {epoch + 1}")

    early_stopping(val_metric)
    if early_stopping.early_stop:
        log_message("Early stopping triggered")
        break
