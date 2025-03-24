import torch
import os
import argparse
import pandas as pd
import re
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import transformers as tfr
import torch.nn as nn
from tqdm import tqdm

# 参数配置
parser = argparse.ArgumentParser(description="Evaluate VGCN-BERT on URL dataset")
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
# parser.add_argument('--model_path', type=str, default="exp_bertram+vgcnbert_50%/vgcn_classifier.pth", help='Path to the saved model checkpoint')
parser.add_argument('--model_path', type=str, default="/home/wza/VGCNBert/exp2_without_tokenizing_50%/vgcn_classifier_epoch_2.pth", help='Path to the saved model checkpoint')
parser.add_argument('--cuda', type=str, default='cuda:0', help='Training GPU')
args = parser.parse_args()

# 设置设备
device = torch.device(args.cuda)

class VGCNClassifier(nn.Module):
    def __init__(self, vgcn_model, num_labels):
        super(VGCNClassifier, self).__init__()
        self.vgcn_model = vgcn_model
        self.classifier = nn.Linear(vgcn_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.vgcn_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # 假设 outputs[0] 是隐藏状态
        logits = self.classifier(hidden_states[:, 0, :])
        return logits

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

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 加载模型
vgcn_bert_instance = tfr.AutoModel.from_pretrained("./vgcn-bert-distilbert-base-uncased", trust_remote_code=True)
classifier_model = VGCNClassifier(vgcn_model=vgcn_bert_instance, num_labels=2).to(device)
classifier_model.load_state_dict(torch.load(args.model_path, map_location=device))
classifier_model.eval()

# 生成词图并移动到指定设备
def generate_wgraph(model, train_texts, tokenizer, device):
    wgraph, wgraph_id_to_tokenizer_id_map = model.wgraph_builder(rows=train_texts, tokenizer=tokenizer)
    wgraph = wgraph.to(device)
    model.set_wgraphs([wgraph], [wgraph_id_to_tokenizer_id_map])

# 加载测试数据
train_data = pd.read_csv('exp_bertram+vgcnbert_50%/data-50%r/Train_sampled_50.csv')
test_data = pd.read_csv('exp_bertram+vgcnbert_50%/data-50%r/Test_sampled_50.csv')
df = pd.concat([train_data, test_data], ignore_index=True)
df['label'] = df['label'].map({'malicious': 1, 'benign': 0})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 创建测试数据加载器
test_loader = DataLoader(URLDataset(test_df['url'].tolist(), test_df['label'].tolist(), tokenizer),
                         batch_size=args.batch_size, shuffle=False)

generate_wgraph(classifier_model.vgcn_model, train_df['url'].tolist(), tokenizer, device)
# generate_wgraph(classifier_model.vgcn_model, test_df['url'].tolist(), tokenizer, device)

# 评估函数
def evaluate_model(model, dataloader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 计算 F1 score
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")

    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['benign', 'malicious']))

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['benign', 'malicious'], yticklabels=['benign', 'malicious'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# 在测试集上评估模型性能
evaluate_model(classifier_model, test_loader, device)
