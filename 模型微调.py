import json
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 函数：将输出解析为二值标签向量
def parse_output(output):
    characteristics = ["专业化转型特征", "精细化转型特征", "特殊化转型特征", "新颖化转型特征"]
    if "未体现上述转型特征" in output:
        return [0, 0, 0, 0]
    else:
        start = output.find("体现了公司的") + len("体现了公司的")
        end = output.find("。") if "。" in output else len(output)
        mentioned = output[start:end].split("、")
        return [1 if char in mentioned else 0 for char in characteristics]

# 加载数据
with open('result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取文本和标签
texts = [entry['input'] for entry in data]
labels = [parse_output(entry['output']) for entry in data]

# 拆分为训练集（80%）和验证集（20%）
train_texts = texts[:80]
train_labels = labels[:80]
val_texts = texts[80:]
val_labels = labels[80:]

# 打印标签分布
train_labels_np = np.array(train_labels)
val_labels_np = np.array(val_labels)
print("训练集标签计数:", train_labels_np.sum(axis=0))
print("验证集标签计数:", val_labels_np.sum(axis=0))

# 从本地路径初始化分词器和模型
local_model_path = r'D:\chinese-bert-wwm'
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertForSequenceClassification.from_pretrained(
    local_model_path,
    num_labels=4,
    problem_type="multi_label_classification"
)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 对文本进行分词
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# 自定义数据集类
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# 创建数据集
train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

# 计算正类权重以处理类别不平衡
train_labels_np = np.array(train_labels)
label_counts = train_labels_np.sum(axis=0)
total_samples = len(train_labels_np)
pos_weight = (total_samples - label_counts) / label_counts  # 正类权重 = 负样本数 / 正样本数
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float).to(device)

# 定义评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    preds = (pred.predictions > 0).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 自定义 Trainer，使用加权损失
class CustomTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 初始化自定义 Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    pos_weight=pos_weight_tensor,
)

# 训练模型
trainer.train()

# 在验证集上评估
eval_results = trainer.evaluate()
print("验证集评估结果:", eval_results)

# 保存微调后的模型
trainer.save_model("./fine_tuned_model")