#Transformers 库中的 multi-lingual BERT 模型（即 mBERT）进行多语言文本分类。
#以下是一个简单的 Python 代码示例，该示例演示如何使用 mBERT 进行多语言情感分析：
#需要针对特定的应用场景对代码进行修改和调整。

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 选择 mBERT 模型和 tokenizer
model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载模型并将其移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# 准备数据集
texts = ['This is a positive text.', 'This is a negative text.']
labels = [1, 0]

# 对文本进行编码
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# 将数据划分为训练集和验证集
train_dataset = torch.utils.data.TensorDataset(encoded_texts['input_ids'], encoded_texts['attention_mask'], torch.tensor(labels))
train_loader = DataLoader(train_dataset, batch_size=2)

# 设置优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_acc += (outputs.logits.argmax(1) == labels).sum().item()

    train_loss /= len(train_loader)
    train_acc /= len(train_dataset)

    print(f'Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')

# 在测试集上评估模型
test_texts = ['This is a positive text.', 'This is a negative text.']
test_labels = [1, 0]

test_dataset = torch.utils.data.TensorDataset(tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')['input_ids'], tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')['attention_mask'], torch.tensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size=2)

model.eval()
with torch.no_grad():
    test_preds = []
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        test_preds.extend(outputs.logits.argmax(1).tolist())

print(f'Test accuracy: {accuracy_score(test_labels, test_preds):.4f}')
print(f'Test precision, recall, f1-score: {precision_recall_fscore_support(test_labels, test_preds, average="binary")}')