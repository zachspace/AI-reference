#BERT 是一种自然语言处理(NLP)中的深度学习模型，通常使用深度学习框架(TensorFlow、PyTorch 等)来实现。

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 读取数据
df = pd.read_csv('data.csv')

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), random_state=42)

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 定义数据集和数据加载器
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

epochs = 5
for epoch in range(epochs):
    train_loss = 0
    train_acc = 0
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = loss_fn(outputs[1], labels)

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

        train_acc += (outputs[1].argmax(1) == labels).sum().item()

    train_loss /= len(train_loader)
    train_acc /= len(train_dataset)

    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = loss_fn(outputs[1], labels)

            val_loss += loss.item()

            val_acc += (outputs[1].argmax(1) == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc /= len(val_dataset)

    print(f'Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')

# 在测试集上评估模型
test_texts = ['This is a positive text.', 'This is a negative text.']
test_labels = [1, 0]

test_dataset = TextDataset(test_texts, test_labels)
test_loader = DataLoader(test_dataset, batch_size=2)

model.eval()
with torch.no_grad():
    test_preds = []
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        test_preds.extend(outputs[0].argmax(1).tolist())

print(f'Test accuracy: {accuracy_score(test_labels, test_preds):.4f}')
print(f'Test precision, recall, f1-score: {precision_recall_fscore_support(test_labels, test_preds, average="binary")}')


#这个代码示例可以让您了解如何使用 BERT 进行文本分类 。
#在 Transformers 库中使用 BERT 非常方便。
#但是，您可能需要针对特定的应用场景对代码进行修改和调整。

