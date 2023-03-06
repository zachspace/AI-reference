#BERT自然语言推断 NL Inferencee（NLI）的示例代码。
#安装transformers库和其他必要的库：

python
Copy code
!pip install transformers
!pip install torch
!pip install numpy

#使用transformers库中的BertForSequenceClassification模型来进行推断。
#加载预训练的BERT模型和它的tokenizer
#将输入的句子编码成模型可以理解的格式。然后，我们将使用编码后的句子进行推断，并打印输出。

python
Copy code
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 输入的句子
sentence1 = "The cat is sitting on the mat."
sentence2 = "The dog is playing in the park."

# 将输入的句子编码成模型可以理解的格式
inputs = tokenizer.encode_plus(sentence1, sentence2, return_tensors='pt', pad_to_max_length=True, max_length=128)

# 进行推断
outputs = model(**inputs)

# 打印输出
print(outputs)
输出将是一个PyTorch张量，其中包含推断结果的类别分数。如果我们想要获得推断结果的类别标签，我们可以使用以下代码：

python
Copy code
# 获得类别分数
scores = outputs[0].detach().numpy()
# 将分数转换为类别标签
predicted_label = scores.argmax(axis=1)
print(predicted_label)


#输出将是一个包含推断结果的类别标签的NumPy数组。


