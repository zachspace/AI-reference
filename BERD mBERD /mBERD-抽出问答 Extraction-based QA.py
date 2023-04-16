#Extraction-based QA 抽出式问答

from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# 加载预训练的BERT模型和tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 输入问题和文本
question = "What is the capital of France?"
text = "The capital of France is Paris. It is also the country's largest city."

# 将问题和文本编码为BERT输入格式
input_ids = tokenizer.encode(question, text)

# 将输入传递给BERT模型
tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer_start_scores, answer_end_scores = model(torch.tensor([input_ids]))

# 获取起始和结束位置
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1

# 提取答案
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

# 输出提取的答案
print(answer)

