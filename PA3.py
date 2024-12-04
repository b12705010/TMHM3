import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 設置資料夾與檔案
data_folder = './TM/'
num_docs = 1095  # TM 中的總檔案數量
label_file = './training_new.txt'

# 讀取標籤
def load_labels(label_file):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            class_id, doc_ids = line.strip().split(' ', 1)
            for doc_id in doc_ids.split():
                labels[int(doc_id)] = int(class_id)
    return labels

labels = load_labels(label_file)

# 獲取測試集的檔案編號
all_docs = set(range(1, num_docs + 1))
train_docs = set(labels.keys())
test_docs = list(all_docs - train_docs)

# 載入 BERT 模型與分詞器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# 提取特徵向量函數
def extract_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()  # [CLS] 向量
    return cls_embedding

# 提取所有文檔的 [CLS] 特徵
def process_documents(doc_ids, data_folder, labels=None):
    embeddings = []
    doc_labels = []
    for doc_id in doc_ids:
        file_path = os.path.join(data_folder, f"{doc_id}.txt")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist!")
            continue
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            embedding = extract_cls_embedding(text)  # 提取 [CLS] 特徵
            embeddings.append(embedding)
            if labels:
                doc_labels.append(labels.get(doc_id, -1))  # 如果有標籤，附加類別
    return np.array(embeddings), np.array(doc_labels) if labels else np.array(embeddings)

# 修正：確保 `embeddings` 維度為 2D
def ensure_2d(embeddings):
    return np.vstack(embeddings)  # 將所有特徵向量堆疊為 2D 矩陣

# 提取訓練集與測試集
train_embeddings, train_labels = process_documents(train_docs, data_folder, labels)
train_embeddings = ensure_2d(train_embeddings)

test_embeddings, _ = process_documents(test_docs, data_folder)
test_embeddings = ensure_2d(test_embeddings)

# 訓練 SVM
classifier = SVC(kernel='linear')
classifier.fit(train_embeddings, train_labels)

# 測試集分類
test_predictions = classifier.predict(test_embeddings)

# 保存結果至 kaggle_submission.csv
results = pd.DataFrame({'Id': test_docs, 'Value': test_predictions})
results.to_csv('kaggle_submission.csv', index=False)
print("Results saved to kaggle_submission.csv!")