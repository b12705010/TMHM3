import os
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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

# 使用 bert-large-uncased 並確保輸出 hidden states 以取得多層特徵
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True)
model.eval()

# 提取多層 [CLS] 特徵的函數 (最後4層平均)
def extract_bert_features(text, layers=[-1, -2, -3, -4]):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    # 平均最後四層的 [CLS] 向量
    selected_layers = [hidden_states[layer][:, 0, :].squeeze(0).numpy() for layer in layers]
    aggregated_features = np.mean(selected_layers, axis=0)
    return aggregated_features

def process_documents(doc_ids, data_folder, labels=None):
    embeddings = []
    doc_labels = []
    for doc_id in doc_ids:
        file_path = os.path.join(data_folder, f"{doc_id}.txt")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist!")
            continue
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()
            embedding = extract_bert_features(text)
            embeddings.append(embedding)
            if labels:
                doc_labels.append(labels.get(doc_id, -1))
    if labels:
        return np.array(embeddings), np.array(doc_labels)
    else:
        return np.array(embeddings), None

def ensure_2d(embeddings):
    return np.vstack(embeddings)

# 提取訓練集與測試集
train_embeddings, train_labels = process_documents(train_docs, data_folder, labels)
train_embeddings = ensure_2d(train_embeddings)

test_embeddings, _ = process_documents(test_docs, data_folder)
test_embeddings = ensure_2d(test_embeddings)

# 對特徵進行標準化
scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)

# 使用 GridSearch 調整 SVM 參數
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf']
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, scoring='f1_macro', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(train_embeddings_scaled, train_labels)

best_svm = grid_search.best_estimator_
test_predictions = best_svm.predict(test_embeddings_scaled)

# 保存結果 (不更動輸出格式)
results = pd.DataFrame({'Id': test_docs, 'Value': test_predictions})
results.to_csv('kaggle_submission_newTry.csv', index=False)
print("Results saved to kaggle_submission_newTry.csv!")