import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# 檢查 MPS (Apple Silicon GPU) 是否可用
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

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

# 使用 roberta-large 並確保輸出 hidden_states 以取得多層特徵
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
model = AutoModel.from_pretrained('roberta-large', output_hidden_states=True)
model.to(device)
model.eval()

# 提取多層 [CLS] 特徵的函數 (最後4層平均)
def extract_bert_features(text, layers=[-1, -2, -3, -4]):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    # 平均最後四層的 [CLS] 向量
    selected_layers = [hidden_states[layer][:, 0, :].squeeze(0).cpu().numpy() for layer in layers]
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

# 提取訓練集與測試集
train_embeddings, train_labels = process_documents(train_docs, data_folder, labels)
test_embeddings, _ = process_documents(test_docs, data_folder)

# 對特徵進行標準化
scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)

# 使用更廣的超參數搜尋範圍
# 加入 gamma（針對 rbf kernel）、並嘗試 class_weight='balanced'以因應資料不均衡
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, scoring='f1_macro', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(train_embeddings_scaled, train_labels)

best_svm = grid_search.best_estimator_
test_predictions = best_svm.predict(test_embeddings_scaled)

# 保存結果 (不更動輸出格式)
results = pd.DataFrame({'Id': test_docs, 'Value': test_predictions})
results.to_csv('kaggle_submission_newTry2.csv', index=False)
print("Results saved to kaggle_submission_newTry2.csv!")

# 輸出最佳參數供參考
print("Best Parameters:", grid_search.best_params_)
print("Best F1 on training set:", grid_search.best_score_)