import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# 使用 roberta-large 並確保取得 hidden states
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
model = AutoModel.from_pretrained('roberta-large', output_hidden_states=True)
model.eval()

# 提取多層特徵向量（最後4層平均）
def extract_bert_features(text, layers=[-1, -2, -3, -4]):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
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
    return np.array(embeddings), np.array(doc_labels) if labels else np.array(embeddings)

def ensure_2d(embeddings):
    return np.vstack(embeddings)

# 提取訓練集與測試集的特徵
train_embeddings, train_labels = process_documents(train_docs, data_folder, labels)
train_embeddings = ensure_2d(train_embeddings)

test_embeddings, _ = process_documents(test_docs, data_folder)
test_embeddings = ensure_2d(test_embeddings)

# 標準化特徵，避免特徵值差異過大影響 SVM
scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)

# 使用 PCA 降維 (可嘗試不用或改變 n_components)
pca = PCA(n_components=128)
train_embeddings_pca = pca.fit_transform(train_embeddings_scaled)
test_embeddings_pca = pca.transform(test_embeddings_scaled)

# 若類別不平衡，考慮使用 class_weight='balanced'
# 調整 SVM 超參數
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, scoring='f1_macro', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(train_embeddings_pca, train_labels)

# 最佳 SVM
best_svm = grid_search.best_estimator_

# 測試集分類
test_predictions_svm = best_svm.predict(test_embeddings_pca)

# 保存 SVM 結果 (不更動輸出格式)
results_svm = pd.DataFrame({'Id': test_docs, 'Value': test_predictions_svm})
results_svm.to_csv('kaggle_submission_svm1210.csv', index=False)
print("SVM Results saved to kaggle_submission_svm1210.csv!")








'''
# xgb 還啥的只有 0.66 超級低，最好的是 現在 github 上的最早版本
import os
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

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

# 1. 使用 PCA 降維
pca = PCA(n_components=128)  # 調整至 128 維
train_embeddings_pca = pca.fit_transform(train_embeddings)
test_embeddings_pca = pca.transform(test_embeddings)

# 2. 調整 SVM 超參數
param_grid = {
    'C': [0.1, 1, 10, 100],  # 正則化參數
    'kernel': ['linear', 'rbf'],  # 核函數
    'gamma': ['scale', 'auto']  # 核函數係數
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, scoring='f1_macro', cv=5, verbose=2)
grid_search.fit(train_embeddings_pca, train_labels)

# 最佳 SVM
best_svm = grid_search.best_estimator_

# 測試集分類
test_predictions_svm = best_svm.predict(test_embeddings_pca)

# 保存 SVM 結果
results_svm = pd.DataFrame({'Id': test_docs, 'Value': test_predictions_svm})
results_svm.to_csv('kaggle_submission_svm.csv', index=False)
print("SVM Results saved to kaggle_submission_svm.csv!")

# 3. 使用 XGBoost
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)

xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
xgb_model.fit(train_embeddings_pca, train_labels_encoded)

# 測試集分類
test_predictions_xgb_encoded = xgb_model.predict(test_embeddings_pca)
test_predictions_xgb = label_encoder.inverse_transform(test_predictions_xgb_encoded)

# 保存 XGBoost 結果
results_xgb = pd.DataFrame({'Id': test_docs, 'Value': test_predictions_xgb})
results_xgb.to_csv('kaggle_submission_xgb.csv', index=False)
print("XGBoost Results saved to kaggle_submission_xgb.csv!")
'''