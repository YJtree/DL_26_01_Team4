import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from model_prep import prepare_finetuning_dataset

# 1. 데이터셋 클래스 (train.py와 동일)
class TravelDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_saved_model():
    print("1. 저장된 최종 모델과 검증 데이터 불러오는 중...")
    # 저장된 베스트 모델 경로 (본인의 폴더 구조에 맞게 수정)
    MODEL_PATH = "./models/yogijogi_kcelectra_model"
    
    # 디바이스 설정 (GPU가 있으면 GPU, 없으면 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 중인 디바이스: {device}")

    # 전처리된 데이터 로드 (검증 데이터만 사용)
    _, val_encodings, _, val_labels = prepare_finetuning_dataset()
    val_dataset = TravelDataset(val_encodings, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 저장된 모델 불러오기
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval() # 평가 모드 전환 (Dropout 등 비활성화)

    print("2. 검증 데이터 평가 진행 중 (1~2분 소요)...")
    all_preds = []
    all_labels = []

    # 기울기 계산 비활성화 (메모리 절약 및 속도 향상)
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 가장 높은 확률을 가진 라벨 인덱스 추출
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*50)
    print("🎯 [최종 평가 결과 도출 완료]")
    print("="*50)

    # 1. Macro F1 및 정확도(Accuracy) 계산
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"▶ 검증 데이터 Macro F1 Score : {macro_f1:.4f}")
    print(f"▶ 검증 데이터 Accuracy(정확도): {accuracy:.4f} ({accuracy*100:.1f}%)")

    # 2. 혼동 행렬 (Confusion Matrix) 출력
    print("\n▶ 혼동 행렬 (Confusion Matrix):")
    print("(가로: 모델 예측 / 세로: 실제 정답)")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # 3. 라벨별 상세 리포트
    print("\n▶ 라벨별 상세 분류 리포트:")
    target_names = ['0번(고즈넉)', '1번(레트로)', '2번(청량)', '3번(아기자기)']
    print(classification_report(all_labels, all_preds, target_names=target_names))

if __name__ == "__main__":
    evaluate_saved_model()