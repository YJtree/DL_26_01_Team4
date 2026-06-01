import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    ElectraForSequenceClassification, 
    ElectraTokenizerFast,             
    get_cosine_schedule_with_warmup,  
)
from sklearn.metrics import classification_report, f1_score, accuracy_score

# 모델 크기 불일치(pre-trained head vs new head) 등의 불필요한 경고창을 숨깁니다.
warnings.filterwarnings("ignore")

# 외부 스크립트(model_prep.py)에서 전처리가 완료된 학습/검증 데이터를 불러오는 함수
from model_prep import prepare_finetuning_dataset

# ---------------------------------------------------------------------
# SECTION 1 : 재현성(Reproducibility) 고정 설정
# ---------------------------------------------------------------------
def set_seed(seed: int = 42):
    """
    딥러닝은 가중치 초기화나 미니배치 샘플링 시 난수(Random)를 사용합니다.
    시드(Seed)를 고정하지 않으면 돌릴 때마다 성능이 달라져 실험 결과를 신뢰할 수 없습니다.
    Python, NumPy, PyTorch(CPU/GPU)의 모든 난수 생성 엔진의 시드를 42로 통일합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN 연산 시 비결정론적(nondeterministic) 알고리즘 사용을 막아 결과를 완벽히 고정합니다.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   

set_seed(42)

# 하이퍼파라미터 세팅
MODEL_NAME = "beomi/KcELECTRA-base-v2022"
SAVE_PATH  = "./models/yogijogi_kcelectra_model"
NUM_LABELS = 4         
BATCH_SIZE = 16        # 한 번에 GPU에 올라가는 문장 개수
EPOCHS = 30            # 전체 데이터셋을 반복 학습할 최대 횟수
LEARNING_RATE = 2e-5   # Pre-trained 언어모델 파인튜닝 시 가장 많이 쓰이는 안전한 학습률
WEIGHT_DECAY = 0.01    # L2 정규화 (가중치가 너무 커지는 것을 막아 과적합 방지)
WARMUP_RATIO = 0.1     # 학습 초반 10% 스텝 동안은 학습률을 서서히 올림 (급격한 가중치 파괴 방지)
PATIENCE = 7           # 검증 성능이 7에폭 동안 안 오르면 조기 종료
GRAD_CLIP = 1.0        # 기울기 폭발(Gradient Exploding)을 막기 위해 기울기 최대치 제한
LABEL_SMOOTHING = 0.05 # 정답을 1.0이 아닌 0.95로 부드럽게 주어 과도한 확신 방지

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# SECTION 2 : 클래스 가중치 계산 함수 (동적 할당) 
# ---------------------------------------------------------------------
def compute_class_weights(labels_list: list, num_classes: int) -> torch.Tensor:
    """
    데이터 불균형 해결을 위한 Inverse Frequency Weighting 로직입니다.
    엑셀에서 추출한 `train_labels`를 직접 카운트하여 정확한 가중치를 알아서 계산
    
    * 계산식: 전체 샘플 수 / (클래스 개수 * 해당 클래스 샘플 수)
    이 식은 scikit-learn의 'balanced' 가중치 계산 방식과 수학적으로 100% 동일합니다.
    적은 데이터(예: 3번 청량)는 가중치가 1.5 이상으로 커져서 틀렸을 때 Loss가 폭발적으로 증가하게 만듭니다.
    """
    label_array = np.array(labels_list)
    unique_labels, counts = np.unique(label_array, return_counts=True)
    
    class_counts = [0] * num_classes
    for label, count in zip(unique_labels, counts):
        class_counts[int(label)] = int(count)
        
    total = sum(class_counts)
    # c가 0일 경우 ZeroDivisionError를 막기 위한 안전장치 추가
    weights = [total / (num_classes * c) if c > 0 else 1.0 for c in class_counts]
    
    print("\n[동적 클래스 가중치 계산 결과]")
    for i in range(num_classes):
        print(f"  라벨 {i} | 샘플 수: {class_counts[i]:>4}개 -> Loss 가중치: {weights[i]:.4f}")
        
    # PyTorch의 CrossEntropyLoss에 넣기 위해 FloatTensor 형태로 GPU에 올립니다.
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# ---------------------------------------------------------------------
# SECTION 3 : Dataset 변환 및 평가 함수
# ---------------------------------------------------------------------
class TravelDataset(Dataset):
    """
    Hugging Face 토크나이저의 결과물(dict 형태)을 
    PyTorch DataLoader가 읽을 수 있도록 Tensor로 감싸주는 래퍼(Wrapper) 클래스입니다.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def evaluate(model, data_loader, criterion):
    """
    검증(Validation) 단계입니다.
    학습이 아니므로 model.eval()과 torch.no_grad()를 사용하여 
    Dropout 비활성화 및 기울기(Gradient) 계산 메모리 점유를 막습니다.
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # 예측값과 정답 간의 오차(Loss) 계산
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            
            # Logits(확률값) 중 가장 큰 값의 인덱스를 뽑아 최종 예측 라벨로 사용
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    # 데이터 불균형이 있으므로 단순 Accuracy가 아닌 Macro-F1 스코어를 메인 지표로 삼습니다.
    return avg_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")

# ---------------------------------------------------------------------
# SECTION 4 : 메인 학습 루프
# ---------------------------------------------------------------------
def main():
    print("\n[데이터] model_prep.py 에서 데이터를 불러옵니다...")
    # 전처리 스크립트 실행 (토크나이징 및 Train/Val 분할)
    train_encodings, val_encodings, train_labels, val_labels = prepare_finetuning_dataset()

    # [핵심] 실제 train_labels를 가지고 모델에 주입할 가중치 텐서를 생성
    CLASS_WEIGHTS = compute_class_weights(train_labels, NUM_LABELS)

    train_dataset = TravelDataset(train_encodings, train_labels)
    val_dataset = TravelDataset(val_encodings, val_labels)

    # 훈련용은 섞어주고(shuffle=True), 검증용은 굳이 섞을 필요 없이 배치만 2배로 키워 빠르게 추론
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    print(f"\n[모델] {MODEL_NAME} 로드 중...")
    tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_NAME)
    # 분류기 헤드가 부착된 ELECTRA 모델 로드
    model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
    model.to(DEVICE)   

    # 손실 함수에 가중치(weight) 삽입. 
    # 이제 모델이 적은 라벨을 틀리면 Loss가 크게 증폭되어 역전파(Backpropagation) 시 더 강하게 교정됩니다.
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTHING)

    # Weight Decay(L2 정규화) 최적화
    # bias나 LayerNorm에는 Weight Decay를 적용하지 않는 것이 Transformer 학습의 정석입니다.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    
    # Cosine 스케줄러 & Warmup
    # 처음에는 학습률을 서서히 올리다가(Warmup), 이후 코사인 곡선을 그리며 부드럽게 낮춥니다.
    total_steps = len(train_loader) * EPOCHS       
    warmup_steps = int(total_steps * WARMUP_RATIO)  
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # AMP (Automatic Mixed Precision)
    # Float32 연산을 Float16으로 섞어서 사용하여 GPU 메모리를 아끼고 학습 속도를 비약적으로 높입니다.
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1, best_val_acc, patience_counter, best_model_state = 0.0, 0.0, 0, None    

    print("\n학습 시작...")
    for epoch in range(1, EPOCHS + 1):
        model.train()        
        total_train_loss, train_preds_all, train_labels_all = 0.0, [], []

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)

            optimizer.zero_grad() # 이전 배치의 기울기 초기화

            # AMP 캐스팅 적용 구역 (순전파)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = criterion(outputs.logits, labels)  

            # AMP 스케일러를 통한 역전파 (FP16 언더플로우 방지)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP) # 기울기 폭발 제어
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # 학습률 업데이트

            total_train_loss += loss.item()
            train_preds_all.extend(torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy())
            train_labels_all.extend(labels.cpu().numpy())

        # 에폭 단위 평가
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc, train_f1 = accuracy_score(train_labels_all, train_preds_all), f1_score(train_labels_all, train_preds_all, average="macro")
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")

        # 모델 저장 로직 (가장 좋은 F1 스코어 기준)
        if val_f1 > best_val_f1:
            best_val_f1, best_val_acc, patience_counter = val_f1, val_acc, 0
            # 현재 가장 좋은 가중치를 메모리에 복사해 둠
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            # Early Stopping (과적합이 시작되었다고 판단되면 학습 강제 종료)
            if patience_counter >= PATIENCE:
                print(f"조기 종료: {PATIENCE}에폭 연속 개선 없음.")
                break

    # 메모리에 저장해둔 '가장 성능이 좋았던 에폭'의 가중치로 롤백
    print(f"\n최고 모델 복원 중 (Val F1: {best_val_f1:.4f})...")
    model.load_state_dict(best_model_state)   
    
    # 최종 결과물 로컬 디스크에 저장
    os.makedirs(SAVE_PATH, exist_ok=True)    
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("저장 완료!")

if __name__ == "__main__":
    main()