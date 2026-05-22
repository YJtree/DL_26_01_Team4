import torch
from torch import nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from model_prep import prepare_finetuning_dataset

# [1] 데이터셋 클래스 정의
# PyTorch와 Hugging Face Trainer가 데이터를 읽을 수 있도록 규격을 맞추는 클래스입니다.
# 토큰화된 텍스트(숫자 배열)와 정답 라벨을 하나의 딕셔너리 형태로 묶어줍니다.
class TravelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 인덱스(idx)에 해당하는 데이터를 하나씩 꺼내어 텐서(Tensor)로 변환합니다.
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 라벨은 정수형(long) 텐서로 변환하여 모델의 정답지로 제공합니다.
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    # 전체 데이터의 개수를 반환합니다.
    def __len__(self):
        return len(self.labels)

def train_model():
    print("1. 토큰화된 데이터 불러오는 중...")
    # model_prep.py에서 작성한 함수를 호출하여 8:2로 분할 및 토큰화된 데이터를 가져옵니다.
    train_encodings, val_encodings, train_labels, val_labels = prepare_finetuning_dataset()

    train_dataset = TravelDataset(train_encodings, train_labels)
    val_dataset = TravelDataset(val_encodings, val_labels)

    print("2. 데이터 불균형 해결을 위한 클래스 가중치 계산 중...")
    # [핵심] compute_class_weight 함수를 사용하여 라벨별 데이터 불균형을 계산합니다.
    # 데이터가 많은 클래스(예: 1번 고즈넉함)에는 낮은 가중치를, 
    # 데이터가 적은 클래스에는 높은 가중치를 자동으로 역산하여 부여합니다.
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    # 계산된 가중치를 PyTorch에서 연산할 수 있도록 실수형(float32) 텐서로 변환합니다.
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"적용될 가중치 비율 (1번~5번): {class_weights}")

    print("3. KcELECTRA 모델 로드 중 (전체 레이어 학습)...")
    # KcELECTRA 기본 모델을 불러옵니다. num_labels=5를 설정하여 
    # 분류기 헤드(머리)를 우리의 5가지 감성 분류 목적에 맞게 교체합니다.
    # 하위 레이어를 동결하지 않고 100% 전체를 학습시켜 데이터 불균형 환경에서도 깊은 특징을 배우도록 합니다.
    model = AutoModelForSequenceClassification.from_pretrained(
        "beomi/KcELECTRA-base-v2022",
        num_labels=5
    )

    print("4. 학습 하이퍼파라미터 세팅 (CPU 환경 및 과적합 방지 최적화)...")
    # 학습에 필요한 각종 설정값(하이퍼파라미터)을 정의합니다.
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=8,              # [수정] 에폭을 6에서 8로 조금 더 줍니다.        per_device_train_batch_size=16,  # 한 번에 모델에 들어가는 데이터의 개수 (Train)
        per_device_eval_batch_size=16,   # 한 번에 모델에 들어가는 데이터의 개수 (Validation)
        learning_rate=5e-5,              # [핵심 수정] 보폭을 2.5배 늘려 학습 속도를 극대화합니다 (2e-5 -> 5e-5).
        warmup_steps=0,                  # [핵심 수정] 웜업을 0으로 만들어 1에폭부터 전력질주하게 만듭니다.
        weight_decay=0.01,               # 특정 가중치가 너무 커지는 것을 막는 정규화 기법
        logging_dir='./logs',            # 텐서보드 등 로그가 저장될 폴더
        eval_strategy="epoch",           # 1 에폭(반복)이 끝날 때마다 검증(Validation)을 수행
        save_strategy="epoch",           # 1 에폭이 끝날 때마다 모델 상태를 저장
        load_best_model_at_end=True,     # [핵심] 조기 종료 시 가장 성능이 좋았던 시점의 가중치를 불러옵니다.
        metric_for_best_model="loss"     # 최고 모델의 기준을 검증 오차(loss) 최소화로 설정
    )

    print("5. 커스텀 트레이너 정의...")
    # [핵심] 기본 Trainer를 상속받아 오차 계산(Loss) 방식을 프로젝트 목적에 맞게 덮어씁니다.
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # 입력 데이터에서 엑셀에 적어둔 정답(labels)을 추출합니다.
            labels = inputs.get("labels")
            
            # 모델에 데이터를 넣고 예측값(outputs)을 뽑아냅니다.
            outputs = model(**inputs)
            
            # 모델이 예측한 각 클래스별 0~4번 확률의 원시값(logits)입니다.
            logits = outputs.get("logits")
            
            # 가중치 텐서를 넘겨주어, 희귀 데이터를 틀렸을 때 더 큰 벌점을 주도록 강력하게 설정합니다.
            loss_fct = nn.CrossEntropyLoss(weight=weights_tensor.to(model.device))
            
            # 예측값과 실제 정답을 비교하여 최종 오차(Loss)를 계산합니다.
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss

    print("6. 커스텀 트레이너 객체 생성 및 학습 시작...")
    # 우리가 만든 CustomTrainer에 모델, 설정값, 데이터셋, 콜백(조기 종료)을 모두 집어넣고 학습을 준비합니다.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # [수정] 조기 종료 인내심을 2에서 3으로 늘려 섣불리 종료되지 않게 막아줍니다.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
    )

    # 본격적인 학습을 시작합니다.
    trainer.train()

    print("7. 학습된 최종 모델 저장 중...")
    # 학습이 완료된 모델의 가중치를 지정된 폴더에 영구적으로 저장합니다.
    model.save_pretrained("./models/yogijogi_kcelectra_model")
    print("완료: 모델이 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    train_model()