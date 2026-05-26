import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_batch_sentence_labeling():
    # 1. 경로 설정
    # 이전 단계에서 형태소 분석기로 분할한 문장 데이터
    input_data_path = "data/sentence_segmented_data.csv" 
    # 문장별 라벨링 결과가 저장될 중간 데이터 파일
    output_data_path = "data/sentence_labeled_data.csv" 
    # 팀에서 직접 파인튜닝을 완료한 모델 폴더 경로
    model_path = "./models/yogijogi_kcelectra_model" 

    if not os.path.exists(input_data_path):
        print(f"[에러] 입력 파일을 찾을 수 없습니다: {input_data_path}")
        return
    if not os.path.exists(model_path):
        print(f"[에러] 학습된 모델을 찾을 수 없습니다: {model_path}")
        return

    print("1. 자체 학습 모델 및 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
# [수정 시작] 5개 클래스(0~4) 중 2번 인덱스를 제거하고 4개 클래스로 재매핑
    with torch.no_grad():
        # 기존 가중치 저장
        old_weights = model.classifier.out_proj.weight.clone()
        old_bias = model.classifier.out_proj.bias.clone()
        
        # 2번 인덱스(3번 라벨)를 제외한 나머지 인덱스 리스트
        new_indices = [0, 1, 3, 4]
        
        # 새로운 텐서 생성
        model.classifier.out_proj.weight = torch.nn.Parameter(old_weights[new_indices])
        model.classifier.out_proj.bias = torch.nn.Parameter(old_bias[new_indices])
        model.config.num_labels = 4
    # [수정 끝]

    # GPU 가용 여부 확인 및 할당
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("2. 문장 단위 분할 데이터 로드 중...")
    df = pd.read_csv(input_data_path)
    
    # 결측치 제거 및 리스트 변환
    df = df.dropna(subset=['sentence'])
    texts = df['sentence'].tolist()

    # 라벨 번호 매핑 딕셔너리 (팀의 실제 학습 모델 설정에 맞게 수정 필요)
    label_map = {
        0: "1번 (고즈넉/사색)",
        1: "2번 (레트로/빈티지)",
        2: "4번 (청량/애니메이션)",
        3: "5번 (아기자기/소박)"
    }

    print(f"3. 총 {len(texts)}개 문장에 대한 AI 배치 추론 시작 (사용 장비: {device})...")
    predicted_labels = []
    
    # 질문자님이 제안해주신 메모리 효율적인 배치(Batch) 처리 도입
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # 확률이 가장 높은 클래스의 인덱스 추출
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # 인덱스를 텍스트 키워드로 변환하여 저장
            predicted_labels.extend([label_map[p] for p in preds])

        # 진행 상황 출력
        if (i + batch_size) % 320 == 0 or (i + batch_size) >= len(texts):
            print(f"진행도: {min(i + batch_size, len(texts))} / {len(texts)}")

    # 4. 예측된 라벨을 원본 데이터프레임에 추가
    df['sentence_label'] = predicted_labels

    print("4. 문장별 라벨링 결과 저장 중...")
    os.makedirs(os.path.dirname(output_data_path) or '.', exist_ok=True)
    df.to_csv(output_data_path, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"라벨링 완료. 결과가 '{output_data_path}'에 저장되었습니다.")

if __name__ == "__main__":
    run_batch_sentence_labeling()