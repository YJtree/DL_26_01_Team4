import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_batch_sentence_labeling():
    # 1. 경로 설정
    input_data_path = "data/sentence_segmented_data.csv" 
    output_data_path = "data/sentence_labeled_data.csv" 
    model_path = "./models/yogijogi_kcelectra_model" 

    if not os.path.exists(input_data_path):
        print(f"[에러] 입력 파일을 찾을 수 없습니다: {input_data_path}")
        return
    if not os.path.exists(model_path):
        print(f"[에러] 학습된 모델을 찾을 수 없습니다: {model_path}")
        return

    print("1. 자체 학습 4분류 모델 및 토크나이저 로드 중...")
    # 강제 변환 없이, 이미 4분류로 학습된 모델을 그대로 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("2. 문장 데이터 읽기...")
    df = pd.read_csv(input_data_path)
    # 결측치 제거
    df = df.dropna(subset=['sentence'])
    texts = df['sentence'].tolist()

    # 확정된 4분류 체계 라벨 맵핑 (이전의 3번이 2번에 통합되었음)
    label_map = {
        0: "1번 (고즈넉/사색)",
        1: "2번 (레트로/빈티지)",
        2: "3번 (청량/애니메이션)",
        3: "4번 (아기자기/소박)"
    }

    print(f"3. 총 {len(texts)}개 문장에 대한 AI 배치 추론 시작 (사용 장비: {device})...")
    predicted_labels = []
    
    # 메모리 효율적인 배치(Batch) 처리
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
            # 확률이 가장 높은 클래스의 인덱스(0~3) 추출
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # 인덱스를 직관적인 텍스트 라벨로 변환하여 리스트에 추가
            predicted_labels.extend([label_map[p] for p in preds])

        if (i + batch_size) % 320 == 0 or (i + batch_size) >= len(texts):
            print(f"   진행률: {min(i + batch_size, len(texts))}/{len(texts)} 완료")

    print("4. 추론 결과 저장 중...")
    df['sentence_label'] = predicted_labels
    df.to_csv(output_data_path, index=False, encoding='utf-8-sig')
    print(f"완료: 최종 라벨링 데이터가 저장되었습니다 -> {output_data_path}")

if __name__ == "__main__":
    run_batch_sentence_labeling()