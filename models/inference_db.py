import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def create_final_database():
    # 1. 경로 설정
    model_path = "./models/yogijogi_kcelectra_model"
    unlabeled_data_path = "data/raw_crawled_data.csv" # 제일 처음 크롤링했던 원본 전체 데이터
    labeled_data_path = "data/golden_dataset.xlsx"    # 조원들이 수작업한 데이터
    
    if not os.path.exists(model_path):
        print("학습된 모델을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    print("1. 모델 및 토크나이저 불러오는 중...")
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
    # 학습된 가중치 불러오기
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 디바이스 설정 (CPU 또는 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 모델을 평가(추론) 모드로 전환

    print("2. 미작업 데이터 추출 중...")
    df_raw = pd.read_csv(unlabeled_data_path, encoding='utf-8-sig')
    df_golden = pd.read_excel(labeled_data_path)
    
    # 원본에서 수작업한 데이터 제외 (차집합)
    df_unlabeled = df_raw[~df_raw['review_text'].isin(df_golden['review_text'])].copy()
    
    # 결측치 제거 및 텍스트 리스트화
    df_unlabeled = df_unlabeled.dropna(subset=['review_text'])
    texts = df_unlabeled['review_text'].tolist()
    
    print(f"-> AI가 분류할 남은 데이터 개수: {len(texts)}개")
    print("3. AI 모델 자동 라벨링 시작 (시간이 다소 소요될 수 있습니다)...")
    
    predicted_labels = []
    
    # 배치(Batch) 단위로 나누어 추론 (메모리 절약)
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 토큰화
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        # 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # 가장 높은 확률을 가진 클래스의 인덱스 추출
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predicted_labels.extend(preds)
            
        # 진행 상황 출력
        if (i + batch_size) % 320 == 0 or (i + batch_size) >= len(texts):
            print(f"진행도: {min(i + batch_size, len(texts))} / {len(texts)}")

    # 4. 예측된 라벨을 데이터프레임에 추가
    df_unlabeled['label'] = predicted_labels
    
    # 서비스에 필요한 필수 기둥만 필터링 (장소명 처리 통합)
    # 크롤링 원본의 열 이름에 맞게 title/place_name 등을 수정하세요.
    if '장소명' in df_unlabeled.columns:
        df_unlabeled = df_unlabeled.rename(columns={'장소명': 'title'})
    
    essential_columns = ['title', 'review_text', 'label']
    df_unlabeled = df_unlabeled[essential_columns]
    
    # 5. 수작업 황금 데이터(df_golden)와 AI 예측 데이터를 병합하여 최종 DB 완성
    df_golden = df_golden[essential_columns]
    df_final_db = pd.concat([df_golden, df_unlabeled], ignore_index=True)
    
    # 6. 최종 저장
    output_path = "data/final_travel_database.xlsx"
    df_final_db.to_excel(output_path, index=False)
    
    print("-" * 30)
    print(f"인간이 라벨링한 데이터: {len(df_golden)}개")
    print(f"AI가 라벨링한 데이터: {len(df_unlabeled)}개")
    print(f"최종 추천 시스템 DB: {len(df_final_db)}개")
    print(f"성공! 최종 데이터베이스가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    create_final_database()