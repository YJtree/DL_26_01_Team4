import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os

def prepare_finetuning_dataset():
    # 1. 조원들이 완성한 최종 라벨링 데이터 불러오기
    # 실제 취합된 파일명으로 경로를 수정해야 합니다.
    input_path = "data/golden_dataset.xlsx" 
    
    if not os.path.exists(input_path):
        print("라벨링이 완료된 통합 엑셀 파일이 없습니다.")
        return

    df = pd.read_excel(input_path)

    # 2. 결측치 및 0번(기타/노이즈) 데이터 제거
    # 라벨이 비어있거나 0으로 표기된 데이터는 학습에 방해가 되므로 버립니다.
    df = df.dropna(subset=['label', 'review_text'])
    df = df[df['label'] != 0]

    # 3. 라벨 인덱스 변환 (1~5 -> 0~4)
    # 파이토치(PyTorch) 등 딥러닝 프레임워크는 라벨이 0부터 시작해야 에러가 나지 않습니다.
    df['label'] = df['label'].astype(int) - 1

    print(f"학습에 사용할 유효 데이터 개수: {len(df)}개")

    # 4. 학습용(Train)과 검증용(Validation) 데이터 8:2 분할
    # stratify=df['label']을 사용하여 5가지 감성 비율이 양쪽에 균등하게 들어가도록 설정합니다.
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['review_text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )

    # 5. KcELECTRA 토크나이저 불러오기
    # 터미널에서 pip install transformers 입력 필요
    print("KcELECTRA 토크나이저를 다운로드 및 로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

    # 6. 텍스트를 숫자로 변환 (Tokenization)
    # 최대 길이(max_length)를 128로 설정하여 너무 긴 네이버 블로그 글은 자르고, 짧은 구글 리뷰는 패딩(0)을 채웁니다.
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    print("데이터셋 분할 및 토큰화가 성공적으로 완료되었습니다.")
    print(f"Train Set: {len(train_labels)}개, Validation Set: {len(val_labels)}개")
    
    # 이 상태에서 곧바로 PyTorch Dataset 객체로 변환하여 학습 루프에 넣게 됩니다.
    return train_encodings, val_encodings, train_labels, val_labels

if __name__ == "__main__":
    prepare_finetuning_dataset()