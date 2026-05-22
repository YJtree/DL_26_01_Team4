import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_model():
    # 1. 저장된 모델과 토크나이저 불러오기
    model_path = "./models/yogijogi_kcelectra_model"
    print("학습된 모델을 불러오는 중입니다...")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
    except Exception as e:
        print(f"모델 로드 실패. 학습이 완료되었는지 확인하세요: {e}")
        return

    # 라벨 번호와 실제 분위기 매핑 (모델은 0~4로 출력하므로 1~5에 맞게 변환)
    label_map = {
        0: "1번 (고즈넉/사색)",
        1: "2번 (레트로/빈티지)",
        2: "3번 (찐로컬/현지인)",
        3: "4번 (청량/애니메이션)",
        4: "5번 (아기자기/소박)"
    }

    # 2. 테스트해볼 가상의 새로운 리뷰들
    test_reviews = ["도시인은 기분좋은 산책하고 힐링하고 갑니다",
    "구 가루이자와 긴자거리 호수에서 긴자거리까지도 15분 정도?"
    ,"걸으면 도착하는 거리!"
    ,"원래 이대로 긴자거리로 이동해 어제 못 먹은 소바를 먹을 계획이었는데 호수 구경 다 하고 거리에 도착하니 아직 가게들이 열 시간이 아니었다."
    ]

    print("\n--- 모델 예측 테스트 시작 ---\n")

    # 3. 각 리뷰를 모델에 통과시켜 예측값 확인
    for review in test_reviews:
        # 텍스트를 숫자로 변환
        inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=128)
        
        # 모델 예측 수행 (기울기 계산 제외)
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 모델의 출력값을 확률(0~100%)로 변환
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)[0]
        
        # 가장 높은 확률을 가진 라벨 찾기
        predicted_index = torch.argmax(probabilities).item()
        predicted_label = label_map[predicted_index]
        confidence = probabilities[predicted_index].item() * 100

        # 결과 출력
        print(f"입력 리뷰: {review}")
        print(f"예측 결과: {predicted_label} (확신도: {confidence:.2f}%)")
        print("-" * 50)

if __name__ == "__main__":
    test_model()