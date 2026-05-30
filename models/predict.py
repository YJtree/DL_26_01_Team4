import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_model():
    # 1. 저장된 모델과 토크나이저 불러오기
    model_path = "./models/yogijogi_kcelectra_model"
    print("학습된 모델을 불러오는 중입니다...")
    
    try:
        # 토크나이저도 파인튜닝이 완료된 로컬 경로에서 안전하게 불러옵니다.
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"모델 로드 실패. 학습이 완료되었는지 확인하세요: {e}")
        return

    # [핵심 수정] 모델을 '추론(Evaluation) 모드'로 전환하여 드롭아웃(Dropout) 노이즈를 차단합니다.
    model.eval()

    # 조원들과 최종 합의된 규격에 맞게 라벨명을 원상복구합니다.
    label_map = {
        0: "1번 (고즈넉/사색)",
        1: "2번 (레트로/빈티지)",
        2: "3번 (청량/애니메이션)",
        3: "4번 (아기자기/소박)"
    }

    # 2. 테스트해볼 가상의 새로운 리뷰들
    test_reviews = [
        "사람이 없는 조용한 숲길을 걸으며 복잡한 생각들을 정리할 수 있었던 마음이 편안해지는 고즈넉한 사찰이었습니다.",
        "관광객은 거의 없고 현지인들이 자주 찾는 낡은 선술집과 빛바랜 간판들에서 짙은 레트로 감성이 묻어나는 동네입니다.", 
        "해안도로를 따라 렌트카를 달리며 창문 밖으로 들어오는 시원한 바닷바람과 에메랄드빛 바다가 정말 청량하고 좋았어요.",
        "좁은 골목길에 숨어있는 귀여운 소품샵과 정성스러운 수제 케이크를 파는 작은 카페들이 너무 아기자기하고 예뻤습니다."
    ]

    print("\n--- 모델 예측 테스트 시작 ---\n")

    # 3. 각 리뷰를 모델에 통과시켜 예측값 확인
    for review in test_reviews:
        # 텍스트를 숫자로 변환 (배치 처리 규격에 맞게 패딩과 자르기 옵션 추가)
        inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # 가중치 업데이트가 필요 없으므로 no_grad()를 선언하여 메모리와 속도를 최적화합니다.
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Logits를 Softmax 함수에 통과시켜 0~100% 확률(확신도)로 변환합니다.
            probs = F.softmax(logits, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            
            pred_label = label_map[pred_idx.item()]
            confidence = max_prob.item() * 100
            
            print(f"입력 리뷰: {review}")
            print(f"예측 결과: {pred_label} (확신도: {confidence:.2f}%)\n" + "-"*50)

if __name__ == "__main__":
    test_model()