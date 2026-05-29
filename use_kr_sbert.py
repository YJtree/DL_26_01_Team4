
# 역할 :
# - 사용자가 선택한 지역(city)과 입력한 분위기 문장(user_query)을 받음
# - data/place_descriptions.csv에서 해당 지역의 장소만 필터링
# - KR-SBERT 모델로 사용자 문장과 장소 설명문을 벡터화
# - 코사인 유사도를 계산해서 의미적으로 가장 가까운 장소 Top3를 추천
# ==============================================================================

import pandas as pd
import os
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 현재 place_recommend.py 파일이 있는 폴더 기준으로 data 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# make_place_descriptions.py가 생성한 장소 설명문 데이터셋
PLACE_FILE = os.path.join(DATA_DIR, "place_descriptions.csv")

# KR-SBERT 모델 로드 : Flask 실행 시 이 파일이 import되면서 모델도 함께 로드. 첫 실행 시 다운로드 필요
MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
model = SentenceTransformer(MODEL_NAME)

# 라벨명 정리 : 예: '2번 (레트로/빈티지)' → '레트로/빈티지'
def clean_label(label):
    label = str(label)

    match = re.search(r"\((.*?)\)", label)

    if match:
        return match.group(1)

    return label

# 4개 분위기 라벨 컬럼 정의
label_columns = {
    "label_1번 (고즈넉/사색)_ratio": "1번 (고즈넉/사색)",
    "label_2번 (레트로/빈티지)_ratio": "2번 (레트로/빈티지)",
    "label_4번 (청량/애니메이션)_ratio": "4번 (청량/바다)",
    "label_5번 (아기자기/소박)_ratio": "5번 (아기자기/소박)"
}


# KR-SBERT가 라벨 의미를 이해할 수 있도록 라벨별 설명문 작성
label_search_texts = {
    "label_1번 (고즈넉/사색)_ratio": "고즈넉하고 조용한 분위기, 차분한 산책, 사색하기 좋은 장소",
    "label_2번 (레트로/빈티지)_ratio": "레트로 감성, 빈티지한 거리, 전통적인 분위기, 오래된 공간",
    "label_4번 (청량/애니메이션)_ratio": "청량한 풍경, 맑은 하늘, 바다, 애니메이션 같은 밝은 분위기",
    "label_5번 (아기자기/소박)_ratio": "아기자기하고 귀여운 분위기, 소박한 감성, 작은 골목과 감성적인 장소"
}

# 4개 라벨과 비율을 큰 순서대로 정렬해서 반환하는 함수
def get_all_label_ratios(row):
    labels = []

    for column_name, label_name in label_columns.items():
        labels.append({
            "label": clean_label(label_name),
            "ratio": float(row[column_name])
        })

    labels.sort(
        key=lambda item: item["ratio"],
        reverse=True
    )

    return labels

# KR-SBERT가 비교할 장소별 문장을 만드는 함수
# 단순히 place_description만 넣어도 되지만, 
# 장소명, 지역명, top1/top2 라벨과 비율까지 함께 넣어서 추천 비교에 사용할 정보가 더 풍부해지도록.
def make_search_text(row):

    city = str(row["city"])
    place_name = str(row["place_name"])
    place_type = str(row["place"])
    place_description = str(row["place_description"])

    top1_label = clean_label(row["top1_label"])
    top2_label = clean_label(row["top2_label"])

    top1_ratio = float(row["top1_ratio"])
    top2_ratio = float(row["top2_ratio"])

    search_text = (
        f"{place_name}은/는 {city}에 있는 {place_type}입니다. "
        f"{place_description} "
        f"대표 분위기는 {top1_label}이며 적합도는 {top1_ratio:.1f}%입니다. "
        f"보조 분위기는 {top2_label}이며 적합도는 {top2_ratio:.1f}%입니다."
    )

    return search_text

# 라벨별 사용자 친화적 설명 문장
label_reason_phrases = {
    "고즈넉/사색": "조용하고 차분한 분위기가 두드러져, 천천히 머물거나 여유롭게 둘러보기 좋습니다.",
    "레트로/빈티지": "오래된 거리와 전통적인 감성이 느껴져, 로컬한 분위기를 즐기기 좋습니다.",
    "청량/바다": "맑고 시원한 풍경이 느껴져, 산책하거나 가볍게 쉬어가기 좋습니다.",
    "청량/애니메이션": "맑고 시원한 풍경이 느껴져, 산책하거나 가볍게 쉬어가기 좋습니다.",
    "아기자기/소박": "아기자기하고 소박한 분위기가 있어, 편안하고 감성적인 시간을 보내기 좋습니다."
}

# 추천 이유 문장을 만드는 함수
def make_recommend_reason(row, user_query):

    place_name = row["place_name"]
    place_type = row["place"]

    top1_label = clean_label(row["top1_label"])
    top2_label = clean_label(row["top2_label"])

    top1_reason = label_reason_phrases.get(
        top1_label,
        f"{top1_label} 분위기가 두드러지는 장소입니다."
    )

    top2_reason = label_reason_phrases.get(
        top2_label,
        f"{top2_label} 분위기도 함께 느껴지는 장소입니다."
    )

    reason = (
        f"'{place_name}'은/는 '{user_query}' 같은 분위기를 찾는 여행자에게 추천할 만한 {place_type}입니다.\n"
        f"{top1_reason}\n"
        f"또한 {top2_label} 분위기도 함께 나타나, 선택한 지역 안에서 비슷한 감성을 느끼고 싶은 여행자에게 잘 어울립니다."
    )

    return reason

# 선택한 지역 안에서 사용자 검색어와 의미적으로 가장 가까운 장소 Top K를 추천하는 함수. app.py에서 이 함수를 호출
def recommend_places(city, user_query, top_k=3):
    """

    Parameters:
        city: 사용자가 선택한 지역명
        user_query: 사용자가 입력한 분위기 검색어
        top_k: 추천할 장소 개수

    Returns:
        추천 장소 리스트
    """

    # 장소 설명문 데이터셋 읽기
    place_df = pd.read_csv(PLACE_FILE)

    # 선택한 지역의 장소만 필터링
    city_df = place_df[place_df["city"] == city].copy()

    # 해당 지역 데이터가 없으면 빈 리스트 반환
    if city_df.empty:
        return []

    # 현재 CSV에서 top1_ratio, top2_ratio는 이미 숫자형이고, 결측값이 없으므로 변환 작업 생략

    # 장소별 비교 문장 생성
    city_df["search_text"] = city_df.apply(make_search_text, axis=1)

    # 사용자 입력 문장을 KR-SBERT 벡터로 변환
    query_embedding = model.encode(
        [user_query],
        convert_to_numpy=True
    )

    # 사용자 입력 문장과 4개 분위기 라벨 설명문 사이의 KR-SBERT 유사도 계산
    label_text_list = [
        label_search_texts[column_name]
        for column_name in label_columns.keys()
    ]

    label_embeddings = model.encode(
        label_text_list,
        convert_to_numpy=True
    )

    label_similarities = cosine_similarity(
        query_embedding,
        label_embeddings
    )[0]

    query_label_scores = {}

    for index, column_name in enumerate(label_columns.keys()):
        query_label_scores[column_name] = max(float(label_similarities[index]), 0)

    # 선택한 지역의 장소 설명문들을 KR-SBERT 벡터로 변환
    place_embeddings = model.encode(
        city_df["search_text"].tolist(),
        convert_to_numpy=True
    )

    # 사용자 입력 문장과 각 장소 설명문 사이의 코사인 유사도 계산
    similarities = cosine_similarity(
        query_embedding,
        place_embeddings
    )[0]

    # 유사도 점수를 데이터프레임에 저장
    city_df["similarity"] = similarities

        # 사용자 입력과 가까운 라벨일수록, 해당 장소의 라벨 비율을 더 크게 반영
    def calculate_label_score(row):
        total_score = 0
        total_weight = 0

        for column_name in label_columns.keys():
            label_ratio = float(row[column_name]) / 100
            label_weight = query_label_scores[column_name]

            total_score += label_ratio * label_weight
            total_weight += label_weight

        if total_weight == 0:
            return 0

        return total_score / total_weight


    city_df["label_score"] = city_df.apply(calculate_label_score, axis=1)

    # 최종 추천 점수
    # similarity: 사용자 입력문과 장소 설명문의 KR-SBERT 유사도
    # label_score: 사용자 입력문과 4개 라벨 설명문의 KR-SBERT 유사도 + 장소별 라벨 비율
    city_df["final_similarity"] = (
        city_df["similarity"] * 0.7
        + city_df["label_score"] * 0.3
    )

    # 유사도 높은 순서대로 정렬
    # 유사도가 같을 경우 top1_ratio, top2_ratio가 높은 장소를 우선
    city_df = city_df.sort_values(
        by=["final_similarity", "similarity", "label_score"],
        ascending=False
    )

    # 상위 Top K개 선택
    top_df = city_df.head(top_k)

    results = []

    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        similarity_score = float(row["similarity"])

        # # 화면 표시용 점수
        # # KR-SBERT 코사인 유사도는 절대적인 정확도 점수가 아니므로, 0~100점으로 과장하지 않고 0~1 사이의 유사도 값으로 표시함        
        # display_score = round(max(similarity_score, 0), 3)
        
        results.append({
            "rank": rank,
            "city": row["city"],
            "place_name": row["place_name"],
            "place": row["place"],
            "top1_label": clean_label(row["top1_label"]),
            "top1_ratio": float(row["top1_ratio"]),
            "top2_label": clean_label(row["top2_label"]),
            "top2_ratio": float(row["top2_ratio"]),
            "labels": get_all_label_ratios(row),
            "recommend_reason": make_recommend_reason(row, user_query)
        })

    return results

# 단독 테스트용 코드 : 터미널에서 python use_kr_sbert.py로 실행하면 추천 결과를 테스트할 수 있음
if __name__ == "__main__":
    test_results = recommend_places(
        city="오노미치",
        user_query="조용하고 감성적인 골목길",
        top_k=3
    )

    for result in test_results:
        print(result["rank"], result["place_name"])
        print(result["recommend_reason"])
        print()