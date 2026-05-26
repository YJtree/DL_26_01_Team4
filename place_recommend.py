import pandas as pd
import os
import re


# 현재 place_recommend.py 파일이 있는 폴더 기준으로 data 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PLACE_FILE = os.path.join(DATA_DIR, "place_descriptions.csv")


def clean_label(label):
    """
    '2번 (레트로/빈티지)' → '레트로/빈티지'
    형태로 라벨명을 정리하는 함수
    """
    label = str(label)

    match = re.search(r"\((.*?)\)", label)

    if match:
        return match.group(1)

    return label


def normalize_text(text):
    """
    검색 비교를 쉽게 하기 위해 텍스트를 정리하는 함수
    """
    text = str(text).lower()
    text = re.sub(r"[^가-힣a-zA-Z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_keywords(text):
    """
    사용자 검색어에서 비교에 사용할 단어를 추출하는 함수
    """
    text = normalize_text(text)

    words = text.split()

    # 너무 짧은 단어는 제외
    keywords = [word for word in words if len(word) >= 2]

    return keywords


# 사용자의 자연어 검색어를 분위기 라벨과 연결하기 위한 규칙
# 장소 데이터 자체는 CSV에서만 가져오고,
# 이 부분은 검색어 해석을 돕는 추천 규칙임
LABEL_KEYWORDS = {
    "고즈넉/사색": [
        "조용", "고즈넉", "차분", "사색", "한적", "평온", "여유", "산책"
    ],
    "레트로/빈티지": [
        "레트로", "빈티지", "전통", "오래된", "옛날", "거리", "감성", "역사"
    ],
    "아기자기/소박": [
        "아기자기", "소박", "작은", "귀여운", "카페", "골목", "로컬"
    ],
    "청량/애니메이션": [
        "청량", "애니메이션", "바다", "하늘", "푸른", "드라이브", "섬"
    ]
}


def calculate_score(row, user_query):
    """
    하나의 장소가 사용자 검색어와 얼마나 잘 맞는지 점수를 계산하는 함수

    점수 계산 기준:
    1. 사용자의 검색어가 장소 설명문에 포함되는지
    2. 검색어가 top1/top2 분위기 라벨과 연결되는지
    3. CSV에 저장된 top1_ratio, top2_ratio가 얼마나 높은지
    """

    query_norm = normalize_text(user_query)
    query_keywords = extract_keywords(user_query)

    place_name = str(row["place_name"])
    place_description = str(row["place_description"])

    top1_label = clean_label(row["top1_label"])
    top2_label = clean_label(row["top2_label"])

    top1_ratio = float(row["top1_ratio"])
    top2_ratio = float(row["top2_ratio"])

    searchable_text = normalize_text(
        f"{place_name} {place_description} {top1_label} {top2_label}"
    )

    score = 0.0

    # 1. 사용자 검색어 단어가 장소명/설명/라벨에 포함되는 경우 점수 부여
    for keyword in query_keywords:
        if keyword in searchable_text:
            score += 10

    # 2. top1 라벨과 사용자 검색어가 연결되는 경우
    top1_keywords = LABEL_KEYWORDS.get(top1_label, [])

    for keyword in top1_keywords:
        if keyword in query_norm:
            score += top1_ratio * 1.2

    # 3. top2 라벨과 사용자 검색어가 연결되는 경우
    top2_keywords = LABEL_KEYWORDS.get(top2_label, [])

    for keyword in top2_keywords:
        if keyword in query_norm:
            score += top2_ratio * 0.9

    # 4. 기본적으로 라벨 적합도가 높은 장소에 약간의 기본 점수 부여
    score += top1_ratio * 0.05
    score += top2_ratio * 0.03

    return score


def make_recommend_reason(row, user_query):
    """
    추천 이유 문장을 생성하는 함수
    실제 CSV의 라벨과 비율을 사용해서 설명함
    """

    place_name = row["place_name"]

    top1_label = clean_label(row["top1_label"])
    top2_label = clean_label(row["top2_label"])

    top1_ratio = float(row["top1_ratio"])
    top2_ratio = float(row["top2_ratio"])

    reason = (
        f"사용자가 입력한 '{user_query}'와 이 장소의 '{top1_label}' 분위기가 잘 연결됩니다. "
        f"DB 기준으로 {place_name}은/는 '{top1_label}' 적합도 {top1_ratio:.1f}%, "
        f"'{top2_label}' 적합도 {top2_ratio:.1f}%로 분류되어 있어 추천되었습니다."
    )

    return reason


def recommend_places(city, user_query, top_k=3):
    """
    선택한 지역 안에서 사용자 검색어와 가장 잘 맞는 장소 Top K를 추천하는 함수
    """

    place_df = pd.read_csv(PLACE_FILE)

    # 선택한 지역의 장소만 사용
    city_df = place_df[place_df["city"] == city].copy()

    if city_df.empty:
        return []

    # 각 장소별 추천 점수 계산
    city_df["raw_score"] = city_df.apply(
        lambda row: calculate_score(row, user_query),
        axis=1
    )

    # 점수가 높은 순서대로 정렬
    city_df = city_df.sort_values(
        by=["raw_score", "top1_ratio", "top2_ratio"],
        ascending=False
    )

    top_df = city_df.head(top_k)

    max_score = top_df["raw_score"].max()

    results = []

    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        if max_score > 0:
            display_score = round((row["raw_score"] / max_score) * 100, 1)
        else:
            display_score = 0

        results.append({
            "rank": rank,
            "city": row["city"],
            "place_name": row["place_name"],
            "place_description": row["place_description"],
            "top1_label": clean_label(row["top1_label"]),
            "top1_ratio": float(row["top1_ratio"]),
            "top2_label": clean_label(row["top2_label"]),
            "top2_ratio": float(row["top2_ratio"]),
            "score": display_score,
            "recommend_reason": make_recommend_reason(row, user_query)
        })

    return results


# 단독 테스트용 코드
if __name__ == "__main__":
    test_results = recommend_places(
        city="오노미치",
        user_query="조용하고 감성적인 골목길",
        top_k=3
    )

    for result in test_results:
        print(result["rank"], result["place_name"], result["score"])
        print(result["recommend_reason"])
        print()