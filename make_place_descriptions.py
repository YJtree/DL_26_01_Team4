
# 역할 : 장소 설명문을 자동으로 만드는 파일. 그래서 place_decripions.csv파일을 생성함
# KR-SBERT 모델에서 벡터화하는데 사용하는 설명문(csv 파일)을 만드는 코드
# ==============================================================================

import pandas as pd

# 장소별 라벨링 결과 CSV 불러오기
df = pd.read_csv("data/final_travel_database.csv")

# 라벨 컬럼명과 실제 라벨명을 연결
label_columns = {
    "label_1번 (고즈넉/사색)_ratio": "1번 (고즈넉/사색)",
    "label_2번 (레트로/빈티지)_ratio": "2번 (레트로/빈티지)",
    "label_4번 (청량/애니메이션)_ratio": "4번 (청량/바다)",
    "label_5번 (아기자기/소박)_ratio": "5번 (아기자기/소박)"
}

# 라벨명을 자연스러운 분위기 문장으로 변환하기 위한 딕셔너리
# top_label 값이 그대로 들어가면 문장이 어색할 수 있으므로, 사람이 읽기 쉬운 설명 문구로 바꾸기
label_phrases = {
    "1번 (고즈넉/사색)": "조용하고 차분한 분위기",
    "2번 (레트로/빈티지)": "전통적이고 레트로한 빈티지 감성",
    "4번 (청량/바다)": "맑고 청량한 풍경과 애니메이션 장면처럼 밝고 생동감 있는 분위기",
    "5번 (아기자기/소박)": "아기자기하고 소박한 감성"
}

# 라벨별 추천 대상 문장 만들기 : 각 라벨이 어떤 사용자의 취향과 연결되는지 설명하기 위한 문장
target_phrases = {
    "1번 (고즈넉/사색)": "조용히 산책하며 차분한 분위기를 느끼고 싶은 사용자",
    "2번 (레트로/빈티지)": "전통적인 거리와 오래된 감성을 좋아하는 사용자",
    "4번 (청량/바다)": "푸른 바다, 맑은 하늘, 시원하고 청량한 풍경을 좋아하는 사용자",
    "5번 (아기자기/소박)": "아기자기하고 소박한 분위기를 좋아하는 사용자"
}

# top1/top2 라벨 추출 : 4개 라벨 비율 중 높은 순서대로 2개 선택
def get_top2_labels(row):
    label_scores = []

    for column_name, label_name in label_columns.items():
        ratio = float(row[column_name])
        label_scores.append((label_name, ratio))

    label_scores.sort(key=lambda x: x[1], reverse=True)

    top1_label, top1_ratio = label_scores[0]
    top2_label, top2_ratio = label_scores[1]

    return top1_label, top1_ratio, top2_label, top2_ratio

# 장소 설명문 생성 함수 : CSV 한 행(row)을 받아서 장소 설명문 하나를 생성
def make_place_description(row):
    city = row["city"]                  # 도시명
    place_name = row["place_name"]      # 장소명
    place_type = row["place"]

    # top1, top2 라벨 가져오기
    top1_label = row["top1_label"]
    top2_label = row["top2_label"]

    # top1, top2 비율 가져오기
    top1_ratio = row["top1_ratio"]
    top2_ratio = row["top2_ratio"]

    # 라벨명을 자연스러운 분위기 문장으로 변환 (딕셔너리.get(찾을키, 없을때 대신 쓸 값))
    top1_text = label_phrases.get(top1_label, top1_label)
    top2_text = label_phrases.get(top2_label, top2_label)

    # top1 라벨을 기준으로 추천 대상 문장 생성
    target_text = target_phrases.get(top1_label, "취향에 맞는 장소를 찾는 사용자")

    # 최종 장소 설명문 생성
    description = (
        f"{place_name}은/는 {city}에 위치한 {place_type}로, "
        f"{top1_text}이 가장 두드러지며 약 {top1_ratio}%의 적합도를 보인다. "
        f"또한 {top2_text}도 약 {top2_ratio:.1f}%로 함께 나타나는 장소이다. "
        f"{target_text}에게 적합하다."
    )

    return description

# 라벨 비율 컬럼을 숫자형으로 변환
for column in label_columns.keys():
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

# 각 장소별 top1/top2 라벨과 비율 생성
df[["top1_label", "top1_ratio", "top2_label", "top2_ratio"]] = df.apply(
    lambda row: pd.Series(get_top2_labels(row)),
    axis=1
)

# 모든 장소에 대해 설명문 생성
df["place_description"] = df.apply(make_place_description, axis=1)

# 결과 저장
df.to_csv("data/place_descriptions.csv", index=False, encoding="utf-8-sig")

print("place_descriptions.csv 파일 생성 완료")