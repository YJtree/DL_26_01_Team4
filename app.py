from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import re

from use_kr_sbert import recommend_places

app = Flask(__name__)

# 현재 app.py 파일이 있는 폴더 기준으로 data 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

REGION_FILE = os.path.join(DATA_DIR, "region_descriptions.csv")
PLACE_FILE = os.path.join(DATA_DIR, "place_descriptions.csv")

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

# 지역별 분위기 태그는 place_descriptions.csv의 top1/top2 라벨을 기준으로 자동 생성
# 지역 데이터 생성 : 전체 평균보다 해당 지역에서 더 두드러지는 라벨을 지역 태그로 사용
def load_region_data():

    region_df = pd.read_csv(REGION_FILE)
    place_df = pd.read_csv(PLACE_FILE)

    # 4개 라벨 비율 컬럼을 숫자형으로 변환
    for column in label_columns.keys():
        place_df[column] = pd.to_numeric(
            place_df[column],
            errors="coerce"
        ).fillna(0)

    # 전체 장소 기준 라벨 평균
    overall_means = place_df[list(label_columns.keys())].mean()

    regions = []

    for _, region_row in region_df.iterrows():
        city = region_row["city"]

        # 해당 지역의 장소 데이터만 필터링
        city_places = place_df[place_df["city"] == city]

        if city_places.empty:
            tags = []
        else:
            # 해당 지역의 라벨 평균
            city_means = city_places[list(label_columns.keys())].mean()

            distinct_scores = {}

            for column_name, label_name in label_columns.items():
                city_average = city_means[column_name]
                overall_average = overall_means[column_name]

                # 전체 평균보다 얼마나 더 높은지 계산
                distinct_score = city_average - overall_average

                distinct_scores[clean_label(label_name)] = distinct_score

            # 전체 평균보다 높은 라벨만 우선 사용
            positive_tags = [
                label
                for label, score in sorted(
                    distinct_scores.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
                if score > 0
            ]

            # 너무 적게 나오면, 그래도 차이가 큰 순서대로 보충
            if len(positive_tags) < 2:
                tags = [
                    label
                    for label, score in sorted(
                        distinct_scores.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )
                ][:2]
            else:
                tags = positive_tags[:3]

        regions.append({
            "city": city,
            "region_description": region_row["region_description"],
            "tags": tags
        })

    return regions


@app.route("/")
def index():

    # 메인 페이지 보여주기
    return render_template("index.html")


@app.route("/api/regions")
def api_regions():

    # 프론트엔드에서 첫 화면 지역 카드 데이터를 요청할 때 사용
    regions = load_region_data()

    return jsonify({
        "regions": regions
    })


@app.route("/api/recommend", methods=["POST"])
def api_recommend():

    # 사용자가 선택한 지역과 검색어를 보내면 해당 지역 안에서 Top3 장소를 추천해서 반환
    data = request.get_json() or {}

    city = data.get("city")
    query = data.get("query")

    if not city or not query:
        return jsonify({
            "error": "city와 query가 필요합니다."
        }), 400

    results = recommend_places(
        city=city,
        user_query=query,
        top_k=3
    )

    return jsonify({
        "city": city,
        "query": query,
        "results": results
    })


if __name__ == "__main__":
    app.run(debug=True)