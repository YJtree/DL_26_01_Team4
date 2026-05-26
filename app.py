from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import re

from place_recommend import recommend_places

app = Flask(__name__)

# 현재 app.py 파일이 있는 폴더 기준으로 data 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

REGION_FILE = os.path.join(DATA_DIR, "region_descriptions.csv")
PLACE_FILE = os.path.join(DATA_DIR, "place_descriptions.csv")


def clean_label(label):
    """
    '2번 (레트로/빈티지)' 형태의 라벨에서
    괄호 안의 '레트로/빈티지'만 추출하는 함수
    """
    label = str(label)

    match = re.search(r"\((.*?)\)", label)

    if match:
        return match.group(1)

    return label


def load_region_data():
    """
    region_descriptions.csv와 place_descriptions.csv를 읽어서
    첫 화면에 보여줄 지역 데이터를 생성하는 함수

    지역명, 지역 설명문은 region_descriptions.csv에서 가져오고,
    지역별 분위기 태그는 place_descriptions.csv의 top1/top2 라벨을 기준으로 자동 생성함.
    """

    region_df = pd.read_csv(REGION_FILE)
    place_df = pd.read_csv(PLACE_FILE)

    regions = []

    for _, region_row in region_df.iterrows():
        city = region_row["city"]

        # 해당 지역의 장소 데이터만 필터링
        city_places = place_df[place_df["city"] == city]

        label_scores = {}

        # 해당 지역에서 많이 등장하는 분위기 라벨을 태그로 사용
        for _, place_row in city_places.iterrows():
            top1_label = clean_label(place_row["top1_label"])
            top2_label = clean_label(place_row["top2_label"])

            top1_ratio = float(place_row["top1_ratio"])
            top2_ratio = float(place_row["top2_ratio"])

            label_scores[top1_label] = label_scores.get(top1_label, 0) + top1_ratio
            label_scores[top2_label] = label_scores.get(top2_label, 0) + top2_ratio * 0.7

        # 점수가 높은 라벨 3개만 태그로 사용
        tags = sorted(
            label_scores,
            key=label_scores.get,
            reverse=True
        )[:3]

        regions.append({
            "city": city,
            "region_description": region_row["region_description"],
            "tags": tags
        })

    return regions


@app.route("/")
def index():
    """
    메인 페이지를 보여주는 라우트
    """
    return render_template("index.html")


@app.route("/api/regions")
def api_regions():
    """
    프론트엔드에서 첫 화면 지역 카드 데이터를 요청할 때 사용
    """
    regions = load_region_data()

    return jsonify({
        "regions": regions
    })


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    프론트엔드에서 사용자가 선택한 지역과 검색어를 보내면
    해당 지역 안에서 Top3 장소를 추천해서 반환
    """
    data = request.get_json()

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