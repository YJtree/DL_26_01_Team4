/* =========================================================
   YOGIJOGI JP main.js

   역할:
   1. Flask 서버에서 지역 데이터 불러오기
   2. 지역 카드 생성
   3. 지역 선택 후 검색 화면으로 전환
   4. 사용자 검색어를 Flask 서버로 보내기
   5. 실제 DB 기반 Top3 추천 결과 출력
========================================================= */


/* 서버에서 받아온 지역 데이터를 저장할 배열 */
let regions = [];

/* 현재 사용자가 선택한 지역 */
let selectedRegion = null;


/* 지역별 카드 색상
   이 부분은 추천 데이터가 아니라 화면 디자인용 색상임 */
const regionColors = {
  "가나자와": {
    accent: "#6e5a84",
    soft: "#e8e1ee"
  },
  "다카마쓰": {
    accent: "#4e7f73",
    soft: "#dcece7"
  },
  "기타큐슈": {
    accent: "#4f6f8f",
    soft: "#dfeaf1"
  },
  "오노미치": {
    accent: "#b07942",
    soft: "#f2e4d1"
  },
  "가루이자와": {
    accent: "#536f45",
    soft: "#e2ead9"
  },
  "미야코지마": {
    accent: "#3f7f89",
    soft: "#d9edf0"
  },
  "마쓰야마": {
    accent: "#9a6a45",
    soft: "#f0dfce"
  },
  "가고시마": {
    accent: "#96584d",
    soft: "#efddd8"
  }
};


/* HTML에 문자를 안전하게 출력하기 위한 함수 */
function escapeHTML(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


/* =========================================================
   1. 지역 데이터 불러오기
   /api/regions에서 region_descriptions.csv 기반 데이터를 받아옴
========================================================= */
async function loadRegions() {
  try {
    const response = await fetch("/api/regions");

    if (!response.ok) {
      throw new Error("지역 데이터를 불러오지 못했습니다.");
    }

    const data = await response.json();

    regions = data.regions;

    renderRegions();

  } catch (error) {
    console.error(error);

    document.getElementById("regionGrid").innerHTML = `
      <p>지역 데이터를 불러오는 중 문제가 발생했습니다.</p>
    `;
  }
}


/* =========================================================
   2. 첫 화면 지역 카드 생성
========================================================= */
function renderRegions() {
  const grid = document.getElementById("regionGrid");

  grid.innerHTML = regions.map((region, index) => {
    const color = regionColors[region.city] || {
      accent: "#355c46",
      soft: "#e2ead9"
    };

    return `
      <article
        class="region-card"
        style="--accent:${color.accent}; --accent-soft:${color.soft};"
        onclick="selectRegion('${escapeHTML(region.city)}')"
      >
        <div class="number">${String(index + 1).padStart(2, "0")}</div>

        <h3>${escapeHTML(region.city)}</h3>

        <p>${escapeHTML(region.region_description)}</p>

        <div class="tags">
          ${region.tags.map(tag => `
            <span class="tag">#${escapeHTML(tag)}</span>
          `).join("")}
        </div>
      </article>
    `;
  }).join("");
}


/* =========================================================
   3. 지역 선택 함수
   지역 카드를 클릭하면 검색 화면으로 이동
========================================================= */
function selectRegion(city) {
  selectedRegion = regions.find(region => region.city === city);

  if (!selectedRegion) {
    alert("선택한 지역 정보를 찾을 수 없습니다.");
    return;
  }

  document.getElementById("homePage").classList.remove("active");
  document.getElementById("searchPage").classList.add("active");

  document.getElementById("selectedRegionName").textContent =
    selectedRegion.city;

  document.getElementById("selectedRegionDesc").textContent =
    selectedRegion.region_description;

  document.getElementById("selectedRegionTags").innerHTML =
    selectedRegion.tags.map(tag => `
      <span class="tag">#${escapeHTML(tag)}</span>
    `).join("");

  document.getElementById("moodInput").value = "";
  document.getElementById("results").classList.remove("active");
  document.getElementById("resultList").innerHTML = "";

  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}


/* =========================================================
   4. 첫 화면으로 돌아가기
========================================================= */
function goHome() {
  document.getElementById("searchPage").classList.remove("active");
  document.getElementById("homePage").classList.add("active");

  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}


/* =========================================================
   5. 예시 검색어 클릭 시 검색창에 자동 입력
========================================================= */
function fillMood(text) {
  document.getElementById("moodInput").value = text;
}


/* =========================================================
   6. 실제 DB 기반 추천 결과 요청
   사용자가 검색어를 입력하고 버튼을 누르면
   /api/recommend로 city와 query를 전송
========================================================= */
async function showResults(event) {
  event.preventDefault();

  if (!selectedRegion) {
    alert("먼저 지역을 선택해주세요.");
    return;
  }

  const query = document.getElementById("moodInput").value.trim();

  if (!query) {
    alert("찾고 싶은 장소의 분위기를 입력해주세요.");
    return;
  }

  const resultList = document.getElementById("resultList");
  const results = document.getElementById("results");

  results.classList.add("active");

  resultList.innerHTML = `
    <article class="result-card">
      <div class="rank">...</div>
      <div>
        <h4>추천 결과를 불러오는 중입니다.</h4>
        <p>선택한 지역의 장소 DB를 기준으로 Top3를 계산하고 있습니다.</p>
      </div>
    </article>
  `;

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        city: selectedRegion.city,
        query: query
      })
    });

    if (!response.ok) {
      throw new Error("추천 결과를 불러오지 못했습니다.");
    }

    const data = await response.json();

    renderResults(data);

  } catch (error) {
    console.error(error);

    resultList.innerHTML = `
      <article class="result-card">
        <div class="rank">!</div>
        <div>
          <h4>추천 결과를 불러오지 못했습니다.</h4>
          <p>서버 실행 상태와 CSV 파일 위치를 확인해주세요.</p>
        </div>
      </article>
    `;
  }
}


/* =========================================================
   7. 추천 결과 화면 출력
========================================================= */
function renderResults(data) {
  const resultTitle = document.getElementById("resultTitle");
  const resultList = document.getElementById("resultList");

  resultTitle.textContent =
    `${data.city}에서 "${data.query}"에 어울리는 장소 Top 3`;

  if (!data.results || data.results.length === 0) {
    resultList.innerHTML = `
      <article class="result-card">
        <div class="rank">0</div>
        <div>
          <h4>추천 결과가 없습니다.</h4>
          <p>다른 분위기 검색어를 입력해보세요.</p>
        </div>
      </article>
    `;

    return;
  }

  resultList.innerHTML = data.results.map(place => {
    const scoreWidth = Math.min(place.score, 100);

    return `
      <article class="result-card">
        <div class="rank">${place.rank}</div>

        <div>
          <h4>${escapeHTML(place.place_name)}</h4>

          <p>${escapeHTML(place.place_description)}</p>

          <p>
            <strong>추천 이유:</strong>
            ${escapeHTML(place.recommend_reason)}
          </p>

          <div class="tags">
            <span class="tag">
              #${escapeHTML(place.top1_label)} ${place.top1_ratio.toFixed(1)}%
            </span>

            <span class="tag">
              #${escapeHTML(place.top2_label)} ${place.top2_ratio.toFixed(1)}%
            </span>
          </div>

          <div class="score-line">
            <div class="score-bar">
              <div class="score-fill" style="--score:${scoreWidth}%;"></div>
            </div>

            <span class="score-text">${place.score.toFixed(1)}점</span>
          </div>
        </div>
      </article>
    `;
  }).join("");
}


/* =========================================================
   페이지가 처음 열릴 때 실행
========================================================= */
loadRegions();