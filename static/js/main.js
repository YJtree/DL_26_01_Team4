/* =========================================================
    역할:
   1. Flask 서버에서 지역 데이터 불러오기
   2. 지역 카드 생성
   3. 지역 선택 후 검색 화면으로 전환
   4. 사용자 검색어를 Flask 서버로 보내기
   5. 실제 DB 기반 Top3 추천 결과 출력

   중요:
   - 이 파일에서는 디자인을 관리하지 않음
========================================================= */


/* 서버에서 받아온 지역 데이터를 저장할 배열 */
let regions = [];

/* 현재 사용자가 선택한 지역 */
let selectedRegion = null;

/* 지역별 대표 이미지 경로 */
const regionImages = {
  "가나자와": "https://res.cloudinary.com/jnto/image/upload/w_750,h_503,c_fill,fl_lossy,f_auto/v1514197289/ishikawa/Ishikawa007_7",
  "다카마쓰": "https://res.cloudinary.com/jnto/image/upload/w_750,h_503,c_fill,fl_lossy,f_auto/v1516729890/kagawa/Kagawa604_1",
  "기타큐슈": "https://res.cloudinary.com/jnto/image/upload/w_750,h_503,c_fill,fl_lossy,f_auto/v1514399115/fukuoka/Fukuoka1622_1",
  "오노미치": "https://res.cloudinary.com/jnto/image/upload/w_750,h_503,c_fill,fl_lossy,f_auto/v1644199844/hiroshima/20200922_shimanami_kaido_03.jpg",
  "가루이자와": "https://res.cloudinary.com/jnto/image/upload/w_750,h_503,c_fill,fl_lossy,f_auto/v1514193491/nagano/Nagano057_4",
  "미야코지마": "https://res.cloudinary.com/jnto/image/upload/w_750,h_503,c_fill,fl_lossy,f_auto/v1516728547/okinawa/Okinawa1980_8",
  "마쓰야마": "https://res.cloudinary.com/jnto/image/upload/w_750,h_503,c_fill,fl_lossy,f_auto/v1514401278/ehime/Ehime625_1",
  "가고시마": "https://res.cloudinary.com/jnto/image/upload/w_750,h_503,c_fill,fl_lossy,f_auto/v1644457629/kagoshima/20201027_sakurajima_13.jpg"
};


/* =========================================================
   HTML에 문자를 안전하게 출력하기 위한 함수

   DB에서 가져온 텍스트를 HTML에 넣을 때,
   <, >, &, 따옴표 같은 문자가 HTML 태그로 해석되지 않게 변환함
========================================================= */
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

   Flask의 /api/regions 주소로 요청을 보냄.
   app.py에서 region_descriptions.csv와 place_descriptions.csv를 읽어서
   지역명, 지역 설명문, 분위기 태그를 JSON으로 보내줌.
========================================================= */
async function loadRegions() {
  const grid = document.getElementById("regionGrid");

  try {
    grid.innerHTML = `
      <div class="loading-card">
        지역 데이터를 불러오는 중입니다.
      </div>
    `;

    const response = await fetch("/api/regions");

    if (!response.ok) {
      throw new Error("지역 데이터를 불러오지 못했습니다.");
    }

    const data = await response.json();

    regions = data.regions || [];

    renderRegions();

  } catch (error) {
    console.error(error);

    grid.innerHTML = `
      <div class="loading-card error">
        지역 데이터를 불러오는 중 문제가 발생했습니다.<br>
        Flask 서버 실행 상태와 CSV 파일 위치를 확인해주세요.
      </div>
    `;
  }
}


/* =========================================================
   2. 첫 화면 지역 카드 생성

   이전 버전에서는 여기에서 regionColors를 사용했지만,
   이제는 색상을 main.js에서 관리하지 않음.

   대신 data-city="가나자와" 같은 속성만 넣고,
   style.css에서 이 data-city 값을 보고 색상을 적용함.
========================================================= */
function renderRegions() {
  const grid = document.getElementById("regionGrid");

  if (!regions || regions.length === 0) {
    grid.innerHTML = `
      <div class="loading-card error">
        표시할 지역 데이터가 없습니다.
      </div>
    `;
    return;
  }

  grid.innerHTML = regions.map((region, index) => {
    const city = region.city;
    const description = region.region_description;
    const tags = region.tags || [];
    const imageSrc = regionImages[city] || "/static/images/default.jpg";

    return `
      <article
        class="region-card"
        data-city="${escapeHTML(city)}"
        role="button"
        tabindex="0"
      >
        <div class="region-image">
          <img src="${escapeHTML(imageSrc)}" alt="${escapeHTML(city)} 이미지">
        </div>

        <div class="region-text">
          <div class="number">
            ${String(index + 1).padStart(2, "0")}
          </div>

          <h3>${escapeHTML(city)}</h3>

          <p>${escapeHTML(description)}</p>

          <div class="tags">
            ${tags.map(tag => `
              <span class="tag">#${escapeHTML(tag)}</span>
            `).join("")}
          </div>
        </div>
      </article>
    `;
  }).join("");

  connectRegionCardEvents();
}


/* =========================================================
   3. 지역 카드 클릭 이벤트 연결

   HTML 안에 onclick을 직접 넣는 대신,
   JS에서 카드들을 찾아 클릭 이벤트를 연결함.

   이렇게 하면 HTML 구조와 JS 동작이 더 분리되고,
   카드 색상도 CSS에서만 관리할 수 있음.
========================================================= */
function connectRegionCardEvents() {
  const cards = document.querySelectorAll(".region-card");

  cards.forEach(card => {
    card.addEventListener("click", () => {
      const city = card.dataset.city;
      selectRegion(city);
    });

    card.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        const city = card.dataset.city;
        selectRegion(city);
      }
    });
  });
}


/* =========================================================
   4. 지역 선택 함수

   사용자가 지역 카드를 클릭하면 실행됨.
   첫 화면을 숨기고 검색 화면을 보여줌.
========================================================= */
function selectRegion(city) {
  selectedRegion = regions.find(region => region.city === city);

  if (!selectedRegion) {
    alert("선택한 지역 정보를 찾을 수 없습니다.");
    return;
  }

  const selectedPanel = document.querySelector(".selected-panel");
  
  if (selectedPanel) {
    selectedPanel.dataset.city = selectedRegion.city;
    
    const imageSrc = regionImages[selectedRegion.city] || "/static/images/default.jpg";
    selectedPanel.style.setProperty("--region-image", `url("${imageSrc}")`);
  }

  document.getElementById("homePage").classList.remove("active");
  document.getElementById("searchPage").classList.add("active");

  document.getElementById("selectedRegionName").textContent =
    selectedRegion.city;

  document.getElementById("selectedRegionDesc").textContent =
    selectedRegion.region_description;

  document.getElementById("selectedRegionTags").innerHTML =
    (selectedRegion.tags || []).map(tag => `
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
   5. 첫 화면으로 돌아가기

   검색 화면에서 지역 선택 화면으로 돌아감.
========================================================= */
function goHome() {
  document.getElementById("searchPage").classList.remove("active");
  document.getElementById("homePage").classList.add("active");

  selectedRegion = null;

  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}


/* =========================================================
   6. 예시 검색어 입력 함수

   예시 버튼을 누르면 검색창에 문구가 자동 입력됨.
========================================================= */
function fillMood(text) {
  const input = document.getElementById("moodInput");

  input.value = text;
  input.focus();
}


/* =========================================================
   7. 실제 DB 기반 추천 결과 요청

   사용자가 검색어를 입력하고 버튼을 누르면
   /api/recommend로 city와 query를 전송함.

   app.py에서는 이 요청을 받아 user_kr_sbert.py를 실행하고,
   place_descriptions.csv 기반 Top3 추천 결과를 반환함.
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
          <p>서버 실행 상태, CSV 파일 위치, place_recommend.py 코드를 확인해주세요.</p>
        </div>
      </article>
    `;
  }
}


/* =========================================================
   8. 숫자 안전 변환 함수

   CSV에서 읽은 값이 문자열일 수도 있으므로
   화면 출력 전에 숫자로 안전하게 변환함.
========================================================= */
function toNumber(value, defaultValue = 0) {
  const number = Number(value);

  if (Number.isNaN(number)) {
    return defaultValue;
  }

  return number;
}


/* =========================================================
   9. 추천 결과 화면 출력

   /api/recommend에서 받은 실제 DB 기반 추천 결과를
   카드 형태로 화면에 출력함.
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
    const labels = (place.labels || []).map(label => {
      return {
        label: label.label,
        ratio: toNumber(label.ratio)
      };
    }).sort((a, b) => b.ratio - a.ratio);

    return `
      <article class="result-card">
        <div class="rank">${escapeHTML(place.rank)}</div>

        <div class="result-content">
          <h4>${escapeHTML(place.place_name)}</h4>

          <p class="recommend-reason">
            <strong>추천 이유:</strong>
            ${escapeHTML(place.recommend_reason)}
          </p>

          <div class="tags">
            ${labels.map(label => `
              <span class="tag">
                #${escapeHTML(label.label)} ${label.ratio.toFixed(1)}%
              </span>
            `).join("")}
          </div>
        </div>
      </article>
    `;
  }).join("");
}


/* =========================================================
   10. 페이지가 처음 열릴 때 실행
========================================================= */
document.addEventListener("DOMContentLoaded", () => {
  loadRegions();
});


/* =========================================================
   11. HTML의 onclick에서 사용할 함수들을 전역으로 등록

   index.html 안의 버튼들이 onclick="goHome()",
   onclick="fillMood(...)", onsubmit="showResults(event)" 형태로 되어 있기 때문에
   window 객체에 등록해줌.
========================================================= */
window.goHome = goHome;
window.fillMood = fillMood;
window.showResults = showResults;