// VoiceGiraffe Benchmark Viewer — app.js

/* ─── State ─── */
const state = {
  page: 0,
  pageSize: 20,
  total: 0,
  filters: { qa_type: "", qa_level: "", language_type: "", run: "", only_wrong: false },
  charts: {},
  summary: null,
};

const quiz = {
  queue: [],
  index: 0,
  selected: null,
  answered: false,
  stats: { total: 0, correct: 0 },
};

/* ═══════════════════════════════════════════════
   1. Utility
   ═══════════════════════════════════════════════ */

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { const j = await r.json(); msg = j.error || j.detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

function pct(v) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return (v * 100).toFixed(1) + "%";
}

function fillSelect(sel, values, current) {
  const first = sel.querySelector("option");
  sel.innerHTML = "";
  if (first) sel.appendChild(first);
  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (current && v === current) opt.selected = true;
    sel.appendChild(opt);
  });
}

/* ═══════════════════════════════════════════════
   2. Tab Routing
   ═══════════════════════════════════════════════ */

function initTabs() {
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll("[id^='panel-']").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      const panel = document.getElementById("panel-" + tab.dataset.tab);
      if (panel) panel.classList.add("active");
    });
  });
}

/* ═══════════════════════════════════════════════
   3. Dashboard / Charts
   ═══════════════════════════════════════════════ */

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    delete state.charts[key];
  }
}

function makeBarChart(canvasId, title, labels, values) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  state.charts[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: title,
        data: values,
        backgroundColor: "rgba(245, 166, 35, .7)",
        borderColor: "rgba(245, 166, 35, 1)",
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 1, ticks: { color: "#94a3b8", callback: v => (v * 100).toFixed(0) + "%" }, grid: { color: "rgba(148,163,184,.15)" } },
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
      },
      plugins: { legend: { labels: { color: "#e2e8f0" } } },
    },
  });
}

function makeRadar(canvasId, labels, values) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;
  state.charts[canvasId] = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [{
        label: "accuracy",
        data: values,
        backgroundColor: "rgba(245, 166, 35, .25)",
        borderColor: "rgba(245, 166, 35, 1)",
        pointBackgroundColor: "rgba(139, 94, 60, .7)",
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { display: false, stepSize: 0.2 },
          grid: { color: "rgba(148,163,184,.2)" },
          angleLines: { color: "rgba(148,163,184,.2)" },
          pointLabels: { color: "#e2e8f0" },
        },
      },
      plugins: { legend: { labels: { color: "#e2e8f0" } } },
    },
  });
}

function renderRunCharts(run) {
  if (!run) {
    document.getElementById("statAcc").textContent = "—";
    ["chartType", "chartLevel", "chartLang", "chartRadar"].forEach(destroyChart);
    return;
  }
  document.getElementById("statAcc").textContent = pct(run.overall_accuracy);

  const groups = run.groups || {};
  function pairs(g) {
    const ks = Object.keys(g || {}).sort();
    return [ks, ks.map(k => g[k].accuracy)];
  }
  const [tk, tv] = pairs(groups.qa_type);
  const [lk, lv] = pairs(groups.qa_level);
  const [gk, gv] = pairs(groups.language_type);
  makeBarChart("chartType", "QA Type", tk, tv);
  makeBarChart("chartLevel", "QA Level", lk, lv);
  makeBarChart("chartLang", "Language", gk, gv);

  const cross = groups.qa_type_x_language || {};
  const ckeys = Object.keys(cross).sort();
  makeRadar("chartRadar", ckeys, ckeys.map(k => cross[k].accuracy));
}

function handleRunChange(e) {
  state.filters.run = e.target.value;
  state.page = 0;
  const r = (state.summary?.runs || []).find(x => x.name === state.filters.run) || null;
  renderRunCharts(r);
  loadSamples();
}

async function loadSummary() {
  try {
    const data = await fetchJSON("/api/summary");
    state.summary = data;

    document.getElementById("statTotal").textContent = data.total_samples;
    document.getElementById("statAudios").textContent = data.unique_audios;
    document.getElementById("statLocal").textContent = `${data.audios_present_locally} / ${data.unique_audios}`;

    const qaTypes = Object.keys(data.qa_type || {}).sort();
    const qaLevels = Object.keys(data.qa_level || {}).sort();
    const langTypes = Object.keys(data.language_type || {}).sort();

    // Dashboard filters
    fillSelect(document.getElementById("fType"), qaTypes, state.filters.qa_type);
    fillSelect(document.getElementById("fLevel"), qaLevels, state.filters.qa_level);
    fillSelect(document.getElementById("fLang"), langTypes, state.filters.language_type);

    // Quiz filters
    const quizType = document.getElementById("quizType");
    const quizLevel = document.getElementById("quizLevel");
    const quizLang = document.getElementById("quizLang");
    if (quizType) fillSelect(quizType, qaTypes, "");
    if (quizLevel) fillSelect(quizLevel, qaLevels, "");
    if (quizLang) fillSelect(quizLang, langTypes, "");

    // Run selector
    const runSel = document.getElementById("runSelect");
    runSel.innerHTML = '<option value="">— none —</option>';
    (data.runs || []).forEach(r => {
      const opt = document.createElement("option");
      opt.value = r.name;
      const accStr = r.overall_accuracy != null ? ` (${pct(r.overall_accuracy)})` : "";
      opt.textContent = r.name + accStr;
      runSel.appendChild(opt);
    });
    if (state.filters.run) runSel.value = state.filters.run;

    // Samples run filter
    const fRun = document.getElementById("fRun");
    if (fRun) fillSelect(fRun, (data.runs || []).map(r => r.name), state.filters.run);

    const currentRun = (data.runs || []).find(r => r.name === state.filters.run) || null;
    renderRunCharts(currentRun);
  } catch (e) {
    console.error("loadSummary failed:", e);
  }
}

/* ═══════════════════════════════════════════════
   4. Model Management
   ═══════════════════════════════════════════════ */

async function loadModels() {
  try {
    const data = await fetchJSON("/api/models");
    const sel = document.getElementById("modelSelect");
    if (!sel) return;
    sel.innerHTML = '<option value="">— select model —</option>';
    (data.available || []).forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    });
    // Reflect loaded state
    if (data.loaded) {
      setModelStatus("loaded", data.loaded + " loaded");
    }
  } catch (e) {
    console.warn("loadModels:", e.message);
  }
}

function setModelStatus(status, text) {
  const dot = document.getElementById("modelDot");
  const lbl = document.getElementById("modelStatus");
  if (!dot || !lbl) return;
  dot.classList.remove("loaded", "loading");
  if (status === "loaded") dot.classList.add("loaded");
  if (status === "loading") dot.classList.add("loading");
  lbl.textContent = text || "";
}

async function handleLoadModel() {
  const sel = document.getElementById("modelSelect");
  const name = sel ? sel.value : "";
  if (!name) { alert("Please select a model first."); return; }

  setModelStatus("loading", "Loading...");
  const btn = document.getElementById("btnLoadModel");
  if (btn) btn.disabled = true;

  try {
    const res = await fetchJSON("/api/models/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: name }),
    });
    setModelStatus("loaded", (res.model || name) + " loaded");
  } catch (e) {
    setModelStatus("", "No model loaded");
    alert("Failed to load model: " + e.message);
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* ═══════════════════════════════════════════════
   5. Quiz Module
   ═══════════════════════════════════════════════ */

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

async function startQuiz() {
  const qaType = document.getElementById("quizType")?.value || "";
  const qaLevel = document.getElementById("quizLevel")?.value || "";
  const lang = document.getElementById("quizLang")?.value || "";

  const params = new URLSearchParams();
  params.set("limit", "200");
  if (qaType) params.set("qa_type", qaType);
  if (qaLevel) params.set("qa_level", qaLevel);
  if (lang) params.set("language_type", lang);

  try {
    const data = await fetchJSON("/api/samples?" + params.toString());
    if (!data.rows || data.rows.length === 0) {
      alert("No samples match the selected filters.");
      return;
    }
    quiz.queue = shuffleArray(data.rows);
    quiz.index = 0;
    quiz.selected = null;
    quiz.answered = false;
    quiz.stats = { total: 0, correct: 0 };

    document.getElementById("quizStats").style.display = "";
    document.getElementById("quizQuestion").style.display = "";
    document.getElementById("quizEmpty").style.display = "none";
    updateQuizStats();
    showQuestion();
  } catch (e) {
    alert("Failed to load quiz samples: " + e.message);
  }
}

function showQuestion() {
  const s = quiz.queue[quiz.index];
  quiz.selected = null;
  quiz.answered = false;

  // Meta tags
  const qTag = document.getElementById("qTag");
  const qLevel = document.getElementById("qLevel");
  const qLang = document.getElementById("qLang");
  const qCounter = document.getElementById("qCounter");
  if (qTag) qTag.textContent = s.qa_type;
  if (qLevel) qLevel.textContent = s.qa_level;
  if (qLang) qLang.textContent = s.language_type;
  if (qCounter) qCounter.textContent = `${quiz.index + 1} / ${quiz.queue.length}`;

  // Audio
  const audio = document.getElementById("quizAudio");
  if (audio) {
    if (s.audio_local_available) {
      audio.src = `/audio/${s.youtube_id}`;
      audio.style.display = "";
    } else {
      audio.removeAttribute("src");
      audio.style.display = "none";
    }
  }

  // Stem
  const stem = document.getElementById("qStem");
  if (stem) stem.textContent = s.question_stem || s.question_full || "(no question)";

  // Options
  const optContainer = document.getElementById("qOptions");
  optContainer.innerHTML = "";
  ["A", "B", "C", "D"].forEach(letter => {
    const card = document.createElement("div");
    card.className = "option-card";
    card.dataset.letter = letter;
    card.innerHTML = `<span class="option-letter">${letter}</span><span class="option-text">${s.options[letter] || ""}</span>`;
    card.addEventListener("click", () => selectOption(letter));
    optContainer.appendChild(card);
  });

  // Buttons
  const btnSubmit = document.getElementById("btnSubmit");
  btnSubmit.style.display = "";
  btnSubmit.disabled = true;
  document.getElementById("btnModelInfer").style.display = "";
  document.getElementById("btnNext").style.display = "none";

  // Result area
  const qResult = document.getElementById("qResult");
  if (qResult) qResult.style.display = "none";
  const resultModel = document.getElementById("resultModel");
  if (resultModel) { resultModel.innerHTML = ""; resultModel.style.display = "none"; }
}

function selectOption(letter) {
  if (quiz.answered) return;
  quiz.selected = letter;
  document.querySelectorAll("#qOptions .option-card").forEach(el => {
    el.classList.toggle("selected", el.dataset.letter === letter);
  });
  document.getElementById("btnSubmit").disabled = false;
}

async function submitAnswer() {
  if (!quiz.selected || quiz.answered) return;
  quiz.answered = true;
  const s = quiz.queue[quiz.index];

  try {
    const res = await fetchJSON("/api/human_answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: s.sample_id, letter: quiz.selected }),
    });

    // Update stats
    if (res.stats) {
      quiz.stats = res.stats;
    } else {
      quiz.stats.total++;
      if (res.correct) quiz.stats.correct++;
    }
    updateQuizStats();

    // Mark options
    document.querySelectorAll("#qOptions .option-card").forEach(el => {
      if (el.dataset.letter === res.gold) el.classList.add("correct");
      if (el.dataset.letter === quiz.selected && !res.correct) el.classList.add("incorrect");
    });

    // Show result
    showResult(res);

    // Toggle buttons
    document.getElementById("btnSubmit").style.display = "none";
    document.getElementById("btnNext").style.display = "";
  } catch (e) {
    alert("Submit failed: " + e.message);
    quiz.answered = false;
  }
}

async function requestModelInfer() {
  const s = quiz.queue[quiz.index];
  const btn = document.getElementById("btnModelInfer");
  btn.disabled = true;
  btn.textContent = "Inferring... ⏳";

  try {
    const res = await fetchJSON("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: s.sample_id }),
    });
    const resultModel = document.getElementById("resultModel");
    resultModel.innerHTML = `<span class="result-label">🤖 Model:</span> <span class="${res.correct ? 'text-success' : 'text-error'}">${res.pred_letter || '∅'} ${res.correct ? '✓' : '✗'}</span>`;
    resultModel.style.display = "";
  } catch (e) {
    const resultModel = document.getElementById("resultModel");
    resultModel.innerHTML = `<span class="result-label">🤖 Model:</span> <span class="text-error">Error: ${e.message}</span>`;
    resultModel.style.display = "";
  }

  btn.disabled = false;
  btn.textContent = "Ask Model 🤖";
}

function showResult(res) {
  const qResult = document.getElementById("qResult");
  qResult.style.display = "";
  document.getElementById("resultHuman").innerHTML =
    `<span class="result-label">👤 You:</span> <span class="${res.correct ? 'text-success' : 'text-error'}">${res.submitted} ${res.correct ? '✓' : '✗'}</span>`;
  document.getElementById("resultGold").innerHTML =
    `<span class="result-label">✅ Gold:</span> <span>${res.gold}</span>`;
}

function nextQuestion() {
  quiz.index++;
  if (quiz.index >= quiz.queue.length) {
    showQuizSummary();
    return;
  }
  showQuestion();
}

function updateQuizStats() {
  document.getElementById("quizCorrect").textContent = quiz.stats.correct;
  document.getElementById("quizTotal").textContent = quiz.stats.total;
  document.getElementById("quizAccuracy").textContent =
    quiz.stats.total > 0 ? (quiz.stats.correct / quiz.stats.total * 100).toFixed(1) + "%" : "0%";
}

function showQuizSummary() {
  const container = document.getElementById("quizQuestion");
  const acc = quiz.stats.total > 0 ? (quiz.stats.correct / quiz.stats.total * 100).toFixed(1) : "0";
  container.innerHTML = `
    <div style="text-align:center; padding:3rem 1rem;">
      <h2 style="margin-bottom:1rem;">🎉 Quiz Complete!</h2>
      <p style="font-size:1.25rem; margin-bottom:0.5rem;">
        You answered <strong>${quiz.stats.correct}</strong> / <strong>${quiz.stats.total}</strong> correctly
      </p>
      <p style="font-size:2rem; font-weight:700; color:var(--accent, #f5a623);">${acc}%</p>
      <button class="btn btn-primary" onclick="startQuiz()" style="margin-top:1.5rem;">Play Again</button>
    </div>
  `;
}

/* ═══════════════════════════════════════════════
   6. Samples Module
   ═══════════════════════════════════════════════ */

function renderSample(s, runSelected) {
  const wrap = document.createElement("div");
  wrap.className = "sample-card";

  const head = document.createElement("div");
  head.className = "sample-head";
  head.innerHTML = `
    <span class="tag">${s.sample_id}</span>
    <span class="tag">${s.qa_type}</span>
    <span class="tag">${s.qa_level}</span>
    <span class="tag">${s.language_type}</span>
    <span class="tag gold">gold: ${s.answer}</span>
    ${runSelected ? `<span class="tag ${s.pred_correct ? 'good' : 'bad'}">pred: ${s.pred ?? '∅'}</span>` : ""}
    ${s.error ? `<span class="tag bad">err: ${s.error}</span>` : ""}
  `;
  wrap.appendChild(head);

  const stem = document.createElement("div");
  stem.className = "q-stem";
  stem.textContent = s.question_stem || "(no stem)";
  wrap.appendChild(stem);

  const opts = document.createElement("div");
  opts.className = "options-grid";
  ["A", "B", "C", "D"].forEach(letter => {
    const div = document.createElement("div");
    div.className = "option";
    if (letter === s.answer) div.classList.add("gold");
    if (runSelected && s.pred && letter === s.pred.toUpperCase()) div.classList.add("pred");
    div.textContent = `${letter}: ${s.options[letter] ?? ""}`;
    opts.appendChild(div);
  });
  wrap.appendChild(opts);

  // Audio row
  const audioRow = document.createElement("div");
  audioRow.className = "audio-row";
  if (s.audio_local_available) {
    const a = document.createElement("audio");
    a.controls = true;
    a.src = `/audio/${s.youtube_id}`;
    audioRow.appendChild(a);
  } else {
    const span = document.createElement("span");
    span.className = "muted";
    span.textContent = `Audio not available (${s.youtube_id})`;
    audioRow.appendChild(span);
    if (s.youtube_id) {
      const btn = document.createElement("button");
      btn.className = "btn btn-sm";
      btn.textContent = "Download";
      btn.onclick = async () => {
        btn.disabled = true; btn.textContent = "Downloading…";
        try {
          const r = await fetchJSON("/api/download", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ youtube_id: s.youtube_id }),
          });
          btn.textContent = r.ok ? "Done ✓" : `Failed: ${r.msg}`;
        } catch (e) {
          btn.textContent = `Error: ${e.message}`;
        }
      };
      audioRow.appendChild(btn);
    }
  }
  wrap.appendChild(audioRow);

  // Ask Model button
  const inferRow = document.createElement("div");
  inferRow.className = "infer-row";
  const inferBtn = document.createElement("button");
  inferBtn.className = "btn btn-sm";
  inferBtn.textContent = "Ask Model 🤖";
  inferBtn.onclick = async () => {
    inferBtn.disabled = true;
    inferBtn.textContent = "Inferring...";
    try {
      const res = await fetchJSON("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: s.sample_id }),
      });
      const resSpan = document.createElement("span");
      resSpan.className = res.correct ? "text-success" : "text-error";
      resSpan.textContent = ` → ${res.pred_letter || '∅'} ${res.correct ? '✓' : '✗'}`;
      inferRow.appendChild(resSpan);
    } catch (e) {
      const resSpan = document.createElement("span");
      resSpan.className = "text-error";
      resSpan.textContent = ` Error: ${e.message}`;
      inferRow.appendChild(resSpan);
    }
    inferBtn.disabled = false;
    inferBtn.textContent = "Ask Model 🤖";
  };
  inferRow.appendChild(inferBtn);
  wrap.appendChild(inferRow);

  // Raw text
  if (runSelected && s.raw_text) {
    const raw = document.createElement("div");
    raw.className = "raw-text";
    raw.textContent = `Model output: ${s.raw_text}`;
    wrap.appendChild(raw);
  }

  return wrap;
}

async function loadSamples() {
  const params = new URLSearchParams();
  params.set("offset", state.page * state.pageSize);
  params.set("limit", state.pageSize);

  const fType = document.getElementById("fType");
  const fLevel = document.getElementById("fLevel");
  const fLang = document.getElementById("fLang");
  const fRun = document.getElementById("fRun");
  const fOnlyWrong = document.getElementById("fOnlyWrong");

  if (fType && fType.value) params.set("qa_type", fType.value);
  if (fLevel && fLevel.value) params.set("qa_level", fLevel.value);
  if (fLang && fLang.value) params.set("language_type", fLang.value);
  if (fRun && fRun.value) params.set("run", fRun.value);
  if (fOnlyWrong && fOnlyWrong.checked) params.set("only_wrong", "true");

  try {
    const data = await fetchJSON("/api/samples?" + params.toString());
    state.total = data.total;

    const runSelected = !!(fRun && fRun.value);
    const list = document.getElementById("samplesList");
    list.innerHTML = "";
    data.rows.forEach(s => list.appendChild(renderSample(s, runSelected)));

    document.getElementById("samplesMeta").textContent = `${data.total} matching samples`;
    const totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
    document.getElementById("pageInfo").textContent = `Page ${state.page + 1} / ${totalPages}`;

    // Disable/enable pagination
    const prevBtn = document.getElementById("prevPage");
    const nextBtn = document.getElementById("nextPage");
    if (prevBtn) prevBtn.disabled = state.page <= 0;
    if (nextBtn) nextBtn.disabled = state.page + 1 >= totalPages;
  } catch (e) {
    console.error("loadSamples:", e);
    document.getElementById("samplesList").innerHTML = `<p class="text-error">Failed to load samples: ${e.message}</p>`;
  }
}

/* ═══════════════════════════════════════════════
   7. Initialization
   ═══════════════════════════════════════════════ */

(async function init() {
  initTabs();
  await loadModels();
  await loadSummary();
  await loadSamples();

  // Model management
  const btnLoadModel = document.getElementById("btnLoadModel");
  if (btnLoadModel) btnLoadModel.addEventListener("click", handleLoadModel);

  // Quiz
  const btnStartQuiz = document.getElementById("btnStartQuiz");
  if (btnStartQuiz) btnStartQuiz.addEventListener("click", startQuiz);

  const btnSubmit = document.getElementById("btnSubmit");
  if (btnSubmit) btnSubmit.addEventListener("click", submitAnswer);

  const btnModelInfer = document.getElementById("btnModelInfer");
  if (btnModelInfer) btnModelInfer.addEventListener("click", requestModelInfer);

  const btnNext = document.getElementById("btnNext");
  if (btnNext) btnNext.addEventListener("click", nextQuestion);

  // Samples filtering
  const btnFilter = document.getElementById("btnFilter");
  if (btnFilter) btnFilter.addEventListener("click", () => { state.page = 0; loadSamples(); });

  const prevPage = document.getElementById("prevPage");
  if (prevPage) prevPage.addEventListener("click", () => { if (state.page > 0) { state.page--; loadSamples(); } });

  const nextPage = document.getElementById("nextPage");
  if (nextPage) nextPage.addEventListener("click", () => {
    const totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
    if (state.page + 1 < totalPages) { state.page++; loadSamples(); }
  });

  // Run select in dashboard
  const runSelect = document.getElementById("runSelect");
  if (runSelect) runSelect.addEventListener("change", handleRunChange);
})();
