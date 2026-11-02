/* ==========================================================================
   NephroCheck — client-side logic
   Builds the form from GET /features, validates, calls POST /predict,
   renders results and keeps live system status up to date.
   ========================================================================== */

const form = document.getElementById("patient-form");
const formSections = document.getElementById("form-sections");
const formAlert = document.getElementById("form-alert");
const predictBtn = document.getElementById("predict-btn");
const resetBtn = document.getElementById("reset-btn");
const sampleBtns = document.querySelectorAll("[data-sample]");
const resultBody = document.getElementById("result-body");
const resultPanel = document.getElementById("result-panel");
const refreshBtn = document.getElementById("refresh-btn");

const apiVersion = document.getElementById("api-version");
const footerVersion = document.getElementById("footer-version");
const fieldCount = document.getElementById("field-count");

const healthStatus = document.getElementById("health-status");
const readyStatus = document.getElementById("ready-status");
const metaModel = document.getElementById("meta-model");
const metaVersion = document.getElementById("meta-version");
const metaFeatures = document.getElementById("meta-features");
const metaTrained = document.getElementById("meta-trained");

const metricTotal = document.getElementById("metric-total");
const metricSuccess = document.getElementById("metric-success");
const metricFailed = document.getElementById("metric-failed");
const metricAvg = document.getElementById("metric-avg");
const metricUptime = document.getElementById("metric-uptime");
const lastUpdated = document.getElementById("last-updated");

const statAccuracy = document.getElementById("stat-accuracy");
const statF1 = document.getElementById("stat-f1");
const statAuc = document.getElementById("stat-auc");

let featuresPayload = null;
let fieldMap = new Map();

const sectionMeta = {
  vitals: { icon: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="8" cy="6" r="2.8"/><path d="M4 13c0-2.2 1.8-4 4-4s4 1.8 4 4" stroke-linecap="round"/></svg>`, index: "01" },
  urine:  { icon: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M5.5 2.5h5L11 7c.8 2.6-.6 5.5-3.2 6.5-2.6 1-5.5-.6-6.5-3.2A5 5 0 0 1 5 7l.5-4.5Z" stroke-linecap="round" stroke-linejoin="round"/></svg>`, index: "02" },
  blood:  { icon: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M8 2.5c.8 2 .8 4.2 0 6.2l-.2.5.4.1c1.8.4 3.2 2 3.2 3.9 0 2.2-1.8 3.8-3.9 3.8s-3.9-1.6-3.9-3.8c0-1.9 1.4-3.5 3.2-3.9l.4-.1-.2-.5c-.8-2-.8-4.2 0-6.2" stroke-linecap="round" stroke-linejoin="round"/></svg>`, index: "03" },
  history:{ icon: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3.5" y="4" width="9" height="9" rx="1.4"/><path d="M6 2.5v2.5M10 2.5v2.5M3.5 7h9" stroke-linecap="round"/></svg>`, index: "04" },
};

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") {
    if (Math.abs(value) >= 1000) return value.toLocaleString();
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(2).replace(/\.?0+$/, "");
  }
  return value;
}

function formatPct(value) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatSeconds(sec) {
  if (sec === null || sec === undefined) return "—";
  if (sec < 60) return `${sec.toFixed(0)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  const mm = m % 60;
  return `${h}h ${mm}m`;
}

function setBadge(el, text, kind) {
  el.textContent = text;
  el.classList.remove("neutral", "online", "offline");
  if (kind) el.classList.add(kind);
}

function showFormAlert(message, kind = "danger") {
  formAlert.textContent = message;
  formAlert.className = `alert ${kind}`;
  formAlert.hidden = false;
  formAlert.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearFormAlert() {
  formAlert.hidden = true;
  formAlert.textContent = "";
  formAlert.className = "alert";
}

function clearFieldErrors() {
  document.querySelectorAll(".field-error").forEach((el) => el.remove());
  document.querySelectorAll(".input.is-invalid").forEach((el) => el.classList.remove("is-invalid"));
}

function buildForm() {
  formSections.innerHTML = "";
  fieldMap.clear();

  if (!featuresPayload || !Array.isArray(featuresPayload.sections)) {
    formSections.innerHTML = `<div class="empty-state"><strong>Could not load the form</strong>Refresh the page or check /features.</div>`;
    predictBtn.disabled = true;
    return;
  }

  let total = 0;
  featuresPayload.sections.forEach((section, index) => {
    const meta = sectionMeta[section.id] || { icon: "", index: String(index + 1).padStart(2, "0") };
    const sectionEl = document.createElement("section");
    sectionEl.className = "form-section";
    sectionEl.dataset.section = section.id;

    const fields = Array.isArray(section.fields) ? section.fields : [];
    total += fields.length;

    const grid = document.createElement("div");
    grid.className = "field-grid";

    fields.forEach((field) => {
      fieldMap.set(field.name, field);
      grid.appendChild(renderField(field));
    });

    sectionEl.innerHTML = `
      <div class="section-head">
        <span class="section-index">${meta.index}</span>
        <div class="section-icon" aria-hidden="true">${meta.icon}</div>
        <div class="section-title">
          <h3>${escapeHtml(section.title)}</h3>
          <p>${escapeHtml(section.description || "")}</p>
        </div>
      </div>
    `;
    sectionEl.appendChild(grid);
    formSections.appendChild(sectionEl);
  });

  fieldCount.textContent = `${total} fields`;
  predictBtn.disabled = false;
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
}

function renderField(field) {
  const wrap = document.createElement("div");
  wrap.className = "field";
  wrap.dataset.field = field.name;

  const labelRow = document.createElement("div");
  labelRow.className = "label-row";

  const label = document.createElement("label");
  label.className = "field-label";
  label.htmlFor = `field-${field.name}`;
  label.innerHTML = `${escapeHtml(field.label)}${field.required ? ' <span class="req">*</span>' : ""}`;

  labelRow.appendChild(label);

  if (!field.required) {
    const tag = document.createElement("span");
    tag.className = "field-tag";
    tag.textContent = "Optional";
    labelRow.appendChild(tag);
  }

  const inputWrap = document.createElement("div");
  inputWrap.className = "input-wrap";
  if (field.type === "numeric" && field.unit && field.unit.length < 9) inputWrap.classList.add("has-unit");

  if (field.type === "categorical") {
    const selectWrap = document.createElement("div");
    selectWrap.className = "select-wrap";
    const select = document.createElement("select");
    select.className = "input";
    select.id = `field-${field.name}`;
    select.name = field.name;

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select…";
    select.appendChild(placeholder);

    (field.options || []).forEach((opt) => {
      const o = document.createElement("option");
      o.value = opt.value;
      o.textContent = opt.label || opt.value;
      select.appendChild(o);
    });

    selectWrap.appendChild(select);
    inputWrap.appendChild(selectWrap);
  } else {
    const input = document.createElement("input");
    input.className = "input";
    input.id = `field-${field.name}`;
    input.name = field.name;
    input.inputMode = "decimal";
    input.autocomplete = "off";

    if (field.name === "sg" && Array.isArray(field.options)) {
      const listId = `list-sg`;
      if (!document.getElementById(listId)) {
        const datalist = document.createElement("datalist");
        datalist.id = listId;
        field.options.forEach((opt) => {
          const option = document.createElement("option");
          option.value = opt.value;
          datalist.appendChild(option);
        });
        document.body.appendChild(datalist);
      }
      input.setAttribute("list", listId);
      input.placeholder = "e.g. 1.02";
    }

    if (field.min !== undefined) input.min = field.min;
    if (field.max !== undefined) input.max = field.max;
    if (field.step !== undefined) input.step = field.step;
    input.type = "number";

    inputWrap.appendChild(input);

    if (field.unit && field.unit.length < 9) {
      const unit = document.createElement("span");
      unit.className = "unit";
      unit.textContent = field.unit;
      inputWrap.appendChild(unit);
    }
  }

  const unitSuffix = field.unit && !field.unit.includes("scale") ? ` ${field.unit}` : "";
  let hint = "";
  if (field.observed_min !== undefined && field.observed_max !== undefined) {
    hint = `Observed in training: ${field.observed_min}–${field.observed_max}${unitSuffix}`;
  } else if (field.type === "numeric" && field.min !== undefined && field.max !== undefined) {
    hint = `Accepted range: ${field.min}–${field.max}${unitSuffix}`;
  }
  if (field.type === "categorical" && !hint) {
    hint = "Leave blank if not available.";
  }
  if (field.name === "al" || field.name === "su") {
    hint += " (0 = none, 5 = heavy)";
  }

  const hintEl = document.createElement("p");
  hintEl.className = "field-hint";
  hintEl.id = `hint-${field.name}`;
  hintEl.textContent = hint;

  wrap.appendChild(labelRow);
  wrap.appendChild(inputWrap);
  if (hint) wrap.appendChild(hintEl);

  const control = inputWrap.querySelector(".input");
  if (control) {
    if (hint) control.setAttribute("aria-describedby", hintEl.id);
    if (field.required) control.setAttribute("aria-required", "true");
    control.addEventListener("input", () => {
      control.classList.remove("is-invalid");
      const err = wrap.querySelector(".field-error");
      if (err) err.remove();
      clearFormAlert();
    });
    control.addEventListener("change", () => {
      control.classList.remove("is-invalid");
      const err = wrap.querySelector(".field-error");
      if (err) err.remove();
      clearFormAlert();
    });
  }

  return wrap;
}

function showFieldError(name, message) {
  const fieldEl = document.querySelector(`.field[data-field="${CSS.escape(name)}"]`);
  if (!fieldEl) return;
  fieldEl.querySelectorAll(".field-error").forEach((e) => e.remove());
  const control = fieldEl.querySelector(".input");
  if (control) control.classList.add("is-invalid");
  const err = document.createElement("p");
  err.className = "field-error";
  err.textContent = message;
  const hint = fieldEl.querySelector(".field-hint");
  if (hint) hint.insertAdjacentElement("afterend", err);
  else fieldEl.appendChild(err);
}

function validateClient() {
  clearFieldErrors();
  clearFormAlert();

  if (!featuresPayload) {
    showFormAlert("Form not loaded yet. Click refresh or reload the page.");
    return null;
  }

  const payload = {};
  const errors = [];

  (featuresPayload.features || []).forEach((field) => {
    const control = document.getElementById(`field-${field.name}`);
    if (!control) return;

    let value = "";
    if (control.tagName === "SELECT") {
      value = control.value;
    } else {
      value = control.value.trim();
    }

    if (value === "") {
      if (field.required) {
        errors.push({ name: field.name, message: "This field is required." });
      }
      return;
    }

    if (field.type === "numeric") {
      const num = Number(value);
      if (Number.isNaN(num)) {
        errors.push({ name: field.name, message: "Enter a valid number." });
        return;
      }
      if (field.dtype === "integer" && !Number.isInteger(num)) {
        errors.push({ name: field.name, message: "Enter a whole number." });
        return;
      }
      if (field.min !== undefined && num < field.min) errors.push({ name: field.name, message: `Must be ≥ ${field.min}.` });
      if (field.max !== undefined && num > field.max) errors.push({ name: field.name, message: `Must be ≤ ${field.max}.` });
      if (errors.some((e) => e.name === field.name)) return;
      payload[field.name] = num;
      return;
    }

    if (field.type === "categorical") {
      const allowed = (field.options || []).map((o) => o.value);
      if (!allowed.includes(value)) {
        errors.push({ name: field.name, message: `Choose one of: ${allowed.join(", ")}.` });
        return;
      }
      payload[field.name] = value;
    }
  });

  errors.forEach((e) => showFieldError(e.name, e.message));

  if (errors.length) {
    showFormAlert("Please fix the highlighted fields before predicting.");
    return null;
  }

  return payload;
}

function setButtonLoading(loading) {
  if (loading) {
    predictBtn.disabled = true;
    predictBtn.classList.add("is-loading");
    const spinner = predictBtn.querySelector(".spinner");
    if (spinner) spinner.hidden = false;
    predictBtn.querySelector(".btn-label").textContent = "Predicting…";
  } else {
    predictBtn.disabled = false;
    predictBtn.classList.remove("is-loading");
    const spinner = predictBtn.querySelector(".spinner");
    if (spinner) spinner.hidden = true;
    predictBtn.querySelector(".btn-label").textContent = "Predict CKD risk";
  }
}

function renderResult(data) {
  resultPanel.setAttribute("aria-busy", "false");
  const isCkd = data.prediction === "CKD";
  const pCkd = Number(data.probability_ckd) || 0;
  const pNot = Number(data.probability_not_ckd) || 1 - pCkd;

  resultBody.innerHTML = `
    <div class="result-content">
      <div class="result-top">
        <div>
          <p class="eyebrow">${isCkd ? "Elevated risk" : "Lower risk"}</p>
          <h2>Prediction: ${escapeHtml(data.prediction)}</h2>
        </div>
        <span class="result-badge ${isCkd ? "danger" : "success"}">${escapeHtml(data.prediction)}</span>
      </div>
      <p class="result-lead">${isCkd ? "The model estimates a high likelihood of Chronic Kidney Disease based on the provided values." : "The model estimates a low likelihood of Chronic Kidney Disease based on the provided values."}</p>

      <div class="prob-list">
        <div>
          <div class="prob-head">
            <span><span class="swatch danger" aria-hidden="true"></span> Probability of CKD</span>
            <strong>${formatPct(pCkd)}</strong>
          </div>
          <div class="bar"><span class="bar-fill danger" style="width: ${Math.max(0, Math.min(100, pCkd * 100))}%"></span></div>
        </div>
        <div>
          <div class="prob-head">
            <span><span class="swatch success" aria-hidden="true"></span> Probability of Not CKD</span>
            <strong>${formatPct(pNot)}</strong>
          </div>
          <div class="bar"><span class="bar-fill success" style="width: ${Math.max(0, Math.min(100, pNot * 100))}%"></span></div>
        </div>
      </div>

      <div class="result-footer">
        <span>Model v${escapeHtml(data.model_version || "—")}</span>
        <span>${escapeHtml(new Date().toLocaleTimeString())}</span>
      </div>
      <div class="result-disclaimer">
        <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="8" cy="8" r="6"/><path d="M8 7v3M8 5h.01" stroke-linecap="round"/></svg>
        <span>${escapeHtml(data.disclaimer || "Model output only, not a medical diagnosis.")}</span>
      </div>
    </div>
  `;
}

async function fetchJson(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

async function loadFeatures() {
  try {
    featuresPayload = await fetchJson("/features");
    if (featuresPayload.model_version) {
      apiVersion.textContent = `v${featuresPayload.model_version}`;
      if (footerVersion) footerVersion.textContent = featuresPayload.model_version;
    }
    buildForm();
  } catch (err) {
    console.error("Failed to load features", err);
    formSections.innerHTML = `<div class="empty-state"><strong>Could not load the form schema</strong>${escapeHtml(err.message)}<br><br>Make sure the API is running and <code>/features</code> is reachable.</div>`;
    predictBtn.disabled = true;
  }
}

async function loadSystemStatus() {
  try {
    const [health, ready, info, metrics] = await Promise.allSettled([
      fetchJson("/health"),
      fetchJson("/ready"),
      fetchJson("/model-info"),
      fetchJson("/metrics"),
    ]);

    if (health.status === "fulfilled" && health.value.status === "ok") {
      setBadge(healthStatus, "OK", "online");
    } else {
      setBadge(healthStatus, "DOWN", "offline");
    }

    if (ready.status === "fulfilled" && ready.value.ready) {
      setBadge(readyStatus, "LOADED", "online");
    } else {
      setBadge(readyStatus, "NOT READY", "offline");
    }

    if (info.status === "fulfilled") {
      const m = info.value;
      metaModel.textContent = m.model_name || "—";
      metaVersion.textContent = m.model_version || "—";
      metaFeatures.textContent = m.n_features ? `${m.n_features}` : "—";
      metaTrained.textContent = m.training_date ? m.training_date.split("T")[0] : "—";
      if (m.model_version) {
        apiVersion.textContent = `v${m.model_version}`;
        if (footerVersion) footerVersion.textContent = m.model_version;
      }
      if (m.evaluation) {
        if (m.evaluation.accuracy !== undefined) statAccuracy.textContent = `${(m.evaluation.accuracy * 100).toFixed(1)}%`;
        if (m.evaluation.f1_score !== undefined) statF1.textContent = m.evaluation.f1_score.toFixed(3);
        if (m.evaluation.roc_auc !== undefined) statAuc.textContent = m.evaluation.roc_auc.toFixed(3);
      }
    }

    if (metrics.status === "fulfilled") {
      const m = metrics.value;
      metricTotal.textContent = formatNumber(m.total_requests);
      metricSuccess.textContent = formatNumber(m.successful_predictions);
      metricFailed.textContent = formatNumber(m.failed_requests);
      metricAvg.textContent = m.average_response_time_ms !== undefined ? `${m.average_response_time_ms.toFixed(1)} ms` : "—";
      metricUptime.textContent = formatSeconds(m.uptime_seconds);
    }

    const now = new Date();
    lastUpdated.textContent = now.toLocaleTimeString();
  } catch (err) {
    console.error("Failed to refresh status", err);
    setBadge(healthStatus, "ERR", "offline");
    setBadge(readyStatus, "ERR", "offline");
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearFormAlert();
  clearFieldErrors();

  const payload = validateClient();
  if (!payload) {
    const firstError = document.querySelector(".field.is-invalid, .field-error");
    if (firstError) firstError.scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }

  setButtonLoading(true);
  resultPanel.setAttribute("aria-busy", "true");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.status === 422) {
      const body = await res.json();
      clearFieldErrors();
      (body.errors || []).forEach((err) => {
        if (err.field) showFieldError(err.field, err.message);
      });
      showFormAlert(body.detail || "Validation failed.");
      setButtonLoading(false);
      resultPanel.setAttribute("aria-busy", "false");
      const firstError = document.querySelector(".field-error");
      if (firstError) firstError.scrollIntoView({ behavior: "smooth", block: "center" });
      await loadSystemStatus();
      return;
    }

    if (res.status === 503) {
      const body = await res.json();
      showFormAlert(body.detail || "Model is not ready. Run 'python train.py'.", "info");
      setButtonLoading(false);
      resultPanel.setAttribute("aria-busy", "false");
      await loadSystemStatus();
      return;
    }

    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      showFormAlert(body.detail || "Something went wrong. Please try again.", "danger");
      setButtonLoading(false);
      resultPanel.setAttribute("aria-busy", "false");
      await loadSystemStatus();
      return;
    }

    const data = await res.json();
    renderResult(data);
    setButtonLoading(false);
    await loadSystemStatus();
  } catch (err) {
    console.error(err);
    showFormAlert("Network error. Check that the API server is running at this origin.", "danger");
    setButtonLoading(false);
    resultPanel.setAttribute("aria-busy", "false");
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  clearFieldErrors();
  clearFormAlert();
  resultBody.innerHTML = `
    <div class="result-placeholder">
      <div class="result-orb" aria-hidden="true">
        <svg viewBox="0 0 32 32" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          <path d="M6.5 17.5h4.2l2.1-5.7 3.5 10.4 2.2-6 1.6 2.6h5.4"/>
          <path d="M16 3.5c6.9 0 12.5 5 12.5 11.4C28.5 23.5 22.8 28.5 16 28.5S3.5 23.5 3.5 14.9C3.5 8.5 9.1 3.5 16 3.5Z" opacity=".4"/>
        </svg>
      </div>
      <h2>No assessment yet</h2>
      <p>Complete the required fields, then select <strong>Predict CKD risk</strong> to see the model's estimate.</p>
      <ul class="mini-list">
        <li><svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="m3 8.4 3.1 3L13 4.9" stroke-linecap="round" stroke-linejoin="round"/></svg>Required: age, blood pressure, specific gravity, albumin, hemoglobin, creatinine.</li>
        <li><svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="m3 8.4 3.1 3L13 4.9" stroke-linecap="round" stroke-linejoin="round"/></svg>Optional lab values sharpen the estimate.</li>
        <li><svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="m3 8.4 3.1 3L13 4.9" stroke-linecap="round" stroke-linejoin="round"/></svg>Use the sample buttons to explore a demo case.</li>
      </ul>
    </div>
  `;
  resultPanel.setAttribute("aria-busy", "false");
});

sampleBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const kind = btn.dataset.sample;
    const samples = featuresPayload?.samples || {};
    const sample = samples[kind] || samples[Object.keys(samples)[0]];
    if (!sample) return;

    clearFieldErrors();
    clearFormAlert();
    form.reset();

    Object.entries(sample).forEach(([name, value]) => {
      const control = document.getElementById(`field-${name}`);
      if (!control) return;
      if (control.tagName === "SELECT") {
        control.value = value;
      } else {
        control.value = value;
      }
      control.classList.remove("is-invalid");
    });

    formAlert.textContent = kind === "ckd" ? "Loaded a sample CKD patient. You can edit any values and predict." : "Loaded a sample healthy patient. You can edit any values and predict.";
    formAlert.className = "alert success";
    formAlert.hidden = false;
  });
});

refreshBtn.addEventListener("click", async () => {
  refreshBtn.classList.add("spinning");
  await loadSystemStatus();
  setTimeout(() => refreshBtn.classList.remove("spinning"), 500);
});

loadFeatures();
loadSystemStatus();
setInterval(loadSystemStatus, 45000);
