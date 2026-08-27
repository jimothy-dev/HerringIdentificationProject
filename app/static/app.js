'use strict';

/*
 * Herring Spawn Labeler - frontend.
 * Vanilla JS, talks to the FastAPI backend at /api/* (see project API contract).
 */

const PAGE_SIZE = 50;
const LS_KEY = 'hsl_filters_v1';
const SENSOR_NAMES = { S2: 'Sentinel-2', L8: 'Landsat 8', L9: 'Landsat 9' };

const $ = (id) => document.getElementById(id);

/* Tiny DOM builder: h('div', 'cls', child, 'text', ...) */
function h(tag, cls, ...children) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  for (const c of children) {
    if (c == null) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

const state = {
  status: null,
  statusError: null,
  filters: { year: '', region: '', labeled: 'all' },
  page: 1,
  total: 0,
  records: [],
  years: [],
  regions: [],
  selected: null,        // currently selected record object
  mode: 'spawn',         // 'spawn' | 'offseason'
  scenes: [],
  window: null,
  scenesError: null,
  scenesLoading: false,
  scenesToken: 0,        // guards against out-of-order scene responses
  sceneIdx: 0,
  falseColor: false,
  preload: [],           // Image refs so preloads are not garbage collected
};

/* ---------------- helpers ---------------- */

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.text()).slice(0, 200); } catch (e) { /* ignore */ }
    throw new Error(`HTTP ${res.status}${detail ? ' - ' + detail : ''}`);
  }
  return res.json();
}

let toastTimer = null;
function toast(msg, isError) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'show' + (isError ? ' error' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = ''; }, 4000);
}

function fmtNum(x) {
  return Number(x).toLocaleString('en-US', { maximumFractionDigits: 0 });
}

function fmtDims(rec) {
  if (rec.length_m != null && rec.width_m != null) {
    return `${fmtNum(rec.length_m)} × ${fmtNum(rec.width_m)} m`;
  }
  if (rec.length_m != null) return `${fmtNum(rec.length_m)} m`;
  return '';
}

function labelClass(label) {
  return label ? ' lab-' + label : '';
}

function cloudPill(p) {
  let cls = 'cloud-na';
  let txt = '–';
  if (p != null) {
    txt = Math.round(p) + '%';
    cls = p < 20 ? 'cloud-green' : (p < 60 ? 'cloud-amber' : 'cloud-red');
  }
  const pill = h('span', 'cloud-pill ' + cls, txt);
  pill.title = 'Cloud cover over the spawn site';
  return pill;
}

function daysChip(d) {
  if (d == null) return null;
  const chip = h('span', 'days-chip', d >= 0 ? 'D+' + d : 'D−' + Math.abs(d));
  chip.title = 'Days from spawn start';
  return chip;
}

/* ---------------- status ---------------- */

async function loadStatus() {
  try {
    state.status = await fetchJSON('/api/status');
    state.statusError = null;
  } catch (err) {
    state.status = null;
    state.statusError = String(err.message || err);
  }
  renderStatus();
}

function renderStatus() {
  const wrap = $('status-strip');
  wrap.textContent = '';
  const s = state.status;
  const progress = $('progress-line');

  if (!s) {
    wrap.append(h('div', 'banner amber',
      h('div', 'banner-title', 'Backend unreachable'),
      h('div', 'banner-body', state.statusError || 'Waiting for the server…')));
    progress.textContent = '';
    return;
  }

  if (s.ee_ready && !s.ee_mock) {
    wrap.append(h('div', 'banner ok',
      h('span', 'dot'),
      'Earth Engine ready' + (s.ee_project ? ' · ' + s.ee_project : '')));
  } else {
    wrap.append(h('div', 'banner amber',
      h('div', 'banner-title', 'MOCK MODE — placeholder scenes'),
      s.ee_error ? h('div', 'banner-body', s.ee_error) : null,
      h('div', 'banner-hint',
        'To use real imagery: run "earthengine authenticate", set your cloud project in config.json, then restart the server.')));
  }

  wrap.append(h('div', 'counts',
    h('span', 'count pos', `${s.n_positive} positive`),
    h('span', 'count neg', `${s.n_negative} negative`)));

  const pct = s.n_records ? Math.round((100 * s.n_labeled) / s.n_records) : 0;
  progress.textContent = '';
  const bar = h('div', 'progress-bar');
  const fill = h('div', 'progress-fill');
  fill.style.width = pct + '%';
  bar.append(fill);
  progress.append(
    h('div', 'progress-text', `${s.n_labeled} of ${s.n_records} records labeled (${pct}%)`),
    bar);
}

/* ---------------- filters + records ---------------- */

function loadFilters() {
  try {
    const saved = JSON.parse(localStorage.getItem(LS_KEY) || '{}');
    if (saved && typeof saved === 'object') {
      if (saved.year != null) state.filters.year = String(saved.year);
      if (saved.region != null) state.filters.region = String(saved.region);
      if (['all', 'unlabeled', 'labeled'].includes(saved.labeled)) state.filters.labeled = saved.labeled;
    }
  } catch (e) { /* corrupted storage: ignore */ }
}

function saveFilters() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(state.filters)); } catch (e) { /* ignore */ }
}

function onFilterChange() {
  state.filters.year = $('filter-year').value;
  state.filters.region = $('filter-region').value;
  state.filters.labeled = $('filter-labeled').value;
  saveFilters();
  state.page = 1;
  loadRecords();
}

function fillSelect(sel, values, current, allLabel) {
  sel.textContent = '';
  const all = document.createElement('option');
  all.value = '';
  all.textContent = allLabel;
  sel.append(all);
  for (const v of values) {
    const o = document.createElement('option');
    o.value = String(v);
    o.textContent = String(v);
    sel.append(o);
  }
  sel.value = current ? String(current) : '';
  if (sel.selectedIndex === -1) sel.value = '';
}

function renderFilterOptions() {
  fillSelect($('filter-year'), state.years, state.filters.year, 'All years');
  fillSelect($('filter-region'), state.regions, state.filters.region, 'All regions');
  $('filter-labeled').value = state.filters.labeled;
}

function maxPage() {
  return Math.max(1, Math.ceil(state.total / PAGE_SIZE));
}

async function loadRecords() {
  const q = new URLSearchParams();
  if (state.filters.year) q.set('year', state.filters.year);
  if (state.filters.region) q.set('region', state.filters.region);
  q.set('labeled', state.filters.labeled || 'all');
  q.set('page', String(state.page));
  q.set('page_size', String(PAGE_SIZE));

  const listEl = $('record-list');
  listEl.classList.add('loading');
  try {
    const data = await fetchJSON('/api/records?' + q.toString());
    state.records = data.records || [];
    state.total = data.total || 0;
    state.page = data.page || 1;
    state.years = data.years || [];
    state.regions = data.regions || [];
    renderFilterOptions();
    renderRecordList();
    renderPagination();
  } catch (err) {
    toast('Failed to load records: ' + (err.message || err), true);
  } finally {
    listEl.classList.remove('loading');
  }
}

function renderRecordList() {
  const list = $('record-list');
  list.textContent = '';
  if (!state.records.length) {
    list.append(h('div', 'list-empty', 'No records match these filters.'));
    return;
  }
  for (const rec of state.records) {
    const isSel = state.selected && rec.id === state.selected.id;
    const row = h('div', 'record-row' + (isSel ? ' selected' : ''));
    const meta = [rec.start_date || '—', fmtDims(rec)].filter(Boolean).join(' · ');
    row.append(
      h('div', 'rec-top',
        h('span', 'rec-name', rec.location_name || rec.id),
        rec.n_labels > 0 ? h('span', 'rec-dot') : null),
      h('div', 'rec-meta', meta));
    if (rec.n_labels > 0) row.title = `${rec.n_labels} label(s) saved`;
    row.addEventListener('click', () => selectRecord(rec));
    list.append(row);
  }
  const sel = list.querySelector('.record-row.selected');
  if (sel) sel.scrollIntoView({ block: 'nearest' });
}

function renderPagination() {
  $('page-info').textContent = `${state.page} / ${maxPage()} · ${state.total.toLocaleString('en-US')} records`;
  $('page-prev').disabled = state.page <= 1;
  $('page-next').disabled = state.page >= maxPage();
}

async function gotoPage(p) {
  p = Math.min(Math.max(1, p), maxPage());
  if (p === state.page) return;
  state.page = p;
  await loadRecords();
}

/* ---------------- record selection ---------------- */

function selectRecord(rec) {
  if (!rec) return;
  if (state.selected && state.selected.id === rec.id) return;
  state.selected = rec;
  state.sceneIdx = 0;
  $('notes-input').value = '';
  renderRecordList();
  renderRecordHeader();
  loadScenes();
}

function fact(label, node) {
  return h('span', 'fact',
    h('span', 'fact-label', label),
    h('span', 'fact-value', node));
}

function renderRecordHeader() {
  const rec = state.selected;
  $('main-empty').hidden = !!rec;
  $('record-view').hidden = !rec;
  if (!rec) return;

  const hd = $('record-header');
  hd.textContent = '';

  const link = h('a', 'map-link', `${rec.lat.toFixed(4)}, ${rec.lon.toFixed(4)}`);
  link.href = `https://www.google.com/maps?q=${rec.lat},${rec.lon}`;
  link.target = '_blank';
  link.rel = 'noopener';
  link.title = 'Open in Google Maps';

  const windowTxt = rec.end_date && rec.end_date !== rec.start_date
    ? `${rec.start_date} → ${rec.end_date}`
    : rec.start_date;

  hd.append(
    h('div', 'hdr-line1',
      h('h2', null, rec.location_name || rec.id),
      h('span', 'hdr-sub', `${rec.region} · ${rec.year} · spawn #${rec.spawn_number}`)),
    h('div', 'hdr-facts',
      fact('Spawn window', windowTxt),
      fact('Extent', fmtDims(rec) || '—'),
      fact('Method', rec.method || '—'),
      fact('Position', link)));
}

async function nextRecord() {
  if (!state.records.length) return;
  const idx = state.selected ? state.records.findIndex((r) => r.id === state.selected.id) : -1;
  if (idx === -1) { selectRecord(state.records[0]); return; }
  if (idx < state.records.length - 1) { selectRecord(state.records[idx + 1]); return; }
  if (state.page < maxPage()) {
    state.page += 1;
    await loadRecords();
    if (state.records.length) selectRecord(state.records[0]);
  } else {
    toast('End of record list.');
  }
}

async function prevRecord() {
  if (!state.records.length) return;
  const idx = state.selected ? state.records.findIndex((r) => r.id === state.selected.id) : -1;
  if (idx === -1) { selectRecord(state.records[0]); return; }
  if (idx > 0) { selectRecord(state.records[idx - 1]); return; }
  if (state.page > 1) {
    state.page -= 1;
    await loadRecords();
    if (state.records.length) selectRecord(state.records[state.records.length - 1]);
  } else {
    toast('Start of record list.');
  }
}

/* ---------------- mode + scenes ---------------- */

function setMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  renderModeButtons();
  if (state.selected) loadScenes();
}

function toggleMode() {
  setMode(state.mode === 'spawn' ? 'offseason' : 'spawn');
}

function renderModeButtons() {
  $('mode-spawn').classList.toggle('active', state.mode === 'spawn');
  $('mode-offseason').classList.toggle('active', state.mode === 'offseason');
}

async function loadScenes() {
  const rec = state.selected;
  if (!rec) return;
  const token = ++state.scenesToken;
  state.scenes = [];
  state.window = null;
  state.scenesError = null;
  state.scenesLoading = true;
  state.sceneIdx = 0;
  state.falseColor = false;
  renderScenesArea();
  try {
    const data = await fetchJSON(
      `/api/records/${encodeURIComponent(rec.id)}/scenes?mode=${state.mode}`);
    if (token !== state.scenesToken) return;
    state.scenes = data.scenes || [];
    state.window = data.window || null;
    state.scenesError = data.error || null;
    preloadThumbs(state.scenes);
  } catch (err) {
    if (token !== state.scenesToken) return;
    state.scenesError = String(err.message || err);
  }
  state.scenesLoading = false;
  renderScenesArea();
}

function preloadThumbs(scenes) {
  state.preload = [];
  for (const s of scenes) {
    for (const u of [s.thumb_true, s.thumb_false]) {
      if (!u) continue;
      const img = new Image();
      img.src = u;
      state.preload.push(img);
    }
  }
}

function retryBtn() {
  const b = h('button', 'btn ghost', 'Retry');
  b.type = 'button';
  b.addEventListener('click', loadScenes);
  return b;
}

function renderScenesArea() {
  $('scenes-loading').hidden = !state.scenesLoading;
  $('window-label').textContent = state.window
    ? `window ${state.window.start} → ${state.window.end}` : '';

  const errEl = $('scenes-error');
  const strip = $('filmstrip');
  const viewer = $('viewer');
  const hasScenes = state.scenes.length > 0;

  errEl.hidden = true;
  errEl.textContent = '';

  if (state.scenesLoading) {
    strip.hidden = true;
    viewer.hidden = true;
    strip.textContent = '';
    return;
  }

  if (state.scenesError) {
    errEl.hidden = false;
    errEl.append(
      h('div', 'err-title', hasScenes ? 'Scene lookup problem' : 'Scene lookup failed'),
      h('div', 'err-body', state.scenesError),
      retryBtn());
  } else if (!hasScenes) {
    errEl.hidden = false;
    errEl.append(
      h('div', 'err-title', 'No scenes found'),
      h('div', 'err-body', 'No usable satellite scenes were found in this window.'),
      retryBtn());
  }

  strip.hidden = !hasScenes;
  viewer.hidden = !hasScenes;
  if (hasScenes) {
    renderFilmstrip();
    renderViewer();
  } else {
    strip.textContent = '';
  }
}

function renderFilmstrip() {
  const strip = $('filmstrip');
  strip.textContent = '';
  state.scenes.forEach((s, i) => {
    const cell = h('div',
      'scene-thumb' + labelClass(s.label) + (i === state.sceneIdx ? ' selected' : ''));
    const img = document.createElement('img');
    img.src = s.thumb_true;
    img.alt = `${s.sensor} ${s.date}`;
    img.loading = 'lazy';
    img.addEventListener('error', () => cell.classList.add('img-broken'));
    cell.append(
      img,
      h('div', 'thumb-top',
        h('span', 'sensor-tag', s.sensor),
        cloudPill(s.cloud_region_pct)),
      h('div', 'thumb-bottom',
        h('span', 'thumb-date', s.date),
        daysChip(s.days_from_start)));
    cell.addEventListener('click', () => selectScene(i));
    strip.append(cell);
  });
  const sel = strip.querySelector('.scene-thumb.selected');
  if (sel) sel.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

function currentScene() {
  return state.scenes[state.sceneIdx] || null;
}

function selectScene(i) {
  if (!state.scenes.length) return;
  if (i < 0 || i >= state.scenes.length) return;
  if (i === state.sceneIdx) { renderViewer(); return; }
  state.sceneIdx = i;
  $('notes-input').value = '';
  renderFilmstrip();
  renderViewer();
}

function toggleFalseColor() {
  if (!state.scenes.length) return;
  state.falseColor = !state.falseColor;
  renderViewer();
}

function renderViewer() {
  const s = currentScene();
  if (!s) return;

  const meta = $('viewer-meta');
  meta.textContent = '';
  const fBtn = h('button', 'btn toggle' + (state.falseColor ? ' active' : ''),
    'False color ', h('kbd', null, 'F'));
  fBtn.type = 'button';
  fBtn.addEventListener('click', toggleFalseColor);
  meta.append(
    h('span', 'vm-date', s.date),
    h('span', 'vm-sensor', SENSOR_NAMES[s.sensor] || s.sensor),
    daysChip(s.days_from_start),
    cloudPill(s.cloud_region_pct),
    h('span', 'vm-spacer'),
    fBtn,
    h('span', 'vm-counter', `${state.sceneIdx + 1} / ${state.scenes.length}`));

  const wrap = $('viewer-img-wrap');
  const img = $('viewer-img');
  const url = state.falseColor ? (s.thumb_false || s.thumb_true) : s.thumb_true;
  if (img.getAttribute('src') !== url) {
    wrap.classList.remove('img-broken');
    img.src = url;
  }

  document.querySelectorAll('.label-btn[data-label]').forEach((b) => {
    b.classList.toggle('active', b.dataset.label === s.label);
  });
  $('scene-prev').disabled = state.sceneIdx <= 0;
  $('scene-next').disabled = state.sceneIdx >= state.scenes.length - 1;
}

/* ---------------- labeling ---------------- */

function adjustLabelCount(recId, prev, next) {
  const delta = (prev == null && next != null) ? 1
    : (prev != null && next == null) ? -1 : 0;
  if (!delta) return;
  const targets = [];
  if (state.selected && state.selected.id === recId) targets.push(state.selected);
  const inList = state.records.find((r) => r.id === recId);
  if (inList && !targets.includes(inList)) targets.push(inList);
  for (const t of targets) t.n_labels = Math.max(0, (t.n_labels || 0) + delta);
  renderRecordList();
}

function advance() {
  if (state.sceneIdx < state.scenes.length - 1) {
    selectScene(state.sceneIdx + 1);
  } else {
    nextRecord(); // last scene: move on to the next record in the list
  }
}

async function setLabel(value) {
  const rec = state.selected;
  const scene = currentScene();
  if (!rec || !scene || state.scenesLoading) return;

  const prev = scene.label;
  const notes = $('notes-input').value.trim();

  // optimistic update, then advance immediately
  scene.label = value;
  adjustLabelCount(rec.id, prev, value);
  renderFilmstrip();
  renderViewer();
  advance();

  const body = {
    record_id: rec.id,
    scene_id: scene.scene_id,
    sensor: scene.sensor,
    scene_date: scene.date,
    label: value,
  };
  if (notes) body.notes = notes;

  try {
    await fetchJSON('/api/labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    loadStatus(); // refresh counts in the sidebar
  } catch (err) {
    // rollback
    scene.label = prev;
    adjustLabelCount(rec.id, value, prev);
    if (state.selected && state.selected.id === rec.id && state.scenes.includes(scene)) {
      renderFilmstrip();
      renderViewer();
    }
    toast('Label failed (reverted): ' + (err.message || err), true);
  }
}

async function clearLabel() {
  const rec = state.selected;
  const scene = currentScene();
  if (!rec || !scene || scene.label == null) return;

  const prev = scene.label;
  scene.label = null;
  adjustLabelCount(rec.id, prev, null);
  renderFilmstrip();
  renderViewer();

  try {
    await fetchJSON(
      `/api/labels?record_id=${encodeURIComponent(rec.id)}&scene_id=${encodeURIComponent(scene.scene_id)}`,
      { method: 'DELETE' });
    loadStatus();
  } catch (err) {
    scene.label = prev;
    adjustLabelCount(rec.id, null, prev);
    if (state.selected && state.selected.id === rec.id && state.scenes.includes(scene)) {
      renderFilmstrip();
      renderViewer();
    }
    toast('Clear failed (restored): ' + (err.message || err), true);
  }
}

/* ---------------- keyboard ---------------- */

function onKeydown(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA'
    || t.tagName === 'SELECT' || t.isContentEditable)) {
    if (e.key === 'Escape' || e.key === 'Enter') t.blur();
    return;
  }
  const k = e.key.toLowerCase();
  if (e.repeat && k !== 'arrowleft' && k !== 'arrowright') return;
  switch (k) {
    case 'p': setLabel('positive'); break;
    case 'n': setLabel('negative'); break;
    case 'u': setLabel('unsure'); break;
    case 'x': setLabel('unusable'); break;
    case 'c': clearLabel(); break;
    case 'f': toggleFalseColor(); break;
    case 'm': toggleMode(); break;
    case 'j': nextRecord(); break;
    case 'k': prevRecord(); break;
    case 'arrowleft':
      e.preventDefault();
      selectScene(state.sceneIdx - 1);
      break;
    case 'arrowright':
      e.preventDefault();
      selectScene(state.sceneIdx + 1);
      break;
    default:
      break;
  }
}

/* ---------------- init ---------------- */

function wireEvents() {
  $('filter-year').addEventListener('change', onFilterChange);
  $('filter-region').addEventListener('change', onFilterChange);
  $('filter-labeled').addEventListener('change', onFilterChange);
  $('page-prev').addEventListener('click', () => gotoPage(state.page - 1));
  $('page-next').addEventListener('click', () => gotoPage(state.page + 1));
  $('mode-spawn').addEventListener('click', () => setMode('spawn'));
  $('mode-offseason').addEventListener('click', () => setMode('offseason'));
  $('scene-prev').addEventListener('click', () => selectScene(state.sceneIdx - 1));
  $('scene-next').addEventListener('click', () => selectScene(state.sceneIdx + 1));
  document.querySelectorAll('.label-btn[data-label]').forEach((b) => {
    b.addEventListener('click', () => setLabel(b.dataset.label));
  });
  $('btn-clear').addEventListener('click', clearLabel);
  const vImg = $('viewer-img');
  vImg.addEventListener('error', () => $('viewer-img-wrap').classList.add('img-broken'));
  vImg.addEventListener('load', () => $('viewer-img-wrap').classList.remove('img-broken'));
  document.addEventListener('keydown', onKeydown);
}

function init() {
  loadFilters();
  $('filter-labeled').value = state.filters.labeled;
  renderModeButtons();
  wireEvents();
  loadStatus();
  setInterval(loadStatus, 60000);
  loadRecords();
}

document.addEventListener('DOMContentLoaded', init);
