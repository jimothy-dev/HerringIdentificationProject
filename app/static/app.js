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
  nRecordsLabeled: null, // records with >=1 label (from /api/records?labeled=labeled)
  notesByScene: {},      // scene_id -> saved notes for the selected record
  filters: {
    year: '', region: '', labeled: 'all', hideEmpty: true,
    sensors: { s2: true, l8: true, l9: true },  // satellite toggles
    maxCloud: 70,                               // cloud ceiling (server default 70)
  },
  nHiddenEmpty: 0,       // records excluded by hide_empty on the last list load
  recordsToken: 0,       // guards against out-of-order record-list responses
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
  cloudOverride: false,  // fetch scenes with max_cloud=100; reset on record/mode switch
  nCloudFiltered: 0,     // scenes dropped by the cloud ceiling on the last scene load
  maxCloudPct: null,     // ceiling applied on the last scene load (for messaging)
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
    // n_labeled counts label ROWS (scenes); record progress needs the count of
    // records with >=1 label, which the labeled filter's total provides.
    try {
      const d = await fetchJSON('/api/records?labeled=labeled&page=1&page_size=1');
      state.nRecordsLabeled = d.total || 0;
    } catch (e) {
      state.nRecordsLabeled = null;
    }
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

  const nRec = state.nRecordsLabeled;
  let pct;
  let text;
  if (nRec != null) {
    pct = s.n_records ? Math.min(100, Math.round((100 * nRec) / s.n_records)) : 0;
    text = `${nRec} of ${s.n_records} records labeled (${pct}%) · ${s.n_labeled} labels`;
  } else {
    // Labeled-record count unavailable: show label rows only, clamp the bar.
    pct = s.n_records ? Math.min(100, Math.round((100 * s.n_labeled) / s.n_records)) : 0;
    text = `${s.n_labeled} labels saved · ${s.n_records} records`;
  }
  progress.textContent = '';
  const bar = h('div', 'progress-bar');
  const fill = h('div', 'progress-fill');
  fill.style.width = pct + '%';
  bar.append(fill);
  progress.append(h('div', 'progress-text', text), bar);
}

/* ---------------- filters + records ---------------- */

function loadFilters() {
  try {
    const saved = JSON.parse(localStorage.getItem(LS_KEY) || '{}');
    if (saved && typeof saved === 'object') {
      // Only accept a numeric year: a garbage saved value would 422 the
      // int-typed query param before the option list can correct it.
      if (saved.year != null && /^\d+$/.test(String(saved.year))) state.filters.year = String(saved.year);
      if (saved.region != null) state.filters.region = String(saved.region);
      if (['all', 'unlabeled', 'labeled'].includes(saved.labeled)) state.filters.labeled = saved.labeled;
      // Default ON; only an explicit boolean overrides it.
      if (typeof saved.hideEmpty === 'boolean') state.filters.hideEmpty = saved.hideEmpty;
      if (saved.sensors && typeof saved.sensors === 'object') {
        for (const s of ['s2', 'l8', 'l9']) {
          if (typeof saved.sensors[s] === 'boolean') state.filters.sensors[s] = saved.sensors[s];
        }
        // Never restore an all-off state: it would query nothing forever.
        if (!state.filters.sensors.s2 && !state.filters.sensors.l8 && !state.filters.sensors.l9) {
          state.filters.sensors = { s2: true, l8: true, l9: true };
        }
      }
      if (typeof saved.maxCloud === 'number' && saved.maxCloud >= 0 && saved.maxCloud <= 100) {
        state.filters.maxCloud = Math.round(saved.maxCloud);
      }
    }
  } catch (e) { /* corrupted storage: ignore */ }
}

function saveFilters() {
  try { localStorage.setItem(LS_KEY, JSON.stringify(state.filters)); } catch (e) { /* ignore */ }
}

/* ---------------- scene controls (satellites + cloud ceiling) ---------------- */

function renderSceneControls() {
  $('sensor-s2').checked = state.filters.sensors.s2;
  $('sensor-l8').checked = state.filters.sensors.l8;
  $('sensor-l9').checked = state.filters.sensors.l9;
  $('cloud-slider').value = String(state.filters.maxCloud);
  $('cloud-number').value = String(state.filters.maxCloud);
}

function onSensorChange(e) {
  const boxes = { s2: $('sensor-s2'), l8: $('sensor-l8'), l9: $('sensor-l9') };
  if (!boxes.s2.checked && !boxes.l8.checked && !boxes.l9.checked) {
    e.target.checked = true; // at least one satellite must stay on
    toast('At least one satellite must stay enabled', true);
    return;
  }
  for (const s of ['s2', 'l8', 'l9']) state.filters.sensors[s] = boxes[s].checked;
  saveFilters();
  e.target.blur(); // keep single-key label shortcuts working
  if (state.selected) loadScenes();
}

let cloudReloadTimer = null;
function setCloudCeiling(value, reloadDelayMs) {
  let v = Math.round(Number(value));
  if (!isFinite(v)) v = 70;
  v = Math.max(0, Math.min(100, v));
  state.filters.maxCloud = v;
  $('cloud-slider').value = String(v);
  $('cloud-number').value = String(v);
  saveFilters();
  // Debounced so dragging the slider fires one query, not twenty.
  clearTimeout(cloudReloadTimer);
  cloudReloadTimer = setTimeout(() => { if (state.selected) loadScenes(); }, reloadDelayMs);
}

/* ---------------- dataset export ---------------- */

async function exportDataset() {
  const btn = $('export-btn');
  const note = $('export-note');
  btn.disabled = true;
  note.textContent = 'building…';
  try {
    const res = await fetchJSON('/api/export', { method: 'POST' });
    if (!res.ok) throw new Error(res.error || 'export failed');
    note.textContent = `${res.n_labels} labels · ${res.n_chips} chips · ${(res.size_bytes / 1048576).toFixed(1)} MB`;
    const a = document.createElement('a');
    a.href = '/api/export/' + encodeURIComponent(res.filename);
    a.download = res.filename;
    document.body.append(a);
    a.click();
    a.remove();
    toast('Dataset exported — download started');
  } catch (err) {
    note.textContent = '';
    toast('Export failed: ' + (err.message || err), true);
  }
  btn.disabled = false;
}

function onFilterChange() {
  state.filters.year = $('filter-year').value;
  state.filters.region = $('filter-region').value;
  state.filters.labeled = $('filter-labeled').value;
  saveFilters();
  state.page = 1;
  loadRecords();
}

function onHideEmptyChange(e) {
  state.filters.hideEmpty = e.target.checked;
  saveFilters();
  state.page = 1;
  loadRecords();
  // Drop focus so single-key label shortcuts are not swallowed by the
  // focused-INPUT guard in onKeydown.
  e.target.blur();
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
  return sel.value; // effective value ('' if `current` was not an option)
}

/* Returns true if a stale saved filter had to be reset to '' (so the caller
 * should re-query -- otherwise state.filters keeps filtering by a value the
 * UI no longer shows). */
function renderFilterOptions() {
  let changed = false;
  const year = fillSelect($('filter-year'), state.years, state.filters.year, 'All years');
  if (year !== state.filters.year) { state.filters.year = year; changed = true; }
  const region = fillSelect($('filter-region'), state.regions, state.filters.region, 'All regions');
  if (region !== state.filters.region) { state.filters.region = region; changed = true; }
  $('filter-labeled').value = state.filters.labeled;
  if (changed) saveFilters();
  return changed;
}

function maxPage() {
  return Math.max(1, Math.ceil(state.total / PAGE_SIZE));
}

/* opts.background: a refresh the user did not ask for (sweep polling) -- it
 * must not yank the sidebar scroll position back to the selected row. */
async function loadRecords(opts) {
  const background = !!(opts && opts.background);
  const q = new URLSearchParams();
  if (state.filters.year) q.set('year', state.filters.year);
  if (state.filters.region) q.set('region', state.filters.region);
  q.set('labeled', state.filters.labeled || 'all');
  q.set('hide_empty', state.filters.hideEmpty ? '1' : '0');
  q.set('page', String(state.page));
  q.set('page_size', String(PAGE_SIZE));

  // The sweep's periodic list refresh can overlap a user-initiated load;
  // token-guard so only the newest response is rendered (same pattern as
  // scenesToken). Records refreshes never touch the scene viewer, so an
  // in-flight scene load is unaffected.
  const token = ++state.recordsToken;
  const listEl = $('record-list');
  listEl.classList.add('loading');
  try {
    const data = await fetchJSON('/api/records?' + q.toString());
    if (token !== state.recordsToken) return;
    state.records = data.records || [];
    state.total = data.total || 0;
    state.page = data.page || 1;
    state.years = data.years || [];
    state.regions = data.regions || [];
    state.nHiddenEmpty = data.n_hidden_empty || 0;
    const corrected = renderFilterOptions();
    if (corrected) {
      // A stale saved filter was reset; re-query once with the fixed filters
      // (the reset values are always valid, so this cannot loop).
      state.page = 1;
      await loadRecords(opts);
      return;
    }
    if (!state.records.length && state.page > maxPage()) {
      // The matching set shrank below this page (e.g. the sweep hid empties
      // out from under a background refresh); clamp and re-query instead of
      // rendering a bogus "No records match" over thousands of records.
      // Cannot loop: page 1 is always <= maxPage().
      state.page = maxPage();
      await loadRecords(opts);
      return;
    }
    renderRecordList(!background);
    renderPagination();
    renderHiddenNote();
  } catch (err) {
    if (token === state.recordsToken) {
      toast('Failed to load records: ' + (err.message || err), true);
    }
  } finally {
    // A stale request must not clear the loading state of a newer one.
    if (token === state.recordsToken) listEl.classList.remove('loading');
  }
}

function renderHiddenNote() {
  const el = $('hidden-note');
  if (state.filters.hideEmpty && state.nHiddenEmpty > 0) {
    el.textContent = `${state.nHiddenEmpty} hidden`;
    el.title = `${state.nHiddenEmpty} record(s) with no usable scenes at the current cloud ceiling are hidden`;
  } else {
    el.textContent = '';
    el.title = '';
  }
}

function renderRecordList(scrollToSelected = true) {
  const list = $('record-list');
  list.textContent = '';
  if (!state.records.length) {
    const extra = (state.filters.hideEmpty && state.nHiddenEmpty > 0)
      ? ` ${state.nHiddenEmpty} record(s) with no usable scenes are hidden — uncheck "Hide empty" to show them.`
      : '';
    list.append(h('div', 'list-empty', 'No records match these filters.' + extra));
    return;
  }
  for (const rec of state.records) {
    const isSel = state.selected && rec.id === state.selected.id;
    const row = h('div', 'record-row' + (isSel ? ' selected' : ''));
    const meta = [rec.start_date || '—', fmtDims(rec)].filter(Boolean).join(' · ');
    const metaEl = h('div', 'rec-meta', meta);
    if (rec.scene_status === 'empty') {
      // Shown despite being empty (labeled record, or "Hide empty" is off).
      metaEl.append(' · ', h('span', 'rec-noscene', 'no scenes'));
      metaEl.title = 'No usable scenes at the current cloud ceiling';
    }
    row.append(
      h('div', 'rec-top',
        h('span', 'rec-name', rec.location_name || rec.id),
        rec.n_labels > 0 ? h('span', 'rec-dot') : null),
      metaEl);
    if (rec.n_labels > 0) row.title = `${rec.n_labels} label(s) saved`;
    row.addEventListener('click', () => selectRecord(rec));
    list.append(row);
  }
  const sel = list.querySelector('.record-row.selected');
  if (sel && scrollToSelected) sel.scrollIntoView({ block: 'nearest' });
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
  state.cloudOverride = false; // the cloud override lasts one record only
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

  // Last record of the page. With the 'unlabeled' filter, records the user
  // just labeled have dropped OUT of the server-side set, shifting later
  // records up -- naive page+1 arithmetic would silently skip a page's worth
  // of records. Re-anchor by record id on a fresh view of the same page.
  if (state.filters.labeled === 'unlabeled') {
    const leftId = state.selected.id;
    const prevIds = new Set(state.records.map((r) => r.id));
    await loadRecords();
    if (!state.records.length && state.page > 1) {
      // The set shrank below this page; clamp and reload.
      state.page = Math.min(state.page, maxPage());
      await loadRecords();
    }
    const i = state.records.findIndex((r) => r.id === leftId);
    if (i !== -1) {
      // Current record is still unlabeled and on this page.
      if (i < state.records.length - 1) { selectRecord(state.records[i + 1]); return; }
      if (state.page < maxPage()) {
        state.page += 1;
        await loadRecords();
        if (state.records.length) { selectRecord(state.records[0]); return; }
      }
      toast('End of record list.');
      return;
    }
    // Current record dropped out (it got labeled). Any record now on the page
    // that was NOT here before must have shifted in from later in the list.
    const shiftedIn = state.records.find((r) => !prevIds.has(r.id));
    if (shiftedIn) { selectRecord(shiftedIn); return; }
    // End of the set: fall back to any still-unlabeled leftover on this page.
    const leftover = state.records.find((r) => r.id !== leftId);
    if (leftover) { selectRecord(leftover); return; }
    toast('End of record list.');
    return;
  }

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

  // Mirror of nextRecord(): with the 'unlabeled' filter the set may have
  // shifted, so re-anchor by record id instead of trusting page arithmetic.
  if (state.filters.labeled === 'unlabeled') {
    const leftId = state.selected.id;
    await loadRecords();
    if (!state.records.length && state.page > 1) {
      state.page = Math.min(state.page, maxPage());
      await loadRecords();
    }
    let i = state.records.findIndex((r) => r.id === leftId);
    if (i > 0) { selectRecord(state.records[i - 1]); return; }
    if (i === -1 && state.page > 1) {
      // The current record shifted up onto an earlier page (or dropped out).
      state.page -= 1;
      await loadRecords();
      i = state.records.findIndex((r) => r.id === leftId);
      if (i > 0) { selectRecord(state.records[i - 1]); return; }
      if (i === -1 && state.records.length) {
        selectRecord(state.records[state.records.length - 1]);
        return;
      }
    } else if (i === 0 && state.page > 1) {
      state.page -= 1;
      await loadRecords();
      if (state.records.length) {
        selectRecord(state.records[state.records.length - 1]);
        return;
      }
    }
    toast('Start of record list.');
    return;
  }

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
  state.cloudOverride = false; // mode switch clears the cloud override
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

function notesForScene(s) {
  return (s && state.notesByScene[s.scene_id]) || '';
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
  state.notesByScene = {};
  state.nCloudFiltered = 0;
  state.maxCloudPct = null;
  clearSegForScene(); // record/mode switch invalidates segment points+mask
  // Never carry a draft note across scene lists (mode toggle / retry) -- it
  // would get saved onto an unrelated scene.
  $('notes-input').value = '';
  renderScenesArea();
  // cloudOverride disables the regional-cloud ceiling for this record until
  // the user switches record/mode (override requests skip the backend's
  // availability cache; labeling works the same either way).
  let scenesUrl = `/api/records/${encodeURIComponent(rec.id)}/scenes?mode=${state.mode}`;
  if (state.cloudOverride) scenesUrl += '&max_cloud=100';
  else if (state.filters.maxCloud !== 70) scenesUrl += '&max_cloud=' + state.filters.maxCloud;
  const sensorsOn = ['s2', 'l8', 'l9'].filter((s) => state.filters.sensors[s]);
  if (sensorsOn.length > 0 && sensorsOn.length < 3) scenesUrl += '&sensors=' + sensorsOn.join(',');
  try {
    const [data, labelData] = await Promise.all([
      fetchJSON(scenesUrl),
      fetchJSON(`/api/labels?record_id=${encodeURIComponent(rec.id)}`).catch(() => null),
    ]);
    if (token !== state.scenesToken) return;
    state.scenes = data.scenes || [];
    state.window = data.window || null;
    state.scenesError = data.error || null;
    state.nCloudFiltered = data.n_cloud_filtered || 0;
    state.maxCloudPct = data.max_cloud_pct != null ? data.max_cloud_pct : null;
    if (labelData && Array.isArray(labelData.labels)) {
      for (const row of labelData.labels) state.notesByScene[row.scene_id] = row.notes || '';
    }
    $('notes-input').value = notesForScene(currentScene());
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

function fmtCeiling(x) {
  if (x == null) return '?'; // Number(null) is 0 -- it would pass isFinite
  const n = Number(x);
  if (!isFinite(n)) return '?';
  return String(Math.round(n * 10) / 10);
}

function renderScenesArea() {
  $('scenes-loading').hidden = !state.scenesLoading;
  $('window-label').textContent = state.window
    ? `window ${state.window.start} → ${state.window.end}`
      + (state.cloudOverride ? ' · cloud ceiling off'
        : (state.filters.maxCloud !== 70 ? ` · cloud ≤ ${state.filters.maxCloud}%` : ''))
    : '';

  const errEl = $('scenes-error');
  const strip = $('filmstrip');
  const viewer = $('viewer');
  const hasScenes = state.scenes.length > 0;

  errEl.hidden = true;
  errEl.textContent = '';
  errEl.classList.remove('cloudy');

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
    if (state.nCloudFiltered > 0 && !state.cloudOverride) {
      // Scenes exist but the regional-cloud ceiling dropped all of them.
      errEl.classList.add('cloudy');
      const showBtn = h('button', 'btn accent', 'Show cloudy scenes');
      showBtn.type = 'button';
      showBtn.addEventListener('click', () => {
        state.cloudOverride = true; // lasts until record/mode switch
        loadScenes();
      });
      errEl.append(
        h('div', 'err-title', 'All scenes too cloudy'),
        h('div', 'err-body',
          `${state.nCloudFiltered} scene(s) in this window exceeded the `
          + `${fmtCeiling(state.maxCloudPct)}% cloud ceiling.`),
        showBtn);
    } else {
      errEl.append(
        h('div', 'err-title', 'No scenes found'),
        h('div', 'err-body', 'No usable satellite scenes were found in this window.'),
        retryBtn());
    }
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
  clearSegForScene(); // segment points/mask belong to one scene only
  // Show the notes already saved for this scene (if any) so a relabel edits
  // rather than silently discards them.
  $('notes-input').value = notesForScene(state.scenes[i]);
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
  const sBtn = h('button', 'btn toggle seg-btn' + (seg.active ? ' active' : ''),
    'Segment ', h('kbd', null, 'S'));
  sBtn.type = 'button';
  sBtn.title = 'Segment mode: click a feature, SAM outlines it and scores spawn likelihood';
  sBtn.addEventListener('click', toggleSegMode);
  const sBadge = h('span', 'seg-badge');
  sBadge.id = 'seg-badge';
  sBadge.hidden = true;
  meta.append(...[
    h('span', 'vm-date', s.date),
    h('span', 'vm-sensor', SENSOR_NAMES[s.sensor] || s.sensor),
    daysChip(s.days_from_start),
    cloudPill(s.cloud_region_pct),
    h('span', 'vm-spacer'),
    fBtn,
    sBtn,
    sBadge,
    h('span', 'vm-counter', `${state.sceneIdx + 1} / ${state.scenes.length}`),
  ].filter(Boolean));

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
  syncSegDom();
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
  const prevNotes = notesForScene(scene);

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
  // Send notes when non-empty, or when the user cleared a previously saved
  // note (explicit '' overwrites; an absent field preserves server-side).
  if (notes || notes !== prevNotes) {
    body.notes = notes;
    state.notesByScene[scene.scene_id] = notes;
  }

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
    if (prevNotes) state.notesByScene[scene.scene_id] = prevNotes;
    else delete state.notesByScene[scene.scene_id];
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
  const prevNotes = notesForScene(scene);
  scene.label = null;
  delete state.notesByScene[scene.scene_id];
  if (currentScene() === scene) $('notes-input').value = '';
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
    if (prevNotes) {
      state.notesByScene[scene.scene_id] = prevNotes;
      if (currentScene() === scene) $('notes-input').value = prevNotes;
    }
    adjustLabelCount(rec.id, null, prev);
    if (state.selected && state.selected.id === rec.id && state.scenes.includes(scene)) {
      renderFilmstrip();
      renderViewer();
    }
    toast('Clear failed (restored): ' + (err.message || err), true);
  }
}

/* ---------------- segment mode ---------------- */
/*
 * Frozen contract (backend built separately):
 *   GET  /api/segment/status  -> {state, backend, device, classifier, error, hint}
 *   POST /api/segment/warmup  -> {ok, state}  (idempotent, returns immediately)
 *   POST /api/segment {record_id, scene_id, sensor, points:[{x,y,label}]}
 *        -> always HTTP 200; ok:false carries {state, error?, hint?};
 *           ok:true carries mask_png (same WxH as the true-color thumb), area/score/etc.
 * Click coords are normalized 0-1 in TRUE-COLOR thumb image space.
 */

const seg = {
  active: false,
  status: null,        // last /api/segment/status payload (or synthesized error)
  statusTimer: null,   // status polling interval id
  points: [],          // accumulated clicks {x,y,label} for the CURRENT scene
  result: null,        // last successful /api/segment response for this scene
  inFlight: false,     // a /api/segment request is running
  pending: false,      // click(s) arrived while in flight; resend once with ALL points
  reqToken: 0,         // bumped on clear/scene-switch to invalidate stale responses
  retryTimer: null,
  retryCount: 0,
  retrying: false,     // model-still-loading auto-retry in progress
  lastError: null,     // {error, hint} from the last failed segment call
  sceneEncoded: false, // scene has returned a mask (first click encodes, ~5-30s)
};

function segWarmup() {
  // Fire-and-forget: the status poll surfaces any problem.
  fetchJSON('/api/segment/warmup', { method: 'POST' }).catch(() => {});
}

function startSegPolling() {
  pollSegStatus();
  if (seg.statusTimer == null) seg.statusTimer = setInterval(pollSegStatus, 3000);
}

function stopSegPolling() {
  if (seg.statusTimer != null) {
    clearInterval(seg.statusTimer);
    seg.statusTimer = null;
  }
}

async function pollSegStatus() {
  let s;
  try {
    s = await fetchJSON('/api/segment/status');
  } catch (err) {
    s = {
      state: 'error', backend: null, device: null, classifier: null,
      error: String(err.message || err),
      hint: 'Segment endpoint unreachable — is the backend running with segment support?',
    };
  }
  seg.status = s;
  updateSegBadge();
  renderSegPanel();
  if (s.state === 'ready' || s.state === 'error') stopSegPolling();
}

function toggleSegMode() {
  seg.active = !seg.active;
  if (seg.active) {
    segWarmup();
    startSegPolling();
  } else {
    stopSegPolling();
    // Cancel the model-loading retry loop too: with segment mode off nothing
    // may keep POSTing /api/segment or resurrect the status poll.
    if (seg.retryTimer) { clearTimeout(seg.retryTimer); seg.retryTimer = null; }
    seg.retrying = false;
    seg.pending = false;
    seg.retryCount = 0;
  }
  const btn = document.querySelector('.seg-btn');
  if (btn) btn.classList.toggle('active', seg.active);
  syncSegDom();
}

/* Reflect segment state into the DOM (cursor, overlay, panel, badge). */
function syncSegDom() {
  const wrap = $('viewer-img-wrap');
  wrap.classList.toggle('seg-active', seg.active);
  const show = seg.active && !!currentScene();
  $('seg-overlay').hidden = !show;
  $('seg-panel').hidden = !show;
  updateSegBadge();
  if (show) {
    renderSegMask();
    renderSegMarkers();
    positionSegOverlay();
    renderSegPanel();
  }
}

function updateSegBadge() {
  const b = $('seg-badge');
  if (!b) return; // viewer meta not rendered yet
  if (!seg.active) { b.hidden = true; return; }
  b.hidden = false;
  const s = seg.status;
  const st = s ? s.state : 'loading';
  b.className = 'seg-badge '
    + (st === 'ready' ? 'ready' : st === 'error' ? 'error' : 'loading');
  if (st === 'ready') {
    b.textContent = (s.backend || 'sam').toUpperCase() + ' · ' + (s.device || '?');
    b.title = 'Segmentation ready · classifier: ' + (s.classifier || 'heuristic');
  } else if (st === 'error') {
    b.textContent = 'SAM error';
    b.title = (s && s.error) || 'unknown error';
  } else {
    b.textContent = 'loading SAM…';
    b.title = 'Model is ' + st + ' — first load can take a while';
  }
}

/* Displayed content rect of #viewer-img (object-fit: contain math). */
function positionSegOverlay() {
  const overlay = $('seg-overlay');
  if (overlay.hidden) return;
  const wrap = $('viewer-img-wrap');
  const img = $('viewer-img');
  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  if (!img.complete || !natW || !natH) {
    overlay.style.width = '0px';
    overlay.style.height = '0px';
    return;
  }
  const ir = img.getBoundingClientRect();
  const wr = wrap.getBoundingClientRect();
  const scale = Math.min(ir.width / natW, ir.height / natH);
  const cw = natW * scale;
  const ch = natH * scale;
  overlay.style.left = (ir.left - wr.left - wrap.clientLeft + (ir.width - cw) / 2) + 'px';
  overlay.style.top = (ir.top - wr.top - wrap.clientTop + (ir.height - ch) / 2) + 'px';
  overlay.style.width = cw + 'px';
  overlay.style.height = ch + 'px';
}

function renderSegMask() {
  const m = $('seg-mask');
  if (seg.result && seg.result.mask_png) {
    if (m.getAttribute('src') !== seg.result.mask_png) m.src = seg.result.mask_png;
    m.hidden = false;
  } else {
    m.removeAttribute('src');
    m.hidden = true;
  }
}

function renderSegMarkers() {
  const box = $('seg-points');
  box.textContent = '';
  for (const p of seg.points) {
    const d = h('div', 'seg-pt ' + (p.label === 0 ? 'bg' : 'fg'));
    d.style.left = (p.x * 100) + '%';
    d.style.top = (p.y * 100) + '%';
    box.append(d);
  }
}

function onViewerClick(e) {
  if (!seg.active || state.scenesLoading) return;
  const scene = currentScene();
  if (!scene) return;
  const img = $('viewer-img');
  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  if (!img.complete || !natW || !natH) return;
  const ir = img.getBoundingClientRect();
  if (!ir.width || !ir.height) return;
  const scale = Math.min(ir.width / natW, ir.height / natH);
  const cw = natW * scale;
  const ch = natH * scale;
  const x = (e.clientX - ir.left - (ir.width - cw) / 2) / cw;
  const y = (e.clientY - ir.top - (ir.height - ch) / 2) / ch;
  if (x < 0 || x > 1 || y < 0 || y > 1) return; // letterbox / background click
  e.preventDefault();
  seg.points.push({ x, y, label: e.shiftKey ? 0 : 1 });
  renderSegMarkers();
  positionSegOverlay();
  sendSegment();
}

/* A fresh click cancels any loading-retry loop and sends immediately
 * (or queues one resend if a request is already in flight). */
function sendSegment() {
  if (seg.retryTimer) { clearTimeout(seg.retryTimer); seg.retryTimer = null; }
  seg.retryCount = 0;
  seg.retrying = false;
  if (seg.inFlight) {
    seg.pending = true; // coalesces: the resend carries ALL accumulated points
    renderSegPanel();
    return;
  }
  doSegmentRequest();
}

async function doSegmentRequest() {
  const rec = state.selected;
  const scene = currentScene();
  if (!rec || !scene || !seg.points.length) return;
  const token = seg.reqToken;
  const body = {
    record_id: rec.id,
    scene_id: scene.scene_id,
    sensor: scene.sensor,
    points: seg.points.map((p) => ({ x: p.x, y: p.y, label: p.label })),
  };
  seg.inFlight = true;
  seg.lastError = null;
  renderSegPanel();

  let data;
  try {
    data = await fetchJSON('/api/segment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    data = { ok: false, state: 'error', error: String(err.message || err) };
  }
  seg.inFlight = false;

  if (token !== seg.reqToken) {
    // Scene switched / cleared while in flight: drop this response, but honor
    // a click queued for the NEW context.
    if (seg.pending) {
      seg.pending = false;
      if (seg.points.length) doSegmentRequest();
    }
    return;
  }

  if (data.ok) {
    seg.result = data;
    seg.sceneEncoded = true;
    seg.retrying = false;
    seg.retryCount = 0;
    seg.lastError = null;
    // A success proves the model is up even if the poll has not caught up.
    if (!seg.status || seg.status.state !== 'ready') {
      seg.status = {
        state: 'ready', backend: data.backend || null, device: data.device || null,
        classifier: data.spawn_score && data.spawn_score.kind === 'model' ? 'trained' : 'heuristic',
        error: null, hint: null,
      };
      stopSegPolling();
      updateSegBadge();
    }
    renderSegMask();
    positionSegOverlay();
    renderSegPanel();
  } else if (data.state === 'loading' || data.state === 'cold') {
    if (!seg.active) { seg.retrying = false; return; } // user left segment mode
    if (data.state === 'cold') segWarmup();
    startSegPolling(); // keep the badge live while we wait
    if (seg.retryCount < 5) {
      seg.retryCount += 1;
      seg.retrying = true;
      seg.pending = false; // the retry resends ALL accumulated points anyway
      renderSegPanel();
      seg.retryTimer = setTimeout(() => {
        seg.retryTimer = null;
        if (seg.active && token === seg.reqToken && seg.points.length && !seg.inFlight) {
          doSegmentRequest();
        }
      }, 3000);
      return;
    }
    seg.retrying = false;
    seg.lastError = {
      error: 'Model is still loading — give it a moment, then click again.',
      hint: data.hint || null,
    };
    renderSegPanel();
  } else {
    seg.retrying = false;
    seg.lastError = {
      error: data.error || 'Segmentation failed.',
      hint: data.hint || null,
    };
    renderSegPanel();
  }

  if (seg.pending) {
    seg.pending = false;
    if (seg.points.length && !seg.inFlight) doSegmentRequest();
  }
}

/* Clears points + mask for the current scene (Esc / Clear / scene switch). */
function clearSegForScene() {
  seg.points = [];
  seg.result = null;
  seg.pending = false;
  seg.sceneEncoded = false;
  seg.retrying = false;
  seg.lastError = null;
  seg.reqToken += 1;
  if (seg.retryTimer) { clearTimeout(seg.retryTimer); seg.retryTimer = null; }
  seg.retryCount = 0;
  renderSegMask();
  renderSegMarkers();
  renderSegPanel();
}

function segClearBtn() {
  const b = h('button', 'btn ghost seg-clear', 'Clear ', h('kbd', null, 'Esc'));
  b.type = 'button';
  b.addEventListener('click', clearSegForScene);
  return b;
}

function fmtArea(m2, px) {
  if (m2 == null) return fmtNum(px) + ' px';
  if (m2 >= 1e6) {
    const km2 = m2 / 1e6;
    return (km2 >= 10 ? km2.toFixed(1) : km2.toFixed(2)) + ' km²';
  }
  return fmtNum(m2) + ' m²';
}

function renderSegPanel() {
  const panel = $('seg-panel');
  panel.textContent = '';
  if (!seg.active) return;

  const st = seg.status;
  if (st && st.state === 'error') {
    panel.append(h('div', 'seg-err',
      'Segmentation backend error: ' + (st.error || 'unknown')));
    if (st.hint) panel.append(h('div', 'seg-hint', st.hint));
    return;
  }

  if (seg.inFlight || seg.pending) {
    panel.append(h('div', 'seg-msg', h('div', 'spinner sm'), 'Segmenting…'));
    if (!seg.sceneEncoded) {
      panel.append(h('div', 'seg-hint-first',
        'First click on a new scene encodes it (~5–30 s).'));
    }
    return;
  }

  if (seg.retrying) {
    panel.append(h('div', 'seg-msg', h('div', 'spinner sm'),
      'Model still loading — retrying…'));
    return;
  }

  if (seg.lastError) {
    panel.append(h('div', 'seg-err', seg.lastError.error));
    if (seg.lastError.hint) panel.append(h('div', 'seg-hint', seg.lastError.hint));
    panel.append(segClearBtn());
    return;
  }

  const r = seg.result;
  if (!r) {
    panel.append(h('div', 'seg-hint',
      'Click a feature in the image to segment it · Shift+click adds a background (exclude) point · Esc clears.'));
    if (!seg.sceneEncoded) {
      panel.append(h('div', 'seg-hint-first',
        'First click on a new scene encodes it (~5–30 s).'));
    }
    return;
  }

  const score = r.spawn_score || {};
  const prob = Math.max(0, Math.min(1, Number(score.prob) || 0));
  const pct = Math.round(prob * 100);
  const bar = h('div', 'seg-score-bar');
  const fill = h('div', 'seg-score-fill');
  fill.style.width = pct + '%';
  bar.append(fill);
  const scoreWrap = h('div', 'seg-score-wrap',
    h('div', 'seg-score-top',
      h('span', 'seg-score-label', 'Spawn score'),
      score.kind ? h('span', 'seg-kind-pill', score.kind) : null,
      h('span', 'seg-score-val', pct + '%')),
    bar,
    score.note ? h('div', 'seg-note', score.note) : null);

  const stats = h('div', 'seg-stats',
    fact('Area', fmtArea(r.area_m2, r.area_px)),
    fact('Coverage', r.coverage_pct != null ? r.coverage_pct.toFixed(2) + '%' : '—'),
    fact('SAM IoU', r.sam_iou != null ? Number(r.sam_iou).toFixed(2) : '—'),
    fact('Time', fmtNum(r.timing_ms) + ' ms'),
    fact('Model', (r.backend || '?').toUpperCase() + ' · ' + (r.device || '?')));

  panel.append(h('div', 'seg-row', scoreWrap, stats, segClearBtn()));
}

/* ---------------- scene availability sweep ---------------- */
/*
 * Frozen contract (backend built separately):
 *   POST /api/availability/sweep {year, region}
 *        -> {ok:true, state:"started"|"already_running", total:int}
 *   GET  /api/availability/status
 *        -> {running, done, total, empty_found, checked_total, unknown_total}
 * The sweep caches scene availability per record; /api/records reflects the
 * cache immediately, so the list is refreshed while the sweep runs.
 */

const sweep = {
  timer: null,      // the single polling interval (guarded: never stacked)
  pollCount: 0,     // refresh the record list every 3rd poll
  last: null,       // last /api/availability/status payload
  doneTimer: null,  // clears the brief "done" note
};

async function startSweep() {
  const btn = $('scan-btn');
  if (btn.disabled) return;
  btn.disabled = true; // polling re-enables it once running is false
  btn.blur();          // keep single-key label shortcuts live
  const body = {
    year: state.filters.year ? Number(state.filters.year) : null,
    region: state.filters.region || null,
  };
  try {
    const d = await fetchJSON('/api/availability/sweep', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (d.state === 'already_running') toast('Scene scan already running.');
    beginSweepPolling();
  } catch (err) {
    btn.disabled = false;
    toast('Could not start scan: ' + (err.message || err), true);
  }
}

function beginSweepPolling() {
  if (sweep.timer != null) return; // already polling: never leak a 2nd interval
  sweep.pollCount = 0;
  sweep.timer = setInterval(pollSweep, 4000);
  pollSweep();
}

function stopSweepPolling() {
  if (sweep.timer != null) {
    clearInterval(sweep.timer);
    sweep.timer = null;
  }
}

async function pollSweep() {
  let s = null;
  try {
    s = await fetchJSON('/api/availability/status');
  } catch (err) { /* backend unreachable: fall through and stop polling */ }
  if (s) {
    sweep.last = s;
  } else if (sweep.last && sweep.last.running) {
    // Failed poll: clear the stale "running" payload BEFORE rendering, or
    // renderSweepUI would freeze the progress label and leave the Scan
    // button disabled forever (a disabled button can never restart a scan).
    sweep.last = Object.assign({}, sweep.last, { running: false });
  }

  if (s && s.running) {
    sweep.pollCount += 1;
    renderSweepUI();
    // Refresh the sidebar every 3rd poll so newly-found empties drop out
    // progressively. loadRecords keeps selection by record id and never
    // touches the scene viewer, so an in-flight scene load is unaffected.
    if (sweep.pollCount % 3 === 0) loadRecords({ background: true });
    return;
  }

  const wasActive = sweep.timer != null;
  stopSweepPolling();
  renderSweepUI();
  if (s && wasActive) {
    // Sweep finished (or was finished by the time we first polled).
    showSweepDone(s);
    loadRecords({ background: true });
  }
}

function renderSweepUI() {
  const btn = $('scan-btn');
  const prog = $('scan-progress');
  const s = sweep.last;
  const running = !!(s && s.running);
  btn.disabled = running;
  if (s) {
    btn.title = `${s.checked_total} record(s) checked for scenes · ${s.unknown_total} unchecked`;
  }
  if (running) {
    if (sweep.doneTimer) { clearTimeout(sweep.doneTimer); sweep.doneTimer = null; }
    prog.textContent = `scanning ${s.done}/${s.total}…`;
    prog.hidden = false;
  } else if (sweep.doneTimer == null) {
    prog.textContent = '';
    prog.hidden = true;
  }
}

function showSweepDone(s) {
  const prog = $('scan-progress');
  prog.textContent = `done — ${s.empty_found} empty hidden`;
  prog.hidden = false;
  if (sweep.doneTimer) clearTimeout(sweep.doneTimer);
  sweep.doneTimer = setTimeout(() => {
    sweep.doneTimer = null;
    prog.textContent = '';
    prog.hidden = true;
  }, 6000);
}

/* One-shot at startup: resume progress display if a sweep is running
 * (e.g. the page was reloaded mid-sweep). */
async function sweepInitCheck() {
  try {
    const s = await fetchJSON('/api/availability/status');
    sweep.last = s;
    if (s.running) beginSweepPolling();
    else renderSweepUI();
  } catch (err) { /* endpoint absent / backend down: button still works on click */ }
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
    case 's': toggleSegMode(); break;
    case 'escape':
      if (seg.active && (seg.points.length || seg.result)) clearSegForScene();
      break;
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
  $('hide-empty').addEventListener('change', onHideEmptyChange);
  $('scan-btn').addEventListener('click', startSweep);
  $('sensor-s2').addEventListener('change', onSensorChange);
  $('sensor-l8').addEventListener('change', onSensorChange);
  $('sensor-l9').addEventListener('change', onSensorChange);
  $('cloud-slider').addEventListener('input', (e) => setCloudCeiling(e.target.value, 450));
  $('cloud-number').addEventListener('change', (e) => setCloudCeiling(e.target.value, 100));
  $('export-btn').addEventListener('click', exportDataset);
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
  vImg.addEventListener('load', () => {
    $('viewer-img-wrap').classList.remove('img-broken');
    positionSegOverlay(); // new natural size => recompute the contain rect
  });
  const vWrap = $('viewer-img-wrap');
  vWrap.addEventListener('click', onViewerClick);
  window.addEventListener('resize', positionSegOverlay);
  if (window.ResizeObserver) {
    new ResizeObserver(positionSegOverlay).observe(vWrap);
  }
  document.addEventListener('keydown', onKeydown);
}

function init() {
  loadFilters();
  $('filter-labeled').value = state.filters.labeled;
  $('hide-empty').checked = state.filters.hideEmpty;
  renderSceneControls();
  renderModeButtons();
  wireEvents();
  loadStatus();
  setInterval(loadStatus, 60000);
  loadRecords();
  sweepInitCheck();
}

document.addEventListener('DOMContentLoaded', init);
