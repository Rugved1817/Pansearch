(function() {
  const state = {
    rows: [],
    columns: [],
    panMatches: [],
    relatedMatches: [],
    pageSize: 20,
    pageIndex: 0,
    quickFilter: '',
  };

  const els = {
    csvFile: document.getElementById('csvFile'),
    statRows: document.getElementById('statRows'),
    statCols: document.getElementById('statCols'),
    panInput: document.getElementById('panInput'),
    btnSearchPan: document.getElementById('btnSearchPan'),
    btnExpandName: document.getElementById('btnExpandName'),
    nameThreshold: document.getElementById('nameThreshold'),
    nameThresholdVal: document.getElementById('nameThresholdVal'),
    ageTolerance: document.getElementById('ageTolerance'),
    countPan: document.getElementById('countPan'),
    countRelated: document.getElementById('countRelated'),
    tableContainer: document.getElementById('tableContainer'),
    pagination: document.getElementById('pagination'),
    quickFilter: document.getElementById('quickFilter'),
    themeToggle: document.getElementById('themeToggle'),
  };

  // Theme toggle
  els.themeToggle?.addEventListener('change', (e) => {
    document.documentElement.classList.toggle('dark', e.target.checked);
  });

  // CSV Upload
  els.csvFile.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        state.rows = res.data;
        state.columns = res.meta.fields || Object.keys(state.rows[0] || {});
        updateStats();
        resetResults();
      }
    });
  });

  function updateStats() {
    els.statRows.textContent = String(state.rows.length);
    els.statCols.textContent = String(state.columns.length);
  }

  function resetResults() {
    state.panMatches = [];
    state.relatedMatches = [];
    state.pageIndex = 0;
    els.countPan.textContent = '0';
    els.countRelated.textContent = '0';
    renderTable([]);
    renderPagination(0, 0);
    els.btnExpandName.disabled = true;
  }

  // PAN utilities
  const PAN_RE = /(?<![A-Z0-9])[A-Z]{5}[A-Z0-9]{4}[A-Z](?![A-Z])/i;
  function toLatin(s) { return (s || '').toString(); }
  function canonicalizePan(input) {
    const t = toLatin(input).toUpperCase().trim();
    const m = t.match(PAN_RE);
    return m ? m[0].toUpperCase() : null;
  }
  function extractPansFromText(text) {
    const t = toLatin(text).toUpperCase();
    const out = [];
    let m;
    const re = new RegExp(PAN_RE.source, 'gi');
    while ((m = re.exec(t))) out.push(m[0].toUpperCase());
    return out;
  }

  // Name similarity (simple blend)
  function normalizeText(s) {
    return (s || '').toString().normalize('NFKC').toLowerCase()
      .replace(/[\u200c\u200d]/g, '')
      .replace(/[\.,:;\-_/\\()\[\]{}|@#*!?"'`]+/g, ' ')
      .replace(/\s+/g, ' ').trim();
  }
  function nameSimilarity(a, b) {
    const aN = normalizeText(a), bN = normalizeText(b);
    if (!aN || !bN) return 0;
    const aT = new Set(aN.split(' '));
    const bT = new Set(bN.split(' '));
    const inter = [...aT].filter(x => bT.has(x)).length;
    const uni = new Set([...aT, ...bT]).size;
    const jacc = inter / Math.max(1, uni);
    // crude character overlap
    const seq = sequenceRatio(aN, bN);
    return 0.6 * seq + 0.4 * jacc;
  }
  function sequenceRatio(a, b) {
    // simple LCS-based approximation
    const m = a.length, n = b.length;
    const dp = Array.from({length: m+1}, () => new Array(n+1).fill(0));
    for (let i=1;i<=m;i++) for (let j=1;j<=n;j++) dp[i][j] = a[i-1]===b[j-1] ? dp[i-1][j-1]+1 : Math.max(dp[i-1][j], dp[i][j-1]);
    const lcs = dp[m][n];
    return (2*lcs)/(m+n);
  }

  function extractCandidateNames(row) {
    const out = [];
    const nameLike = ['buyer','seller','party','parties','executant','claimant','buyer_name','seller_name','name','sname','bname'];
    for (const k of Object.keys(row)) {
      if (nameLike.includes(k.toLowerCase())) {
        const v = row[k];
        if (v) {
          const parts = String(v).split(/[;\n\r]|\b\d\):/);
          for (const p of parts) { const t = p.trim(); if (t) out.push(t); }
        }
      }
    }
    return dedupe(out.map(s => s.trim()).filter(Boolean)).slice(0, 20);
  }

  function dedupe(arr) {
    const seen = new Set();
    const out = [];
    for (const v of arr) { const k = normalizeText(v); if (!k || seen.has(k)) continue; seen.add(k); out.push(v); }
    return out;
  }

  function extractAge(row) {
    for (const v of Object.values(row)) {
      const s = String(v||'');
      const m = s.match(/(?:वय|age)\s*[-:]?\s*(\d{1,3})/i);
      if (m) return parseInt(m[1], 10);
    }
    return null;
  }
  function extractAddress(row) {
    let best = '';
    for (const v of Object.values(row)) {
      const s = String(v||'');
      if (s.includes('पत्ता') || /address/i.test(s)) { if (s.length > best.length) best = s; }
    }
    return best;
  }
  function addressSimilarity(a, b) {
    const A = new Set(normalizeText(a).split(' '));
    const B = new Set(normalizeText(b).split(' '));
    const inter = [...A].filter(x => B.has(x)).length;
    const uni = new Set([...A, ...B]).size;
    return inter / Math.max(1, uni);
  }

  // UI actions
  els.nameThreshold.addEventListener('input', () => {
    els.nameThresholdVal.textContent = els.nameThreshold.value;
  });
  els.quickFilter.addEventListener('input', () => {
    state.quickFilter = els.quickFilter.value;
    render();
  });

  els.btnSearchPan.addEventListener('click', () => {
    const pan = canonicalizePan(els.panInput.value);
    if (!pan) { alert('Enter a valid PAN'); return; }
    const matches = [];
    for (const row of state.rows) {
      const values = Object.values(row);
      let hit = false;
      for (const v of values) {
        if (extractPansFromText(v).includes(pan)) { hit = true; break; }
      }
      if (hit) matches.push(row);
    }
    state.panMatches = matches;
    els.countPan.textContent = String(matches.length);
    state.relatedMatches = [];
    els.countRelated.textContent = '0';
    els.btnExpandName.disabled = matches.length === 0;
    state.pageIndex = 0;
    renderList(matches);
  });

  els.btnExpandName.addEventListener('click', () => {
    if (state.panMatches.length === 0) return;
    const nameThreshold = parseFloat(els.nameThreshold.value);
    const ageTol = parseInt(els.ageTolerance.value, 10) || 0;

    const candidateNames = dedupe(state.panMatches.flatMap(extractCandidateNames));
    const ages = state.panMatches.map(extractAge).filter(x => Number.isFinite(x));
    const targetAge = ages.length ? ages.sort((a,b)=>a-b)[Math.floor(ages.length/2)] : null;
    const addresses = state.panMatches.map(extractAddress).filter(Boolean);
    const targetAddr = addresses.sort((a,b)=>b.length-a.length)[0] || '';

    const related = [];
    for (const row of state.rows) {
      const rowNames = extractCandidateNames(row);
      let best = 0;
      for (const rn of rowNames) for (const cn of candidateNames) best = Math.max(best, nameSimilarity(rn, cn));
      if (best < nameThreshold) continue;
      let bonus = 0;
      if (targetAddr) bonus += 0.2 * addressSimilarity(targetAddr, extractAddress(row));
      if (targetAge != null) {
        const ra = extractAge(row);
        if (ra != null && Math.abs(ra - targetAge) <= ageTol) bonus += 0.1;
      }
      if (best + bonus >= nameThreshold) related.push(row);
    }
    // ensure pan matches included
    const key = (r) => state.columns.map(c => `${c}=${r[c]||''}`).join('|');
    const seen = new Set();
    const all = [];
    for (const r of [...state.panMatches, ...related]) { const k = key(r); if (seen.has(k)) continue; seen.add(k); all.push(r); }

    state.relatedMatches = all;
    els.countRelated.textContent = String(all.length);
    state.pageIndex = 0;
    renderList(all);
  });

  function renderList(list) {
    const filtered = state.quickFilter ? list.filter(row => Object.values(row).some(v => String(v||'').toLowerCase().includes(state.quickFilter.toLowerCase()))) : list;
    state.current = filtered;
    renderTable(paginate(filtered));
    renderPagination(filtered.length, state.pageSize);
  }

  function paginate(list) {
    const start = state.pageIndex * state.pageSize;
    return list.slice(start, start + state.pageSize);
  }

  function renderTable(rows) {
    if (!rows || rows.length === 0) { els.tableContainer.innerHTML = '<div class="panel__content hint">No rows</div>'; return; }
    const cols = state.columns.length ? state.columns : Object.keys(rows[0]);
    let html = '<table><thead><tr>' + cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr></thead><tbody>';
    for (const r of rows) {
      html += '<tr>' + cols.map(c => `<td>${escapeHtml(r[c])}</td>`).join('') + '</tr>';
    }
    html += '</tbody></table>';
    els.tableContainer.innerHTML = html;
  }

  function renderPagination(total, pageSize) {
    const pages = Math.ceil(total / pageSize);
    if (pages <= 1) { els.pagination.innerHTML = ''; return; }
    let html = '';
    for (let i=0;i<pages;i++) {
      html += `<button class="page ${i===state.pageIndex?'active':''}" data-i="${i}">${i+1}</button>`;
    }
    els.pagination.innerHTML = html;
    els.pagination.querySelectorAll('button.page').forEach(btn => btn.addEventListener('click', e => {
      state.pageIndex = parseInt(btn.dataset.i, 10);
      renderList(state.relatedMatches.length ? state.relatedMatches : state.panMatches);
    }));
  }

  function escapeHtml(v) {
    const s = (v==null?'' : String(v));
    return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
  }
})();
