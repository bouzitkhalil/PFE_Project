/* ═══════════════════════════════════════════════════════════════
   Analytics page – 5 Chart.js charts + KPI polling
   ═══════════════════════════════════════════════════════════════ */

Chart.defaults.color = '#64748b';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size   = 12;

const INDIGO  = '#6366f1';
const TEAL    = '#14b8a6';
const EMERALD = '#10b981';
const AMBER   = '#f59e0b';
const ROSE    = '#f43f5e';
const BLUE    = '#3b82f6';
const VIOLET  = '#8b5cf6';
const PINK    = '#ec4899';

const PALETTE = [INDIGO, TEAL, EMERALD, AMBER, ROSE, BLUE, VIOLET, PINK];

let trendChart, deptChart, hourlyAChart, topEmpChart, weeklyChart;

// ── Trend chart (14-day line) ─────────────────────────────────────
function buildTrend(data) {
  const ctx = document.getElementById('trendChart');
  if (!ctx) return;
  if (trendChart) trendChart.destroy();

  const labels = data.map(d => {
    const dt = new Date(d.day);
    return dt.toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' });
  });

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Événements',
        data: data.map(d => d.count),
        borderColor: TEAL,
        borderWidth: 2.5,
        pointBackgroundColor: TEAL,
        pointRadius: 4,
        pointHoverRadius: 7,
        tension: 0.4,
        fill: true,
        backgroundColor: (c) => {
          const g = c.chart.ctx.createLinearGradient(0, 0, 0, 200);
          g.addColorStop(0, TEAL + '44');
          g.addColorStop(1, TEAL + '00');
          return g;
        },
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: '#1a2540', borderColor: '#334155', borderWidth: 1 } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { maxRotation: 30 } },
        y: { grid: { color: 'rgba(255,255,255,.04)' }, beginAtZero: true, ticks: { precision: 0 } }
      }
    }
  });
}

// ── Department donut ──────────────────────────────────────────────
function buildDept(data) {
  const ctx = document.getElementById('deptChart');
  if (!ctx) return;
  if (deptChart) deptChart.destroy();

  deptChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: data.map(d => d.department),
      datasets: [{
        data: data.map(d => d.count),
        backgroundColor: PALETTE.map(c => c + 'cc'),
        borderColor: PALETTE,
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      cutout: '65%',
      plugins: {
        legend: { position: 'right', labels: { padding: 14, usePointStyle: true, pointStyleWidth: 8 } },
        tooltip: { backgroundColor: '#1a2540', borderColor: '#334155', borderWidth: 1 }
      }
    }
  });
}

// ── Hourly bar (analytics page) ────────────────────────────────────
function buildHourlyA(data) {
  const ctx = document.getElementById('hourlyA');
  if (!ctx) return;
  if (hourlyAChart) hourlyAChart.destroy();

  const hours  = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, '0')}h`);
  const entries = new Array(24).fill(0);
  const exits   = new Array(24).fill(0);
  data.forEach(d => {
    const h = parseInt(d.hour);
    if (d.event_type === 'entry') entries[h] = d.count;
    else exits[h] = d.count;
  });

  hourlyAChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: hours,
      datasets: [
        { label: 'Entrées', data: entries, backgroundColor: EMERALD + '99', borderColor: EMERALD, borderWidth: 1.5, borderRadius: 4 },
        { label: 'Sorties', data: exits,   backgroundColor: BLUE + '99',    borderColor: BLUE,    borderWidth: 1.5, borderRadius: 4 },
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top', labels: { usePointStyle: true, pointStyleWidth: 8, padding: 16 } },
        tooltip: { backgroundColor: '#1a2540', borderColor: '#334155', borderWidth: 1 }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.03)' } },
        y: { grid: { color: 'rgba(255,255,255,.04)' }, beginAtZero: true, ticks: { precision: 0 } }
      }
    }
  });
}

// ── Top employees horizontal bar ──────────────────────────────────
function buildTopEmp(data) {
  const ctx = document.getElementById('topEmpChart');
  if (!ctx) return;
  if (topEmpChart) topEmpChart.destroy();

  topEmpChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.name),
      datasets: [{
        label: 'Entrées totales',
        data: data.map(d => d.total_events),
        backgroundColor: PALETTE.slice(0, data.length).map(c => c + 'bb'),
        borderColor:     PALETTE.slice(0, data.length),
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#1a2540', borderColor: '#334155', borderWidth: 1 }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.04)' }, beginAtZero: true, ticks: { precision: 0 } },
        y: { grid: { display: false } }
      }
    }
  });
}

// ── Weekly line chart ─────────────────────────────────────────────
function buildWeekly(data) {
  const ctx = document.getElementById('weeklyChart');
  if (!ctx) return;
  if (weeklyChart) weeklyChart.destroy();

  const labels = data.map(d => {
    const dt = new Date(d.day);
    return dt.toLocaleDateString('fr-FR', { weekday: 'long', day: 'numeric', month: 'short' });
  });

  weeklyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Employés présents',
        data: data.map(d => d.unique_emp),
        borderColor: VIOLET,
        borderWidth: 2.5,
        pointBackgroundColor: VIOLET,
        pointRadius: 5,
        pointHoverRadius: 8,
        tension: 0.4,
        fill: true,
        backgroundColor: (c) => {
          const g = c.chart.ctx.createLinearGradient(0, 0, 0, 160);
          g.addColorStop(0, VIOLET + '44');
          g.addColorStop(1, VIOLET + '00');
          return g;
        }
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: '#1a2540', borderColor: '#334155', borderWidth: 1 } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.03)' } },
        y: { grid: { color: 'rgba(255,255,255,.04)' }, beginAtZero: true, ticks: { precision: 0 } }
      }
    }
  });
}

// ── KPI helpers ───────────────────────────────────────────────────
function set(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Load stats ────────────────────────────────────────────────────
async function loadStats() {
  try {
    const r = await fetch('/api/stats');
    if (!r.ok) return;
    const data = await r.json();

    set('a-present', data.presence.present);
    set('a-entries', data.today.entry);
    set('a-exits',   data.today.exit);
    set('a-occ',     data.presence.occupancy + '%');

    const now = new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    set('last-refresh', `Mise à jour : ${now}`);

    buildTrend(data.daily);
    buildDept(data.departments);
    buildHourlyA(data.hourly);
    buildTopEmp(data.top_emp);
    buildWeekly(data.weekly);
  } catch(e) { console.error('Analytics load failed', e); }
}

loadStats();
setInterval(loadStats, 30000);
