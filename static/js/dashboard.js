/* ═══════════════════════════════════════════════════════════════
   Dashboard charts + live KPI polling
   ═══════════════════════════════════════════════════════════════ */

// ── Chart.js global defaults ──────────────────────────────────────
Chart.defaults.color = '#64748b';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size   = 12;

const INDIGO = '#6366f1';
const TEAL   = '#14b8a6';
const EMERALD= '#10b981';
const AMBER  = '#f59e0b';
const ROSE   = '#f43f5e';
const BLUE   = '#3b82f6';

// Gradient helper
function mkGradient(ctx, color, alpha1=0.35, alpha2=0.01) {
  const g = ctx.createLinearGradient(0, 0, 0, 300);
  g.addColorStop(0, color.replace(')', `, ${alpha1})`).replace('rgb', 'rgba').replace('#', 'rgba(').replace(/rgba\(([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i, (_, r,g2,b) => `rgba(${parseInt(r,16)},${parseInt(g2,16)},${parseInt(b,16)}`));
  g.addColorStop(1, `rgba(255,255,255,0)`);
  return g;
}

// Simpler gradient using hex
function hexGrad(ctx, hex, h=250) {
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, hex + '55');
  g.addColorStop(1, hex + '00');
  return g;
}

// ── Daily chart ───────────────────────────────────────────────────
let dailyChart, occupancyChart, hourlyChart;

function buildDailyChart(data) {
  const ctx = document.getElementById('dailyChart');
  if (!ctx) return;
  if (dailyChart) dailyChart.destroy();

  const labels = data.map(d => {
    const dt = new Date(d.day);
    return dt.toLocaleDateString('fr-FR', { weekday: 'short', day: 'numeric', month: 'short' });
  });
  const counts = data.map(d => d.count);

  dailyChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Événements',
        data: counts,
        borderColor: INDIGO,
        borderWidth: 2.5,
        pointBackgroundColor: INDIGO,
        pointRadius: 4,
        pointHoverRadius: 7,
        tension: 0.4,
        fill: true,
        backgroundColor: (context) => hexGrad(context.chart.ctx, INDIGO, 200),
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

function buildOccupancyChart(present, total) {
  const ctx = document.getElementById('occupancyChart');
  if (!ctx) return;
  if (occupancyChart) occupancyChart.destroy();
  const absent = Math.max(total - present, 0);

  occupancyChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Présents', 'Absents'],
      datasets: [{
        data: [present, absent],
        backgroundColor: [EMERALD + 'cc', '#1e2d45'],
        borderColor: [EMERALD, '#334155'],
        borderWidth: 2,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true,
      cutout: '72%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 14, usePointStyle: true, pointStyleWidth: 8 } },
        tooltip: { backgroundColor: '#1a2540', borderColor: '#334155', borderWidth: 1 }
      }
    }
  });
}

function buildHourlyChart(data) {
  const ctx = document.getElementById('hourlyChart');
  if (!ctx) return;
  if (hourlyChart) hourlyChart.destroy();

  const hours  = Array.from({length: 24}, (_, i) => `${String(i).padStart(2,'0')}h`);
  const entries= new Array(24).fill(0);
  const exits  = new Array(24).fill(0);
  data.forEach(d => {
    const h = parseInt(d.hour);
    if (d.event_type === 'entry') entries[h] = d.count;
    else exits[h] = d.count;
  });

  hourlyChart = new Chart(ctx, {
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
      plugins: { legend: { position: 'top', labels: { usePointStyle: true, pointStyleWidth: 8, padding: 16 } },
                 tooltip: { backgroundColor: '#1a2540', borderColor: '#334155', borderWidth: 1 } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.03)' }, stacked: false },
        y: { grid: { color: 'rgba(255,255,255,.04)' }, beginAtZero: true, ticks: { precision: 0 } }
      }
    }
  });
}

// ── KPI DOM helpers ───────────────────────────────────────────────
function animateCount(el, to) {
  if (!el) return;
  const from = parseInt(el.textContent) || 0;
  const step = Math.ceil(Math.abs(to - from) / 15) || 1;
  let curr = from;
  const dir = to > from ? 1 : -1;
  const timer = setInterval(() => {
    curr += dir * step;
    if ((dir > 0 && curr >= to) || (dir < 0 && curr <= to)) { curr = to; clearInterval(timer); }
    el.textContent = curr;
  }, 30);
}

// ── Alert resolve ─────────────────────────────────────────────────
async function resolveAlert(id, btn) {
  try {
    await fetch(`/api/alerts/${id}/resolve`, { method: 'POST' });
    const row = btn.closest('.alert-item');
    row.style.opacity = '0';
    row.style.transform = 'translateX(20px)';
    row.style.transition = '.3s';
    setTimeout(() => row.remove(), 300);
    const cnt = document.getElementById('alert-count');
    if (cnt) cnt.textContent = Math.max(0, parseInt(cnt.textContent) - 1);
  } catch(e) { console.error(e); }
}

// ── Main load / refresh ───────────────────────────────────────────
async function loadDashboard() {
  try {
    const r = await fetch('/api/stats');
    if (!r.ok) return;
    const data = await r.json();

    // KPIs
    animateCount(document.getElementById('kpi-present'), data.presence.present);
    animateCount(document.getElementById('kpi-absent'),  data.presence.absent);
    animateCount(document.getElementById('kpi-entries'), data.today.entry);
    animateCount(document.getElementById('kpi-exits'),   data.today.exit);

    const occ = data.presence.occupancy;
    const occEl = document.getElementById('kpi-occ');
    if (occEl) occEl.textContent = occ;
    const bar = document.getElementById('occ-bar');
    if (bar) bar.style.width = occ + '%';

    // Charts
    buildDailyChart(data.daily);
    buildOccupancyChart(data.presence.present, data.presence.total);
    buildHourlyChart(data.hourly);

  } catch(e) { console.error('Dashboard refresh failed', e); }
}

loadDashboard();
setInterval(loadDashboard, 15000);
