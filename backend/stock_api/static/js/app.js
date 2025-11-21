const state = { period: '1d', chart: null, lastClose: null, loading: false };


function fmt(v) { return Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 }) }
function setStatus(msg) { document.getElementById('chartStatus').textContent = msg || '' }

function getOrCreateChart() {
  const canvas = document.getElementById('priceChart');
  // If a chart is already attached to this canvas, reuse it
  const existing = Chart.getChart(canvas);
  if (existing) return existing;
  const ctx = canvas.getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [], datasets: [
        { label: 'Open Price', data: [], borderColor: '#1e90ff', borderWidth: 2, pointRadius: 0, tension: 0.25, fill: false },
        { label: 'Close Price', data: [], borderColor: '#000000', borderWidth: 2, pointRadius: 0, tension: 0.25, fill: false }
      ]
    },
    options: {
      animation: { duration: 600, easing: 'easeOutQuart' },
      responsive: true,
      scales: {
        x: { type: 'time', time: { unit: 'day', tooltipFormat: 'MMM dd, yyyy' }, ticks: { color: '#555', autoSkip: true, maxRotation: 0 }, grid: { color: '#ececec' } },
        y: { ticks: { color: '#555', callback: v => '₹' + Number(v).toLocaleString() }, grid: { color: '#f0f0f0' } }
      },
      plugins: { legend: { labels: { color: '#333' } }, tooltip: { mode: 'index', intersect: false, callbacks: { label: ctx => `${ctx.dataset.label}: ₹${fmt(ctx.parsed.y)}` } } }
    }
  });
}

function renderChart(rows) {
  if (!rows || !rows.length) { setStatus('No chart data found'); return; }
  const labels = rows.map(r => new Date(r.Date));
  const openValues = rows.map(r => r.Open);
  const closeValues = rows.map(r => r.Close);

  // Always get or create without forcing a new instance
  state.chart = getOrCreateChart();
  state.chart.data.labels = labels;
  state.chart.data.datasets[0].data = openValues;
  state.chart.data.datasets[1].data = closeValues;
  state.chart.update();

  state.lastClose = closeValues.at(-1);
  setStatus('');
}

async function loadChart() {
  if (state.loading) return; // prevent overlap
  state.loading = true;
  setStatus('Loading...');
  try {
    const res = await fetch(`/api/chart/${state.period}/`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Chart API failed');
    const data = await res.json();
    renderChart(data);
  } catch (e) { console.error(e); setStatus('Error loading chart'); }
  finally { state.loading = false; setStatus(''); }

  setStatus('Loading...');
  try {
    const res = await fetch(`/api/chart/${state.period}/`, { cache: 'no-store' });
    const data = await res.json();
    renderChart(data);
  } catch (e) { setStatus('Error loading chart') }
  finally { setStatus('') }
}

async function loadPrediction() {
  setStatus('Predicting...');
  try {
    const res = await fetch('/api/predict/');
    const data = await res.json();
    const open = document.getElementById('predictedOpen');
    const close = document.getElementById('predictedClose');
    open.style.transform = 'scale(1.1)'; close.style.transform = 'scale(1.1)';
    open.textContent = '₹' + fmt(data.predicted_open || 0);
    close.textContent = '₹' + fmt(data.predicted_close || 0);
    setTimeout(() => { open.style.transform = 'scale(1)'; close.style.transform = 'scale(1)'; }, 400);
  } catch (e) { setStatus('Prediction failed') }
  finally { setStatus('') }
}

async function loadEvaluation() {
  try {
    const res = await fetch('/api/evaluate/');
    const data = await res.json();
    const open = (data.r2_score_open * 100).toFixed(2);
    const close = (data.r2_score_close * 100).toFixed(2);
    const el = document.getElementById('accuracy');
    el.style.opacity = '0';
    setTimeout(() => { el.innerHTML = `<b> Open: ${open}%, Close: ${close}% </b>`; el.style.opacity = '1'; }, 200);
  } catch { }
}

async function refreshAll() {
  await loadChart();
  await loadPrediction();
  await loadEvaluation();
}

document.getElementById('periodBar').addEventListener('click', e => {
  const btn = e.target.closest('button[data-period]');
  if (!btn) return;
  document.querySelectorAll('#periodBar button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  state.period = btn.dataset.period;
  refreshAll();
});

document.getElementById('btnPredict').addEventListener('click', loadPrediction);
window.addEventListener('DOMContentLoaded', refreshAll);