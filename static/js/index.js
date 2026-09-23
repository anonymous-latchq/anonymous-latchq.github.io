/* Local, dependency-free interactions. No analytics or external requests. */
(() => {
  'use strict';
  const video = document.getElementById('overview-video');
  const chapters = Array.from(document.querySelectorAll('[data-time]'));
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let manuallyPaused = false;
  let observerPaused = false;
  let onScreen = false;
  let pendingSeek = null;
  const play = () => { video.play().catch(() => { /* Native controls remain available. */ }); };
  const seek = time => {
    if (video.readyState === 0) { pendingSeek = time; video.load(); return; }
    video.currentTime = time;
    pendingSeek = null;
    manuallyPaused = false;
    play();
  };
  chapters.forEach(button => button.addEventListener('click', () => seek(Number(button.dataset.time))));
  video.addEventListener('loadedmetadata', () => { if (pendingSeek !== null) seek(pendingSeek); });
  video.addEventListener('timeupdate', () => {
    const chapter = chapters.reduce((index, button, i) => video.currentTime >= Number(button.dataset.time) ? i : index, 0);
    chapters.forEach((button, i) => button.setAttribute('aria-pressed', String(i === chapter)));
  });
  video.addEventListener('pause', () => { if (!observerPaused && !video.ended) manuallyPaused = true; });
  video.addEventListener('play', () => { manuallyPaused = false; observerPaused = false; });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      onScreen = entries[0].isIntersecting && entries[0].intersectionRatio >= 0.2;
      if (onScreen && !reducedMotion.matches && !manuallyPaused && !document.hidden) play();
      else if (!onScreen && !video.paused) { observerPaused = true; video.pause(); }
    }, { threshold: 0.2 }).observe(video);
  }
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && !video.paused) { observerPaused = true; video.pause(); }
    else if (!document.hidden && onScreen && !manuallyPaused && !reducedMotion.matches) play();
  });
  reducedMotion.addEventListener('change', e => { if (e.matches) { observerPaused = true; video.pause(); } });

  // Paper Table 1; fixed order is shared with the accessible HTML table.
  const methods = ['FP16', 'KIVI-2', 'KVQuant', 'SQuat', 'TurboQuant', 'LatchQ'];
  const models = [
    { name: 'Llama-3.1-8B-Instruct', rates: { '2.25': [53.74, 52.23, 50.11, 51.55, 49.03, 53.03], '3.0': [53.74, 53.00, 52.64, 52.61, 52.23, 53.51] } },
    { name: 'Qwen2.5-7B-Instruct', rates: { '2.25': [54.40, 52.08, 49.93, 42.23, 41.69, 52.88], '3.0': [54.40, 53.34, 52.63, 47.68, 52.03, 53.75] } }
  ];
  const chartRegion = document.getElementById('chart-region');
  const rateButtons = document.querySelectorAll('[data-rate]');
  let activeRate = '2.25';
  function renderCharts() {
    const minimum = 35;
    const maximum = 55;
    const ticks = [35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55];
    const position = value => (value - minimum) / (maximum - minimum) * 100;
    chartRegion.replaceChildren(...models.map(model => {
      const panel = document.createElement('article');
      panel.className = 'chart-panel';
      const title = document.createElement('h3'); title.textContent = model.name;
      const subtitle = document.createElement('p'); subtitle.textContent = `Target: ${activeRate} bits per dimension · 8 tasks`;
      panel.append(title, subtitle);
      const values = model.rates[activeRate];
      // Keep every method visible at both bit rates.
      methods.forEach((_, i) => {
        const value = values[i];
        const row = document.createElement('div'); row.className = 'chart-row'; row.dataset.method = methods[i];
        const label = document.createElement('span'); label.className = 'bar-label'; label.textContent = methods[i] + ([2, 4].includes(i) ? '*' : '');
        const track = document.createElement('div'); track.className = 'chart-track'; track.setAttribute('aria-hidden', 'true');
        track.classList.add('zoom-track');
        ticks.forEach(tick => {
          const gridline = document.createElement('span'); gridline.className = 'zoom-gridline';
          gridline.style.left = `${position(tick)}%`; track.append(gridline);
        });
        const fill = document.createElement('div'); fill.className = 'bar-fill';
        fill.style.width = `${position(value)}%`; track.append(fill);
        const score = document.createElement('span'); score.className = 'bar-value'; score.textContent = value.toFixed(2);
        row.append(label, track, score); panel.append(row);
      });
      const axis = document.createElement('div'); axis.className = 'chart-axis'; axis.setAttribute('aria-hidden', 'true');
      axis.classList.add('zoom-axis');
      ticks.forEach(value => {
        const tick = document.createElement('span'); tick.textContent = value;
        tick.style.left = `${position(value)}%`;
        axis.append(tick);
      });
      panel.append(axis);
      return panel;
    }));
  }
  rateButtons.forEach(button => button.addEventListener('click', () => {
    rateButtons.forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    activeRate = button.dataset.rate;
    renderCharts();
  }));
  renderCharts();
})();
