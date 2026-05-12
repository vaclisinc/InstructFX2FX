(function () {
  const slides    = document.querySelectorAll('.slide');
  const counter   = document.querySelector('.slide-counter');
  const container = document.querySelector('.slide-container');
  const total     = slides.length;
  let current     = 0;

  function readHash() {
    const m = location.hash.match(/^#slide-(\d+)$/);
    if (m) {
      const n = parseInt(m[1], 10) - 1;
      if (n >= 0 && n < total) return n;
    }
    return 0;
  }

  function show(idx) {
    slides[current].classList.remove('active');
    current = Math.max(0, Math.min(total - 1, idx));
    slides[current].classList.add('active');
    counter.textContent = (current + 1) + ' / ' + total;
    history.replaceState(null, '', '#slide-' + (current + 1));
  }

  function next() { show(current + 1); }
  function prev() { show(current - 1); }

  // Keyboard navigation
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
      e.preventDefault(); next();
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault(); prev();
    } else if (e.key === 'f' || e.key === 'F') {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen();
      else document.exitFullscreen();
    }
  });

  // Click navigation: left third = prev, rest = next
  container.addEventListener('click', function (e) {
    if (e.target.closest('a, button')) return;
    const rect = this.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x < rect.width * 0.33) prev();
    else next();
  });

  // Responsive letterbox scaling
  function scaleSlides() {
    const sw = 1000, sh = 562.5;
    const scale = Math.min(window.innerWidth / sw, window.innerHeight / sh);
    const left = Math.round((window.innerWidth  - sw * scale) / 2);
    const top  = Math.round((window.innerHeight - sh * scale) / 2);
    container.style.transformOrigin = 'top left';
    container.style.transform = 'scale(' + scale + ')';
    container.style.left = left + 'px';
    container.style.top  = top  + 'px';
  }
  window.addEventListener('resize', scaleSlides);
  scaleSlides();

  // Init from URL hash
  show(readHash());
  window.addEventListener('hashchange', function () { show(readHash()); });
})();
