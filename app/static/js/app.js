// =======================
// ===== Splash logic =====
// =======================
(function () {
  const splash = document.getElementById("splash");
  const appElements = document.querySelectorAll(".app");
  const body = document.body;

  const SPLASH_MS = 2400; // 2.4 seconds

  const reveal = () => {
    if (splash) splash.classList.add("fade-splash");
    body.classList.remove("preload");
    appElements.forEach(el => el.classList.add("revealed"));
    setTimeout(() => splash && splash.remove(), 700);
  };

  const timer = setTimeout(reveal, SPLASH_MS);

  window.addEventListener("load", () => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (mq.matches) {
      clearTimeout(timer);
      reveal();
    }
  });
})();

// ========================
// ===== Tips Modal =======
// ========================
(function () {
  const modal = document.getElementById("tipsModal");

  function open() {
    if (!modal) return;
    modal.classList.add("show");
    modal.setAttribute("aria-hidden", "false");
  }

  function close() {
    if (!modal) return;
    modal.classList.remove("show");
    modal.setAttribute("aria-hidden", "true");
  }

  // Expose globally
  window.openTipsModal = open;
  window.closeTipsModal = close;

  // Close on backdrop click
  document.addEventListener("click", (e) => {
    if (modal && modal.classList.contains("show") && e.target === modal) {
      close();
    }
  });

  // Close on ESC key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal && modal.classList.contains("show")) {
      close();
    }
  });
})();

// =================================
// ===== Scroll Reveal Animations ===
// =================================
(function () {
  const items = document.querySelectorAll(".reveal");
  if (!items.length) return;

  if (!("IntersectionObserver" in window)) {
    items.forEach(el => el.classList.add("in"));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        if (el.dataset.delay) {
          el.style.transitionDelay = `${el.dataset.delay}ms`;
        }
        el.classList.add("in");
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.2 });

  items.forEach(el => observer.observe(el));
})();
