// Browzy start page
// Works both as the extension's new-tab override and as the local page served
// by the backend. In the extension it opens the real browser side panel.

(function () {
  'use strict';

  const clock = document.getElementById('clock');
  const form = document.getElementById('search-form');
  const input = document.getElementById('q');
  const askButton = document.getElementById('ask-browzy');

  function tick() {
    if (!clock) return;
    const now = new Date();
    clock.textContent = now.toLocaleDateString(undefined, {
      weekday: 'short', month: 'short', day: 'numeric'
    }) + ' \u00b7 ' + now.toLocaleTimeString(undefined, {
      hour: '2-digit', minute: '2-digit'
    });
  }

  tick();
  setInterval(tick, 1000 * 15);

  function looksLikeUrl(value) {
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(value)) return true;
    return /^[^\s]+\.[a-z]{2,}(?::\d+)?(?:[/?#].*)?$/i.test(value);
  }

  if (form && input) {
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      const value = input.value.trim();
      if (!value) return;
      if (looksLikeUrl(value)) {
        window.location.href = /^https?:\/\//i.test(value) ? value : 'https://' + value;
      } else {
        window.location.href = 'https://www.google.com/search?q=' + encodeURIComponent(value);
      }
    });
  }

  if (askButton) {
    askButton.addEventListener('click', async function () {
      try {
        if (typeof chrome !== 'undefined' && chrome.sidePanel && chrome.windows) {
          const win = await chrome.windows.getCurrent();
          await chrome.sidePanel.open({ windowId: win.id });
          return;
        }
        // Local HTTP fallback (for debugging outside the extension).
        window.open('http://localhost:8764/static/sidepanel.html', '_blank', 'noopener');
      } catch (error) {
        console.error('[Browzy] Could not open the side panel:', error);
      }
    });
  }
})();
