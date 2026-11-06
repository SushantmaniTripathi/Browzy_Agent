// Browzy Side Panel - Main JavaScript
// Handles WebSocket connection, chat, history, theme and panel resizing.
// Runs both as an extension page (inside the browser) and as a standalone page
// served by the local backend (http://localhost:8764/static/sidepanel.html).

(function () {
  'use strict';

  // ==========================================================================
  // STATE
  // ==========================================================================
  const state = {
    ws: null,
    connected: false,
    reconnectAttempts: 0,
    maxReconnectAttempts: 10,
    reconnectDelay: 1000,
    heartbeatTimer: null,
    activeTab: 'ask',
    theme: 'dark',
    messageId: 0,
    sessionId: null,
    model: null,
    streaming: false,
    actionBubbleEl: null
  };

  const els = {};
  const STORAGE_KEYS = {
    theme: 'browzy-theme',
    session: 'browzy-session-id'
  };

  // ==========================================================================
  // STORAGE (chrome.storage when embedded, localStorage fallback)
  // ==========================================================================
  const store = {
    async get(key, fallback) {
      try {
        if (window.chrome && chrome.storage && chrome.storage.local) {
          const result = await chrome.storage.local.get(key);
          return result && result[key] !== undefined ? result[key] : fallback;
        }
        const raw = localStorage.getItem(key);
        return raw === null ? fallback : JSON.parse(raw);
      } catch (e) {
        console.warn('[Browzy] Storage read failed:', e);
        return fallback;
      }
    },
    async set(key, value) {
      try {
        if (window.chrome && chrome.storage && chrome.storage.local) {
          await chrome.storage.local.set({ [key]: value });
          return;
        }
        localStorage.setItem(key, JSON.stringify(value));
      } catch (e) {
        console.warn('[Browzy] Storage write failed:', e);
      }
    }
  };

  function newSessionId() {
    try {
      if (window.crypto && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
      }
    } catch (_) { /* ignore */ }
    return 'session-' + Date.now() + '-' + Math.random().toString(36).slice(2, 10);
  }

  // ==========================================================================
  // INITIALIZATION
  // ==========================================================================
  async function init() {
    cacheElements();
    await loadPreferences();

    setupTabs();
    setupThemeToggle();
    setupChatInput();
    setupVoice();
    setupQuickActions();
    setupCloseButton();
    setupHistory();
    setupNewChat();
    setupDataVault();

    updateConnectionStatus('connecting');
    connectWebSocket();
    loadVault();
  }

  function cacheElements() {
    els.panel = document.getElementById('browzy-panel');
    els.themeToggle = document.getElementById('theme-toggle');
    els.closePanel = document.getElementById('close-panel');
    els.tabAsk = document.getElementById('tab-ask');
    els.tabData = document.getElementById('tab-data');
    els.panelAsk = document.getElementById('panel-ask');
    els.panelData = document.getElementById('panel-data');
    els.connectionStatus = document.getElementById('connection-status');
    els.statusDot = els.connectionStatus?.querySelector('.status-dot');
    els.statusText = els.connectionStatus?.querySelector('.status-text');
    els.chatMessages = document.getElementById('chat-messages');
    els.chatInput = document.getElementById('chat-input');
    els.sendBtn = document.getElementById('send-btn');
    els.micBtn = document.getElementById('mic-btn');
    els.voiceStatus = document.getElementById('voice-status');
    els.historySection = document.getElementById('history-section');
    els.historyToggle = document.getElementById('history-toggle');
    els.historyList = document.getElementById('history-list');
    els.newChat = document.getElementById('new-chat');

    // Your Data tab
    els.vaultDropzone = document.getElementById('vault-dropzone');
    els.vaultFile = document.getElementById('vault-file');
    els.vaultStatus = document.getElementById('vault-status');
    els.vaultSources = document.getElementById('vault-sources');
    els.vaultCount = document.getElementById('vault-count');
    els.vaultFields = document.getElementById('vault-fields');
    els.formState = document.getElementById('form-state');
    els.fillPageBtn = document.getElementById('fill-page-btn');
    els.fillHint = document.getElementById('fill-hint');
    els.clearVault = document.getElementById('clear-vault');
  }

  async function loadPreferences() {
    const theme = await store.get(STORAGE_KEYS.theme, null);
    if (theme === 'dark' || theme === 'light') {
      state.theme = theme;
    } else {
      // Browzy ships dark by default; the toggle still offers a light variant.
      state.theme = 'dark';
    }
    applyTheme();

    const session = await store.get(STORAGE_KEYS.session, null);
    state.sessionId = typeof session === 'string' && session ? session : newSessionId();
    if (!session) await store.set(STORAGE_KEYS.session, state.sessionId);
  }

  // ==========================================================================
  // WEBSOCKET CONNECTION
  // ==========================================================================
  function connectWebSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    let ws;
    try {
      ws = new WebSocket('ws://localhost:8764/ws');
    } catch (e) {
      console.error('[Browzy] Could not create WebSocket:', e);
      scheduleReconnect();
      return;
    }
    state.ws = ws;

    ws.onopen = () => {
      console.log('[Browzy] WebSocket connected');
      state.connected = true;
      state.reconnectAttempts = 0;
      updateConnectionStatus('connected');
      startHeartbeat();

      // Restore this conversation and refresh the history list.
      if (state.sessionId) {
        sendMessage({ type: 'load_session', session_id: state.sessionId });
      }
      loadHistory();
      sendMessage({ type: 'get_models' });
    };

    ws.onmessage = (event) => {
      try {
        handleWebSocketMessage(JSON.parse(event.data));
      } catch (e) {
        console.error('[Browzy] Failed to parse message:', e);
      }
    };

    ws.onclose = () => {
      console.log('[Browzy] WebSocket disconnected');
      state.connected = false;
      state.streaming = false;
      stopHeartbeat();
      updateConnectionStatus('disconnected');
      scheduleReconnect();
    };

    ws.onerror = (error) => {
      console.error('[Browzy] WebSocket error:', error);
    };
  }

  function startHeartbeat() {
    stopHeartbeat();
    state.heartbeatTimer = setInterval(() => {
      if (state.connected) sendMessage({ type: 'ping' });
    }, 25000);
  }

  function stopHeartbeat() {
    if (state.heartbeatTimer) {
      clearInterval(state.heartbeatTimer);
      state.heartbeatTimer = null;
    }
  }

  function scheduleReconnect() {
    if (state.reconnectAttempts >= state.maxReconnectAttempts) {
      updateConnectionStatus('disconnected', 'Connection lost');
      return;
    }

    state.reconnectAttempts++;
    const delay = state.reconnectDelay * Math.min(state.reconnectAttempts, 5);
    updateConnectionStatus('connecting', `Reconnecting... (${state.reconnectAttempts})`);
    setTimeout(connectWebSocket, delay);
  }

  function sendMessage(message) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(message));
      return true;
    }
    console.warn('[Browzy] WebSocket not connected, dropping message:', message.type);
    return false;
  }

  function handleWebSocketMessage(message) {
    switch (message.type) {
      case 'connected':
        break;
      case 'pong':
        break;
      case 'assistant_start':
        finalizeStreamingMessage();
        state.actionBubbleEl = null;
        setStreaming(true);
        break;
      case 'token':
        appendToken(message.data);
        break;
      case 'status':
        setActivity(message.data);
        break;
      case 'task_state':
        if (message.data?.state === 'queued') setActivity('queued');
        else if (message.data?.state === 'running') setActivity('working');
        break;
      case 'action':
        setActivity('working');
        addActionMessage(message.data);
        break;
      case 'done':
        finalizeStreamingMessage();
        state.actionBubbleEl = null;
        setStreaming(false);
        loadHistory();
        break;
      case 'models':
        handleModels(message.data, message.selected);
        break;
      case 'ollama_error':
        discardEmptyStreamingMessage();
        state.actionBubbleEl = null;
        setStreaming(false);
        addSystemMessage(message.message || 'The local AI model is unavailable.', 'error');
        break;
      case 'log':
        addLogMessage(message.text || message.data || '');
        break;
      case 'error':
        finalizeStreamingMessage();
        state.actionBubbleEl = null;
        setStreaming(false);
        addSystemMessage(message.message || 'Something went wrong.', 'error');
        break;
      case 'history':
        renderHistory(message.data);
        break;
      case 'session_loaded':
        loadSession(message.data);
        break;
      case 'session_deleted':
        if (state.sessionId === message.session_id) {
          startNewChat();
        }
        loadHistory();
        break;
      default:
        console.log('[Browzy] Unknown message type:', message.type);
    }
  }

  function handleModels(models, selected) {
    if (!Array.isArray(models) || models.length === 0) {
      state.model = null;
      return;
    }
    const names = models.map((model) => (typeof model === 'string' ? model : model && model.name)).filter(Boolean);
    const chosen = selected && names.includes(selected) ? selected : names[0];
    state.model = chosen || null;
    if (state.model) {
      els.panel?.setAttribute('data-model', state.model);
    }
  }

  function setStreaming(streaming) {
    state.streaming = streaming;
    if (els.sendBtn) {
      els.sendBtn.disabled = !state.connected || streaming;
    }
    if (!streaming) {
      setActivity(null);
    }
  }

  function setActivity(activity) {
    if (!els.connectionStatus || !state.connected) return;

    const labels = {
      thinking: 'Thinking...',
      answering: 'Answering...',
      working: 'Working...',
      queued: 'Waiting for browser...'
    };

    if (labels[activity]) {
      els.connectionStatus.className = 'connection-status connected busy';
      if (els.statusText) els.statusText.textContent = labels[activity];
      return;
    }

    updateConnectionStatus('connected');
  }

  // ==========================================================================
  // CONNECTION STATUS UI
  // ==========================================================================
  function updateConnectionStatus(status, customText = null) {
    if (!els.connectionStatus) return;

    els.connectionStatus.className = `connection-status ${status}`;

    const texts = {
      connected: 'Connected',
      disconnected: 'Disconnected',
      connecting: 'Connecting...'
    };

    if (els.statusText) {
      els.statusText.textContent = customText || texts[status] || status;
    }

    if (els.sendBtn) {
      els.sendBtn.disabled = status !== 'connected' || state.streaming;
    }
  }

  // ==========================================================================
  // CHAT MESSAGES
  // ==========================================================================
  function addSystemMessage(text, type = 'system') {
    const msgEl = createMessageElement({ role: 'system', content: text, type });
    els.chatMessages.appendChild(msgEl);
    scrollToBottom();
  }

  function addUserMessage(text) {
    const msgEl = createMessageElement({ role: 'user', content: text });
    els.chatMessages.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
  }

  function addAssistantMessage() {
    const msgEl = createMessageElement({ role: 'assistant', content: '', streaming: true });
    els.chatMessages.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
  }

  function createMessageElement({ role, content, type = 'normal', streaming = false }) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;
    wrapper.dataset.messageId = ++state.messageId;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : role === 'assistant' ? 'B' : '\u2726';

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';

    if (type === 'error') {
      contentEl.style.borderColor = 'var(--color-danger)';
    }

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    if (streaming) {
      bubble.dataset.streaming = 'true';
    }
    bubble.textContent = content;

    contentEl.appendChild(bubble);
    wrapper.appendChild(avatar);
    wrapper.appendChild(contentEl);

    return wrapper;
  }

  function appendToken(token) {
    let streamingEl = els.chatMessages.querySelector('.message-bubble[data-streaming="true"]');

    if (!streamingEl) {
      const msgEl = addAssistantMessage();
      streamingEl = msgEl.querySelector('.message-bubble');
    }

    streamingEl.textContent += token;
    scrollToBottom();
  }

  function finalizeStreamingMessage() {
    const streamingEl = els.chatMessages.querySelector('.message-bubble[data-streaming="true"]');
    if (streamingEl) {
      delete streamingEl.dataset.streaming;
      renderRichText(streamingEl);
      addDocumentLink(streamingEl);
    }
  }

  function renderRichText(bubble) {
    const text = bubble.textContent || '';
    const pattern = /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s<]+)/gi;
    const fragment = document.createDocumentFragment();
    let cursor = 0;
    let match;

    while ((match = pattern.exec(text))) {
      if (match.index > cursor) {
        fragment.appendChild(document.createTextNode(text.slice(cursor, match.index)));
      }
      const link = document.createElement('a');
      link.href = match[2] || match[3];
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = match[1] || match[3];
      fragment.appendChild(link);
      cursor = pattern.lastIndex;
    }

    if (cursor < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(cursor)));
    }
    bubble.replaceChildren(fragment);
  }

  function addDocumentLink(bubble) {
    if (!/cheapest|comparison|compare/i.test(bubble.textContent || '')) return;
    if (bubble.querySelectorAll('a[href^="http"]').length < 1) return;
    const content = bubble.textContent || '';
    const contentHtml = bubble.innerHTML;
    const contentEl = bubble.parentElement;
    if (!contentEl || contentEl.querySelector('.document-link')) return;

    const documentLink = document.createElement('a');
    documentLink.className = 'document-link';
    documentLink.target = '_blank';
    documentLink.rel = 'noopener noreferrer';
    documentLink.textContent = 'Open comparison as a document';
    const html = `<!doctype html><meta charset="utf-8"><title>Browzy comparison</title>` +
      `<style>body{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px;white-space:pre-wrap}a{color:#145dcc}</style>` +
      `<h1>Browzy comparison</h1><p>${contentHtml.replace(/\n/g, '<br>')}</p>`;
    documentLink.href = URL.createObjectURL(new Blob([html], { type: 'text/html' }));
    contentEl.appendChild(documentLink);
  }

  function escapeHtml(value) {
    return String(value || '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[char]));
  }

  function discardEmptyStreamingMessage() {
    const streamingEl = els.chatMessages.querySelector('.message-bubble[data-streaming="true"]');
    if (!streamingEl) return;
    const isEmpty = !streamingEl.textContent.trim();
    delete streamingEl.dataset.streaming;
    if (isEmpty) {
      const wrapper = streamingEl.closest('.message');
      if (wrapper) wrapper.remove();
    }
  }

  function addLogMessage(text) {
    const msgEl = createMessageElement({
      role: 'system',
      content: `[LOG] ${text}`,
      type: 'log'
    });
    msgEl.querySelector('.message-content').style.borderColor = 'var(--color-info)';
    els.chatMessages.appendChild(msgEl);
    scrollToBottom();
  }

  function addActionMessage(action) {
    if (!action) return;
    const label = action.label || `Running ${action.tool || 'an action'}`;

    // Group every action from one assistant turn into a single bubble instead
    // of filling the chat with one row per click/scroll.
    let bubble = state.actionBubbleEl;
    if (!bubble || !bubble.isConnected) {
      const msgEl = createMessageElement({ role: 'system', content: '', type: 'action' });
      msgEl.querySelector('.message-avatar').textContent = '\u26a1';
      bubble = msgEl.querySelector('.message-content');
      bubble.style.borderColor = 'var(--color-info)';
      bubble.textContent = '';
      state.actionBubbleEl = bubble;
      els.chatMessages.appendChild(msgEl);
    }

    const line = document.createElement('div');
    line.className = 'action-line';
    line.textContent = label;
    bubble.appendChild(line);
    scrollToBottom();
  }

  function scrollToBottom() {
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  }

  function showWelcome() {
    els.chatMessages.innerHTML = '';
    addSystemMessage('Welcome to Browzy. Ask anything — I can help you understand the page you are on.');
  }

  // ==========================================================================
  // CHAT INPUT
  // ==========================================================================
  function setupChatInput() {
    els.chatInput.addEventListener('input', () => {
      els.chatInput.style.height = 'auto';
      els.chatInput.style.height = Math.min(els.chatInput.scrollHeight, 160) + 'px';
    });

    els.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });

    els.sendBtn.addEventListener('click', sendChatMessage);
  }

  function sendChatMessage() {
    const text = els.chatInput.value.trim();
    if (!text || state.streaming) return;

    if (!state.connected) {
      addSystemMessage('Browzy is not connected to the backend. Start the backend and try again.', 'error');
      return;
    }

    sendPrompt(text);
  }

  function setupQuickActions() {
    document.querySelectorAll('.quick-action').forEach((button) => {
      button.addEventListener('click', () => {
        if (button.dataset.action === 'fill-form') {
          openFillPanel();
          return;
        }
        sendPrompt(button.dataset.prompt || '');
      });
    });
  }

  function openFillPanel() {
    // Form filling is manual by design: switch to Your Data and put the real
    // Fill this form button in front of the user instead of sending a chat
    // command that could trigger automation on the wrong page.
    switchTab('data');
    if (els.fillPageBtn) {
      els.fillPageBtn.scrollIntoView({ block: 'center', behavior: 'smooth' });
      els.fillPageBtn.focus({ preventScroll: true });
    }
  }

  function sendPrompt(text) {
    const value = (text || '').trim();
    if (!value || state.streaming) return;

    if (!state.connected) {
      addSystemMessage('Browzy is not connected to the backend. Start the backend and try again.', 'error');
      return;
    }

    if (!state.sessionId) {
      state.sessionId = newSessionId();
      store.set(STORAGE_KEYS.session, state.sessionId);
    }

    state.actionBubbleEl = null;
    els.chatInput.value = '';
    els.chatInput.style.height = 'auto';
    addUserMessage(value);

    sendMessage({
      type: 'chat',
      text: value,
      session_id: state.sessionId,
      model: state.model || 'default'
    });
  }

  // ==========================================================================
  // VOICE INPUT (Hindi / Marathi / English / code-mixed)
  // ==========================================================================
  const VOICE_MAX_MS = 60000;

  const voiceState = {
    recorder: null,
    chunks: [],
    stream: null,
    startedAt: 0,
    ticker: null,
    maxTimer: null,
    busy: false,
    cancelled: false
  };

  function voiceSupported() {
    return Boolean(
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia &&
      window.MediaRecorder
    );
  }

  function setVoiceStatus(text, kind) {
    const el = els.voiceStatus;
    if (!el) return;
    if (!text) {
      el.hidden = true;
      el.textContent = '';
      el.classList.remove('error', 'listening');
      return;
    }
    el.hidden = false;
    el.textContent = text;
    el.classList.toggle('error', kind === 'error');
    el.classList.toggle('listening', kind === 'listening');
  }

  function pickVoiceMime() {
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus'];
    for (const type of candidates) {
      if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return '';
  }

  function micErrorMessage(error) {
    const name = error && error.name;
    if (name === 'NotAllowedError' || name === 'SecurityError') {
      return 'Microphone is blocked. Allow microphone access for Browzy in Windows Settings → Privacy → Microphone, then try again.';
    }
    if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      return 'No microphone was found. Plug one in and try again.';
    }
    if (name === 'NotReadableError') {
      return 'The microphone is busy in another app. Close it and try again.';
    }
    return (error && error.message) || 'Could not start the microphone.';
  }

  function setupVoice() {
    if (!els.micBtn) return;
    if (!voiceSupported()) {
      els.micBtn.hidden = true;
      return;
    }

    els.micBtn.addEventListener('click', () => {
      if (voiceState.busy) return;
      if (voiceState.recorder) stopVoice();
      else startVoice();
    });

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && voiceState.recorder) {
        event.preventDefault();
        cancelVoice();
      }
    });
  }

  async function startVoice() {
    setVoiceStatus('');
    voiceState.cancelled = false;

    if (!state.connected) {
      setVoiceStatus('Browzy is not connected to the backend, so voice cannot be transcribed.', 'error');
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1
        }
      });
    } catch (error) {
      setVoiceStatus(micErrorMessage(error), 'error');
      return;
    }

    voiceState.stream = stream;
    voiceState.chunks = [];

    const mime = pickVoiceMime();
    let recorder;
    try {
      recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    } catch (error) {
      releaseVoiceStream();
      setVoiceStatus('This browser cannot record audio.', 'error');
      return;
    }

    voiceState.recorder = recorder;
    recorder.addEventListener('dataavailable', (event) => {
      if (event.data && event.data.size > 0) voiceState.chunks.push(event.data);
    });
    recorder.addEventListener('stop', handleRecorderStop);

    recorder.start(250);
    voiceState.startedAt = Date.now();

    els.micBtn.classList.add('recording');
    els.micBtn.setAttribute('aria-pressed', 'true');
    els.micBtn.title = 'Stop and transcribe';
    tickVoice();
    voiceState.ticker = setInterval(tickVoice, 250);
    voiceState.maxTimer = setTimeout(() => {
      if (voiceState.recorder) stopVoice();
    }, VOICE_MAX_MS);
  }

  function tickVoice() {
    const seconds = Math.floor((Date.now() - voiceState.startedAt) / 1000);
    setVoiceStatus(`Listening… ${seconds}s — tap the mic to stop, Esc to cancel`, 'listening');
  }

  function stopVoice() {
    const recorder = voiceState.recorder;
    if (!recorder) return;
    clearInterval(voiceState.ticker);
    clearTimeout(voiceState.maxTimer);
    voiceState.ticker = null;
    voiceState.maxTimer = null;
    try {
      if (recorder.state !== 'inactive') recorder.stop();
    } catch (_) {
      /* already stopped */
    }
  }

  function cancelVoice() {
    voiceState.cancelled = true;
    stopVoice();
    setVoiceStatus('Recording cancelled.');
    setTimeout(() => setVoiceStatus(''), 1600);
  }

  function releaseVoiceStream() {
    if (!voiceState.stream) return;
    voiceState.stream.getTracks().forEach((track) => track.stop());
    voiceState.stream = null;
  }

  async function handleRecorderStop() {
    const chunks = voiceState.chunks;
    const cancelled = voiceState.cancelled;

    voiceState.recorder = null;
    voiceState.chunks = [];
    voiceState.cancelled = false;
    releaseVoiceStream();

    if (els.micBtn) {
      els.micBtn.classList.remove('recording');
      els.micBtn.setAttribute('aria-pressed', 'false');
      els.micBtn.title = 'Speak in Hindi, Marathi or English — click to start, click to stop';
    }

    if (cancelled) return;

    const blob = new Blob(chunks, { type: (chunks[0] && chunks[0].type) || 'audio/webm' });
    if (blob.size < 800) {
      setVoiceStatus('That was too short — hold the mic and speak your command.', 'error');
      return;
    }

    voiceState.busy = true;
    if (els.micBtn) els.micBtn.classList.add('busy');
    setVoiceStatus('Transcribing…');

    try {
      const form = new FormData();
      form.append('file', blob, 'voice.webm');
      form.append('language', 'auto');

      const data = await api('/api/voice/transcribe', { method: 'POST', body: form });
      const text = (data.text || '').trim();
      if (!text) {
        setVoiceStatus("I couldn't make out any words. Try again, a little closer to the mic.", 'error');
        return;
      }

      insertTranscript(text);
      setVoiceStatus(`Added ${voiceLanguageName(data.language)}: “${truncateVoice(text, 70)}”`);
      setTimeout(() => setVoiceStatus(''), 5000);
    } catch (error) {
      setVoiceStatus(error.message || 'Transcription failed.', 'error');
    } finally {
      voiceState.busy = false;
      if (els.micBtn) els.micBtn.classList.remove('busy');
    }
  }

  function insertTranscript(text) {
    const input = els.chatInput;
    if (!input) return;
    const existing = input.value;
    const separator = existing && !/\s$/.test(existing) ? ' ' : '';
    input.value = existing ? existing + separator + text : text;
    input.dispatchEvent(new Event('input'));
    input.focus();
    const end = input.value.length;
    input.setSelectionRange(end, end);
    requestAnimationFrame(() => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight + 2, 160) + 'px';
    });
  }

  function voiceLanguageName(tag) {
    if (!tag) return 'text';
    const names = { 'hi-IN': 'Hindi', 'mr-IN': 'Marathi', 'en-IN': 'English' };
    return tag
      .split('+')
      .map((part) => names[part] || part)
      .join(' + ');
  }

  function truncateVoice(text, limit) {
    return text.length > limit ? text.slice(0, limit - 1) + '…' : text;
  }

  // ==========================================================================
  // TABS
  // ==========================================================================
  function setupTabs() {
    els.tabAsk.addEventListener('click', () => switchTab('ask'));
    els.tabData.addEventListener('click', () => switchTab('data'));
  }

  function switchTab(tabName) {
    state.activeTab = tabName;
    const isAsk = tabName === 'ask';

    els.tabAsk.classList.toggle('active', isAsk);
    els.tabData.classList.toggle('active', !isAsk);
    els.tabAsk.setAttribute('aria-selected', String(isAsk));
    els.tabData.setAttribute('aria-selected', String(!isAsk));

    els.panelAsk.classList.toggle('active', isAsk);
    els.panelData.classList.toggle('active', !isAsk);

    if (isAsk) {
      stopFormPolling();
    } else {
      loadVault();
      refreshFormStatus();
      startFormPolling();
    }
  }

  // ==========================================================================
  // THEME
  // ==========================================================================
  function setupThemeToggle() {
    els.themeToggle.addEventListener('click', toggleTheme);
  }

  function toggleTheme() {
    state.theme = state.theme === 'light' ? 'dark' : 'light';
    applyTheme();
    store.set(STORAGE_KEYS.theme, state.theme);
  }

  function applyTheme() {
    els.panel.setAttribute('data-theme', state.theme);
  }

  // ==========================================================================
  // HISTORY / PAST MEMORIES
  // ==========================================================================
  function setupHistory() {
    if (!els.historyToggle || !els.historySection) return;
    els.historyToggle.addEventListener('click', toggleHistory);
  }

  function setupNewChat() {
    if (!els.newChat) return;
    els.newChat.addEventListener('click', startNewChat);
  }

  async function startNewChat() {
    state.sessionId = newSessionId();
    await store.set(STORAGE_KEYS.session, state.sessionId);
    finalizeStreamingMessage();
    showWelcome();
    loadHistory();
    els.chatInput.focus();
  }

  function loadHistory() {
    sendMessage({ type: 'get_history', limit: 20 });
  }

  function renderHistory(history) {
    if (!els.historyList) return;

    if (!history || history.length === 0) {
      els.historyList.innerHTML = `
        <div class="history-empty">
          <span class="empty-icon">💭</span>
          <p class="empty-text">No past conversations yet</p>
          <span class="empty-hint">Your chat history will appear here</span>
        </div>
      `;
      return;
    }

    els.historyList.innerHTML = history.map(item => `
      <div class="history-item ${item.type || 'chat'}" data-session-id="${escapeHtml(item.session_id)}" data-role="button" tabindex="0" aria-label="Load conversation from ${escapeHtml(formatDate(item.timestamp))}">
        <div class="history-item-icon">${item.type === 'task' ? '⚡' : '💬'}</div>
        <div class="history-item-content">
          <div class="history-item-title">${escapeHtml(item.title || item.preview || 'Untitled')}</div>
          <div class="history-item-meta">
            <span class="history-item-time">${escapeHtml(formatDate(item.timestamp))}</span>
            ${item.model ? `<span class="history-item-model">${escapeHtml(item.model)}</span>` : ''}
          </div>
          ${item.preview ? `<div class="history-item-preview">${escapeHtml(item.preview)}</div>` : ''}
        </div>
        <button type="button" class="history-delete" title="Delete this memory" aria-label="Delete this memory">✕</button>
      </div>
    `).join('');

    els.historyList.querySelectorAll('.history-item').forEach(item => {
      const open = () => loadHistoryItem(item.dataset.sessionId);
      item.addEventListener('click', open);
      item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          open();
        }
      });

      const removeBtn = item.querySelector('.history-delete');
      if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          e.preventDefault();
          requestDeleteHistoryItem(item, removeBtn);
        });
        removeBtn.addEventListener('keydown', (e) => {
          // Keep Enter/Space on the button from also loading the conversation.
          if (e.key === 'Enter' || e.key === ' ') {
            e.stopPropagation();
          }
        });
      }
    });
  }

  // Two-step delete: first click turns the button into "Delete?", a second click
  // within 3 seconds removes that one conversation. Nothing else is touched.
  function requestDeleteHistoryItem(item, button) {
    const sessionId = item.dataset.sessionId;
    if (!sessionId) return;

    if (button.dataset.confirm === '1') {
      button.dataset.confirm = '';
      button.disabled = true;
      button.classList.remove('confirming');
      button.textContent = '…';
      sendMessage({ type: 'delete_session', session_id: sessionId });
      return;
    }

    button.dataset.confirm = '1';
    button.classList.add('confirming');
    button.textContent = 'Delete?';
    setTimeout(() => {
      if (button.dataset.confirm === '1') {
        button.dataset.confirm = '';
        button.classList.remove('confirming');
        button.textContent = '✕';
      }
    }, 3000);
  }

  function toggleHistory() {
    if (!els.historySection) return;
    const isCollapsed = els.historySection.classList.toggle('collapsed');
    els.historyToggle.setAttribute('aria-expanded', String(!isCollapsed));
    els.historyToggle.setAttribute(
      'aria-label',
      isCollapsed ? 'Show past memories' : 'Hide past memories'
    );
  }

  function loadHistoryItem(sessionId) {
    if (!sessionId) return;
    state.sessionId = sessionId;
    store.set(STORAGE_KEYS.session, sessionId);
    sendMessage({ type: 'load_session', session_id: sessionId });
  }

  function loadSession(sessionData) {
    if (!sessionData || !Array.isArray(sessionData.messages)) return;

    if (sessionData.session_id) {
      state.sessionId = sessionData.session_id;
      store.set(STORAGE_KEYS.session, state.sessionId);
    }

    els.chatMessages.innerHTML = '';

    if (sessionData.messages.length === 0) {
      showWelcome();
      return;
    }

    sessionData.messages.forEach(msg => {
      if (msg.role === 'user') {
        addUserMessage(msg.content);
      } else if (msg.role === 'assistant') {
        const msgEl = addAssistantMessage();
        const bubble = msgEl.querySelector('.message-bubble');
        bubble.textContent = msg.content;
        delete bubble.dataset.streaming;
        renderRichText(bubble);
        addDocumentLink(bubble);
      } else {
        addSystemMessage(msg.content, msg.type === 'error' ? 'error' : 'system');
      }
    });

    scrollToBottom();
  }

  function formatDate(timestamp) {
    if (!timestamp) return 'Unknown';
    const normalized = typeof timestamp === 'string' && !/[zZ]|[+-]\d{2}:?\d{2}$/.test(timestamp)
      ? timestamp.replace(' ', 'T') + 'Z'
      : timestamp;
    const date = new Date(normalized);
    if (Number.isNaN(date.getTime())) return 'Unknown';

    const diff = Date.now() - date.getTime();
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;

    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
  }

  // ==========================================================================
  // YOUR DATA (uploaded documents -> fields -> fill form)
  // ==========================================================================
  const API_BASE = 'http://127.0.0.1:8764';
  const vaultState = {
    fields: {},
    sources: [],
    schema: [],
    hasData: false,
    form: { ok: false, has_form: false, fields: 0 },
    pollTimer: null,
    busy: false,
    learning: false
  };

  async function api(path, options) {
    let response;
    try {
      response = await fetch(API_BASE + path, options);
    } catch (e) {
      throw new Error('Browzy backend is not reachable. Is it running?');
    }
    let data = null;
    try {
      data = await response.json();
    } catch (_) {
      /* empty body */
    }
    if (!response.ok) {
      const detail = data && (data.detail || data.error);
      throw new Error(typeof detail === 'string' && detail ? detail : `Request failed (${response.status})`);
    }
    return data || {};
  }

  function setupDataVault() {
    if (!els.vaultDropzone || !els.vaultFile) return;

    els.vaultDropzone.addEventListener('click', () => els.vaultFile.click());
    els.vaultDropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        els.vaultFile.click();
      }
    });

    els.vaultFile.addEventListener('change', () => {
      const file = els.vaultFile.files && els.vaultFile.files[0];
      els.vaultFile.value = '';
      if (file) uploadVaultFile(file);
    });

    ['dragenter', 'dragover'].forEach((name) => {
      els.vaultDropzone.addEventListener(name, (e) => {
        e.preventDefault();
        els.vaultDropzone.classList.add('dragging');
      });
    });
    ['dragleave', 'drop'].forEach((name) => {
      els.vaultDropzone.addEventListener(name, (e) => {
        e.preventDefault();
        els.vaultDropzone.classList.remove('dragging');
      });
    });
    els.vaultDropzone.addEventListener('drop', (e) => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) uploadVaultFile(file);
    });

    els.fillPageBtn.addEventListener('click', fillCurrentForm);
    els.clearVault.addEventListener('click', clearVault);

    els.vaultFields.addEventListener('change', async (e) => {
      const input = e.target.closest('input[data-key]');
      if (!input) return;
      const key = input.dataset.key;
      const value = input.value.trim();
      try {
        const data = await api('/api/profile/field', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key, value })
        });
        vaultState.fields = data.fields || {};
        vaultState.hasData = Object.keys(vaultState.fields).length > 0;
        renderVaultFields();
        renderVaultCount();
        refreshFormStatus();
      } catch (err) {
        setVaultStatus(err.message, 'error');
      }
    });

    els.vaultFields.addEventListener('click', async (e) => {
      const button = e.target.closest('.vault-field-remove');
      if (!button) return;
      e.preventDefault();
      e.stopPropagation();
      const key = button.dataset.key;
      if (!key) return;

      if (button.dataset.confirm !== '1') {
        button.dataset.confirm = '1';
        button.classList.add('confirming');
        button.textContent = 'Delete?';
        setTimeout(() => {
          if (button.dataset.confirm === '1') {
            button.dataset.confirm = '';
            button.classList.remove('confirming');
            button.textContent = '✕';
          }
        }, 3000);
        return;
      }

      button.dataset.confirm = '';
      button.disabled = true;
      try {
        const data = await api('/api/profile/field', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key, value: '' })
        });
        vaultState.fields = data.fields || {};
        vaultState.hasData = Object.keys(vaultState.fields).length > 0;
        renderVaultFields();
        renderVaultCount();
        setVaultStatus('Deleted that saved field.', 'ok');
        refreshFormStatus();
      } catch (err) {
        setVaultStatus(err.message, 'error');
        button.disabled = false;
        button.textContent = '✕';
      }
    });

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && state.activeTab === 'data') {
        refreshFormStatus();
      }
    });
  }

  function setVaultStatus(text, kind) {
    if (!els.vaultStatus) return;
    els.vaultStatus.textContent = text || '';
    els.vaultStatus.className = 'vault-status' + (kind ? ' ' + kind : '');
  }

  async function loadVault() {
    try {
      const data = await api('/api/profile');
      vaultState.fields = data.fields || {};
      vaultState.sources = data.sources || [];
      vaultState.schema = data.schema || [];
      vaultState.hasData = Boolean(data.has_data);
      renderVault();
      refreshFormStatus();
    } catch (err) {
      setVaultStatus(err.message, 'error');
    }
  }

  function renderVault() {
    renderVaultCount();
    renderVaultSources();
    renderVaultFields();
  }

  function renderVaultCount() {
    if (!els.vaultCount) return;
    const count = Object.keys(vaultState.fields).length;
    els.vaultCount.textContent = count ? `${count} field${count === 1 ? '' : 's'}` : 'Empty';
    els.vaultCount.className = 'badge ' + (count ? 'badge-success' : 'badge-info');
  }

  function renderVaultSources() {
    if (!els.vaultSources) return;
    if (!vaultState.sources.length) {
      els.vaultSources.innerHTML = '';
      return;
    }
    els.vaultSources.innerHTML = vaultState.sources
      .slice()
      .reverse()
      .map((source) => `
        <span class="source-chip" title="${escapeHtml(source.file || 'document')}">
          <strong>${escapeHtml(source.file || 'document')}</strong>
          <span>${Number(source.fields) || 0} fields</span>
        </span>
      `)
      .join('');
  }

  function renderVaultFields() {
    if (!els.vaultFields) return;
    const keys = Object.keys(vaultState.fields);
    if (!keys.length) {
      els.vaultFields.innerHTML = '<div class="vault-empty">No data yet — upload a document above.</div>';
      return;
    }

    const groups = [];
    const seen = new Set();
    vaultState.schema.forEach((item) => {
      if (vaultState.fields[item.key] === undefined) return;
      seen.add(item.key);
      let group = groups.find((g) => g.name === item.section);
      if (!group) {
        group = { name: item.section || 'Other', rows: [] };
        groups.push(group);
      }
      group.rows.push({ key: item.key, label: item.label || item.key });
    });

    const extra = keys.filter((key) => !seen.has(key));
    if (extra.length) {
      groups.push({ name: 'Other', rows: extra.map((key) => ({ key, label: key.replace(/_/g, ' ') })) });
    }

    els.vaultFields.innerHTML = groups.map((group) => `
      <div class="vault-group-title">${escapeHtml(group.name)}</div>
      ${group.rows.map((row) => `
        <label class="vault-field">
          <span class="vault-field-label">${escapeHtml(row.label)}</span>
          <input type="text" data-key="${escapeHtml(row.key)}"
                 value="${escapeHtml(vaultState.fields[row.key] || '')}"
                 aria-label="${escapeHtml(row.label)}">
          <button type="button" class="vault-field-remove" data-key="${escapeHtml(row.key)}"
                  title="Delete this saved field" aria-label="Delete ${escapeHtml(row.label)}">✕</button>
        </label>
      `).join('')}
    `).join('');
  }

  async function uploadVaultFile(file) {
    if (vaultState.busy) return;
    vaultState.busy = true;
    setVaultStatus(`Reading ${file.name}… this can take a moment for scans.`, 'busy');

    const body = new FormData();
    body.append('file', file, file.name);

    try {
      const data = await api('/api/profile/upload', { method: 'POST', body });
      vaultState.fields = data.fields || {};
      vaultState.sources = data.sources || [];
      vaultState.hasData = Object.keys(vaultState.fields).length > 0;
      renderVault();
      setVaultStatus(
        `Added ${data.found} field${data.found === 1 ? '' : 's'} from ${data.file}. The file was deleted.`,
        'ok'
      );
      refreshFormStatus();
    } catch (err) {
      setVaultStatus(err.message, 'error');
    } finally {
      vaultState.busy = false;
    }
  }

  async function clearVault() {
    if (!Object.keys(vaultState.fields).length) {
      setVaultStatus('The vault is already empty.', '');
      return;
    }
    if (!window.confirm('Delete every saved field? You can upload the document again later.')) {
      return;
    }
    if (els.clearVault) els.clearVault.disabled = true;
    setVaultStatus('Clearing saved data...', 'busy');
    try {
      const data = await api('/api/profile', { method: 'DELETE' });
      vaultState.fields = data.fields || {};
      vaultState.sources = data.sources || [];
      vaultState.hasData = Object.keys(vaultState.fields).length > 0;
      renderVault();
      setVaultStatus(vaultState.hasData ? 'Some saved data could not be cleared.' : 'Vault cleared.', vaultState.hasData ? 'error' : 'ok');
      refreshFormStatus();
    } catch (err) {
      setVaultStatus(err.message, 'error');
    } finally {
      if (els.clearVault) els.clearVault.disabled = false;
    }
  }

  async function refreshFormStatus() {
    if (state.activeTab !== 'data' || !els.formState) return;
    try {
      const data = await api('/api/form/status');
      vaultState.form = data;
      const badge = els.formState;
      if (!data.ok) {
        badge.textContent = 'No page';
        badge.className = 'badge';
      } else if (data.has_form) {
        badge.textContent = `${data.fields} field${data.fields === 1 ? '' : 's'} ready`;
        badge.className = 'badge badge-success';
      } else if (data.total > 0) {
        badge.textContent = 'Nothing left to fill';
        badge.className = 'badge badge-info';
      } else {
        badge.textContent = 'No form found';
        badge.className = 'badge';
      }

      const canFill = Boolean(data.ok && data.has_form && vaultState.hasData);
      els.fillPageBtn.disabled = !canFill;

      if (!vaultState.hasData) {
        els.fillHint.textContent = 'Upload a document above first — Browzy only fills from your own data.';
      } else if (!data.ok) {
        els.fillHint.textContent = 'Open the Browzy browser window, then try again.';
      } else if (!data.has_form && data.total > 0) {
        els.fillHint.textContent = 'Every field Browzy can fill on this page is already filled.';
      } else if (!data.has_form) {
        els.fillHint.textContent = 'Open a signup or checkout form and Browzy will fill its empty fields.';
      } else {
        els.fillHint.textContent = 'Browzy will fill the empty fields on this page from your vault.';
      }

      // Remember what the user types on forms Browzy has never seen. The backend
      // only stores values that stayed the same between two polls, so half-typed
      // words are never saved.
      if (data.ok && !vaultState.busy && !vaultState.learning) {
        vaultState.learning = true;
        api('/api/form/learn', { method: 'POST' })
          .then((learned) => {
            const count = Number(learned && learned.learned) || 0;
            if (!count) return;
            setVaultStatus(
              `Saved ${count} new field${count === 1 ? '' : 's'} from this form.`,
              'ok'
            );
            loadVault();
          })
          .catch(() => {})
          .finally(() => {
            vaultState.learning = false;
          });
      }
    } catch (err) {
      els.formState.textContent = 'Backend offline';
      els.formState.className = 'badge badge-danger';
      els.fillPageBtn.disabled = true;
    }
  }

  async function fillCurrentForm() {
    if (vaultState.busy) return;
    vaultState.busy = true;
    els.fillPageBtn.disabled = true;
    setVaultStatus('Filling the form…', 'busy');
    try {
      const result = await api('/api/form/fill', { method: 'POST' });
      const filled = Number(result.filled) || 0;
      const learned = Number(result.learned) || 0;
      const blocked = (result.blocked || []).length;
      const unmatched = (result.unmatched || []).length;
      if (filled) {
        const notes = [];
        if (learned) notes.push(`saved ${learned} new field${learned === 1 ? '' : 's'} for next time`);
        if (unmatched) notes.push(`${unmatched} didn't match your saved data`);
        if (blocked) notes.push('search, login and payment inputs were left alone');
        const suffix = notes.length ? ` ${notes.join('; ')}.` : '';
        setVaultStatus(
          `Filled ${filled} field${filled === 1 ? '' : 's'} on this page.${suffix}`,
          'ok'
        );
      } else if (blocked) {
        setVaultStatus('This page only has search, coupon or login inputs — nothing Browzy fills automatically.', 'error');
      } else {
        setVaultStatus('No empty fields matched your saved data on this page.', 'error');
      }
    } catch (err) {
      setVaultStatus(err.message, 'error');
    } finally {
      vaultState.busy = false;
      refreshFormStatus();
    }
  }

  function startFormPolling() {
    stopFormPolling();
    vaultState.pollTimer = setInterval(() => {
      if (document.visibilityState === 'visible') refreshFormStatus();
    }, 4000);
  }

  function stopFormPolling() {
    if (vaultState.pollTimer) {
      clearInterval(vaultState.pollTimer);
      vaultState.pollTimer = null;
    }
  }

  // ==========================================================================
  // CLOSE BUTTON
  // ==========================================================================
  function setupCloseButton() {
    els.closePanel.addEventListener('click', async () => {
      // Chrome 156+ exposes sidePanel.close(); fall back to window.close() for
      // older builds and when the panel is opened as a normal tab.
      try {
        if (typeof chrome !== 'undefined' && chrome.sidePanel && typeof chrome.sidePanel.close === 'function') {
          const win = await chrome.windows.getCurrent();
          await chrome.sidePanel.close({ windowId: win.id });
          return;
        }
      } catch (_) {
        /* fall through */
      }
      try {
        window.close();
      } catch (_) {
        els.panel.style.display = 'none';
      }
    });
  }

  // ==========================================================================
  // START
  // ==========================================================================
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose for debugging
  window.BrowzyPanel = {
    state,
    sendMessage,
    addSystemMessage,
    addUserMessage,
    addAssistantMessage,
    appendToken,
    finalizeStreamingMessage,
    startNewChat
  };
})();
