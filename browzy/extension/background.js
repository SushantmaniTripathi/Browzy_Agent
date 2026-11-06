// Browzy background service worker
// Opens the browser's native side panel from the toolbar action or Alt+B.

function enableActionClickPanel() {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((error) => console.error('[Browzy] sidePanel behavior:', error));
}

chrome.runtime.onInstalled.addListener(enableActionClickPanel);
chrome.runtime.onStartup.addListener(enableActionClickPanel);
enableActionClickPanel();

async function openSidePanel() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    if (tab && tab.windowId !== undefined && tab.windowId !== null) {
      await chrome.sidePanel.open({ windowId: tab.windowId });
      return true;
    }
  } catch (error) {
    console.error('[Browzy] Could not open side panel:', error);
  }
  return false;
}

chrome.commands.onCommand.addListener(async (command) => {
  if (command === 'open-browzy') {
    await openSidePanel();
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message && message.type === 'open-side-panel') {
    openSidePanel().then((opened) => sendResponse({ opened }));
    return true;
  }
  return false;
});
