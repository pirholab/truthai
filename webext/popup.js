const API = "http://localhost:8000/health";

// Check API status on popup open
async function checkAPIStatus() {
  const statusElement = document.getElementById("api-status");
  const indicatorElement = document.getElementById("api-indicator");
  
  try {
    const response = await fetch(API);
    if (response.ok) {
      statusElement.textContent = "Online";
      indicatorElement.className = "status-indicator status-online";
    } else {
      statusElement.textContent = "Error";
      indicatorElement.className = "status-indicator status-offline";
    }
  } catch (error) {
    statusElement.textContent = "Offline";
    indicatorElement.className = "status-indicator status-offline";
  }
}

// Check if we're on Facebook and get status
function checkFacebookStatus() {
  const facebookStatusElement = document.getElementById("facebook-status");
  const snackbarStatusElement = document.getElementById("snackbar-status");
  const snackbarToggle = document.getElementById("snackbar-toggle");
  
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    if (tabs[0] && tabs[0].url && tabs[0].url.includes('facebook.com')) {
      facebookStatusElement.textContent = "Active";
      
      // Get stats from content script
      chrome.tabs.sendMessage(tabs[0].id, {action: "getStats"}, (response) => {
        if (response && response.postsAnalyzed !== undefined) {
          document.getElementById("posts-count").textContent = response.postsAnalyzed;
          
          // Update snackbar toggle state
          if (response.snackbarEnabled !== undefined) {
            snackbarToggle.checked = response.snackbarEnabled;
            snackbarStatusElement.textContent = response.snackbarEnabled ? "Enabled" : "Disabled";
          }
        }
      });
    } else {
      facebookStatusElement.textContent = "Not on Facebook";
    }
  });
}

// Handle snackbar toggle
function setupSnackbarToggle() {
  const snackbarToggle = document.getElementById("snackbar-toggle");
  const snackbarStatusElement = document.getElementById("snackbar-status");
  
  snackbarToggle.addEventListener('change', () => {
    const enabled = snackbarToggle.checked;
    snackbarStatusElement.textContent = enabled ? "Enabled" : "Disabled";
    
    // Send message to content script
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      if (tabs[0] && tabs[0].url && tabs[0].url.includes('facebook.com')) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: "toggleSnackbar",
          enabled: enabled
        }, (response) => {
          if (response && response.success) {
            console.log('Snackbar toggled:', response.enabled);
          }
        });
      }
    });
  });
}

// Initialize popup
document.addEventListener('DOMContentLoaded', () => {
  checkAPIStatus();
  checkFacebookStatus();
  setupSnackbarToggle();
  
  // Refresh status every 5 seconds
  setInterval(() => {
    checkAPIStatus();
    checkFacebookStatus();
  }, 5000);
});
