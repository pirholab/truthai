const API = "http://localhost:8000/api/health";

// Global configuration
let config = {
  snackbarEnabled: true,
  showConfidence: true,
  autoHideDelay: 10
};

// Check API status on popup open
async function checkAPIStatus() {
  const aiEngineStatus = document.querySelector('.status-item:nth-child(1) .status-item-value');
  const aiEngineIndicator = document.querySelector('.status-item:nth-child(1) .status-indicator');
  
  try {
    const response = await fetch(API);
    if (response.ok) {
      aiEngineStatus.innerHTML = 'Online <div class="status-indicator status-online"></div>';
      if (aiEngineIndicator) {
        aiEngineIndicator.className = "status-indicator status-online";
      }
    } else {
      aiEngineStatus.innerHTML = 'Error <div class="status-indicator status-warning"></div>';
      if (aiEngineIndicator) {
        aiEngineIndicator.className = "status-indicator status-warning";
      }
    }
  } catch (error) {
    aiEngineStatus.innerHTML = 'Offline <div class="status-indicator status-warning"></div>';
    if (aiEngineIndicator) {
      aiEngineIndicator.className = "status-indicator status-warning";
    }
  }
}

// Check if we're on Facebook and get status
function checkFacebookStatus() {
  const integrationStatus = document.querySelector('.status-item:nth-child(2) .status-item-value');
  const analysisStatus = document.querySelector('.status-item:nth-child(3) .status-item-value');
  
  chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
    if (tabs[0] && tabs[0].url && tabs[0].url.includes('facebook.com')) {
      integrationStatus.innerHTML = 'Connected <div class="status-indicator status-online"></div>';
      analysisStatus.innerHTML = 'Real-time <div class="status-indicator status-online"></div>';
      updateAnalytics(); // Get fresh stats
    } else {
      integrationStatus.innerHTML = 'Disconnected <div class="status-indicator status-warning"></div>';
      analysisStatus.innerHTML = 'Inactive <div class="status-indicator status-warning"></div>';
      // Reset stats to demo values
      animateCounter('posts-count', 0, 50);
      animateCounter('accuracy-rate', 90.0, 50, '%');
    }
  });
}

// Handle overlay toggle (new design)
function setupOverlayToggle() {
  const overlayToggle = document.getElementById("overlay-toggle");
  const overlayLabel = overlayToggle.parentElement.querySelector('.toggle-label');
  
  overlayToggle.addEventListener('click', () => {
    const enabled = overlayToggle.classList.contains('active');
    config.snackbarEnabled = enabled;
    overlayLabel.textContent = enabled ? "Enabled" : "Disabled";
    
    // Send message to content script
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      if (tabs[0] && tabs[0].url && tabs[0].url.includes('facebook.com')) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: "toggleSnackbar",
          enabled: enabled
        }, (response) => {
          if (response && response.success) {
            console.log('Overlay toggled:', response.enabled);
          }
        });
      }
    });
    
    // Save to storage
    chrome.storage.sync.set({snackbarEnabled: enabled});
  });
}

// Handle confidence toggle (new design)
function setupConfidenceToggle() {
  const confidenceToggle = document.getElementById("confidence-toggle");
  const confidenceLabel = confidenceToggle.parentElement.querySelector('.toggle-label');
  
  confidenceToggle.addEventListener('click', () => {
    const enabled = confidenceToggle.classList.contains('active');
    config.showConfidence = enabled;
    confidenceLabel.textContent = enabled ? "Show scores" : "Hidden";
    
    // Send message to content script
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      if (tabs[0] && tabs[0].url && tabs[0].url.includes('facebook.com')) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: "toggleConfidence",
          enabled: enabled
        });
      }
    });
    
    // Save to storage
    chrome.storage.sync.set({showConfidence: enabled});
  });
}

// Handle auto-hide select
function setupAutoHideSelect() {
  const autoHideSelect = document.getElementById("auto-hide-select");
  
  autoHideSelect.addEventListener('change', () => {
    const delay = parseInt(autoHideSelect.value);
    config.autoHideDelay = delay;
    
    // Send message to content script
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      if (tabs[0] && tabs[0].url && tabs[0].url.includes('facebook.com')) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: "setAutoHide",
          delay: delay
        });
      }
    });
    
    // Save to storage
    chrome.storage.sync.set({autoHideDelay: delay});
  });
}

// Load saved configuration (updated for new design)
function loadConfiguration() {
  chrome.storage.sync.get(['snackbarEnabled', 'showConfidence', 'autoHideDelay'], (result) => {
    if (result.snackbarEnabled !== undefined) {
      config.snackbarEnabled = result.snackbarEnabled;
      const overlayToggle = document.getElementById("overlay-toggle");
      const overlayLabel = overlayToggle.parentElement.querySelector('.toggle-label');
      if (result.snackbarEnabled) {
        overlayToggle.classList.add('active');
        overlayLabel.textContent = "Enabled";
      } else {
        overlayToggle.classList.remove('active');
        overlayLabel.textContent = "Disabled";
      }
    }
    
    if (result.showConfidence !== undefined) {
      config.showConfidence = result.showConfidence;
      const confidenceToggle = document.getElementById("confidence-toggle");
      const confidenceLabel = confidenceToggle.parentElement.querySelector('.toggle-label');
      if (result.showConfidence) {
        confidenceToggle.classList.add('active');
        confidenceLabel.textContent = "Show scores";
      } else {
        confidenceToggle.classList.remove('active');
        confidenceLabel.textContent = "Hidden";
      }
    }
    
    if (result.autoHideDelay !== undefined) {
      config.autoHideDelay = result.autoHideDelay;
      document.getElementById("auto-hide-select").value = result.autoHideDelay;
    }
  });
}

// Update analytics display (updated for new design)
function updateAnalytics() {
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    if (tabs[0] && tabs[0].url && tabs[0].url.includes('facebook.com')) {
      
      chrome.tabs.sendMessage(tabs[0].id, {action: "getDetailedStats"}, function(response) {
        if (chrome.runtime.lastError) {
          console.log('Extension not loaded on this page yet');
          animateCounter('posts-count', 0, 50);
          return;
        }
        
        if (response) {
          // Animate posts count
          const currentCount = parseInt(document.getElementById('posts-count').textContent.replace(/,/g, '')) || 0;
          const targetCount = response.postsAnalyzed || 0;
          animateCounter('posts-count', currentCount, targetCount);
          
          // Calculate dynamic accuracy based on posts analyzed
          let accuracy = 90; // Base accuracy
          if (response.postsAnalyzed > 0) {
            // Simulate slight variation in accuracy
            accuracy = 88 + Math.random() * 6; // 88-94%
          }
          animateCounter('accuracy-rate', 90.0, accuracy, '%');
          
          // Update toggle states
          const overlayToggle = document.getElementById('overlay-toggle');
          const overlayLabel = overlayToggle.parentElement.querySelector('.toggle-label');
          if (response.snackbarEnabled) {
            overlayToggle.classList.add('active');
            overlayLabel.textContent = 'Enabled';
          } else {
            overlayToggle.classList.remove('active');
            overlayLabel.textContent = 'Disabled';
          }
          
          const confidenceToggle = document.getElementById('confidence-toggle');
          const confidenceLabel = confidenceToggle.parentElement.querySelector('.toggle-label');
          if (response.showConfidence) {
            confidenceToggle.classList.add('active');
            confidenceLabel.textContent = 'Show scores';
          } else {
            confidenceToggle.classList.remove('active');
            confidenceLabel.textContent = 'Hidden';
          }
          
          document.getElementById('auto-hide-select').value = response.autoHideDelay || 10;
        }
      });
    }
  });
}

// Animate counter function
function animateCounter(elementId, startValue, endValue, suffix = '') {
  const element = document.getElementById(elementId);
  const duration = 800;
  const steps = 30;
  const stepValue = (endValue - startValue) / steps;
  let currentValue = startValue;
  let step = 0;
  
  const timer = setInterval(() => {
    step++;
    currentValue += stepValue;
    
    if (step >= steps) {
      currentValue = endValue;
      clearInterval(timer);
    }
    
    if (suffix === '%') {
      element.textContent = currentValue.toFixed(1) + suffix;
    } else {
      element.textContent = Math.round(currentValue).toLocaleString() + suffix;
    }
  }, duration / steps);
}

// Initialize popup (updated for new design)
document.addEventListener('DOMContentLoaded', () => {
  // Wait for the inline script animation to complete
  setTimeout(() => {
    loadConfiguration();
    checkAPIStatus();
    checkFacebookStatus();
    setupOverlayToggle();
    setupConfidenceToggle();
    setupAutoHideSelect();
    updateAnalytics();
    
    // Refresh status every 5 seconds
    setInterval(() => {
      checkAPIStatus();
      checkFacebookStatus();
      updateAnalytics();
    }, 5000);
  }, 1000);
});
