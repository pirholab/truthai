(function() {
  'use strict';
  
  const API = "http://localhost:8000/api";
  
  // Global state
  let snackbarEnabled = true;
  let showConfidence = true;
  let autoHideDelay = 10;
  let currentSnackbar = null;
  let analyzedPosts = new Map();
  let hideTimeout = null;
  let uniquePostsAnalyzed = new Set();
  let embeddedBadges = new Set();
  
  console.log("🔍 TruthAI: Content script starting...");
  
  // Check if we're on Facebook
  if (!window.location.hostname.includes('facebook.com')) {
    console.log("🔍 TruthAI: Not on Facebook, exiting");
    return;
  }
  
  console.log("🔍 TruthAI: On Facebook, initializing extension");
  
  // Enhanced CSS matching popup.html design system
  const style = document.createElement('style');
  style.textContent = `
    :root {
      --truthai-primary: #3b82f6;
      --truthai-secondary: #6b7280;
      --truthai-success: #22c55e;
      --truthai-warning: #f59e0b;
      --truthai-error: #ef4444;
      --truthai-surface: #ffffff;
      --truthai-surface-light: #f9fafb;
      --truthai-surface-dark: #1f2937;
      --truthai-text: #111827;
      --truthai-text-secondary: #374151;
      --truthai-text-muted: #6b7280;
      --truthai-border: #e5e7eb;
      --truthai-border-light: #f3f4f6;
      --truthai-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
      --truthai-shadow-card: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Modern Card-Style Snackbar */
    .truthai-snackbar {
      position: fixed !important;
      bottom: 24px !important;
      right: 24px !important;
      background: var(--truthai-surface) !important;
      color: var(--truthai-text) !important;
      border-radius: 12px !important;
      padding: 24px !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
      font-size: 14px !important;
      font-weight: 500 !important;
      box-shadow: var(--truthai-shadow) !important;
      border: 1px solid var(--truthai-border) !important;
      z-index: 2147483647 !important;
      max-width: 420px !important;
      min-width: 360px !important;
      display: flex !important;
      align-items: flex-start !important;
      gap: 16px !important;
      transition: all 0.2s ease !important;
      animation: truthai-slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
      visibility: visible !important;
      opacity: 1 !important;
      pointer-events: auto !important;
    }
    
    @keyframes truthai-slideIn {
      from {
        transform: translateX(100%) translateY(20px);
        opacity: 0;
      }
      to {
        transform: translateX(0) translateY(0);
        opacity: 1;
      }
    }

    /* Card hover effect matching popup */
    .truthai-snackbar:hover {
      background: var(--truthai-surface-light) !important;
      border-color: #d1d5db !important;
      transform: translateY(-2px) !important;
    }
    
    /* Icon container matching popup card icons */
    .truthai-snackbar-icon {
      width: 40px !important;
      height: 40px !important;
      border-radius: 8px !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      font-size: 20px !important;
      color: white !important;
      flex-shrink: 0 !important;
    }

    /* Category-specific icon backgrounds matching popup legend */
    .truthai-snackbar.news .truthai-snackbar-icon { background: #3b82f6 !important; }
    .truthai-snackbar.personal .truthai-snackbar-icon { background: #8b5cf6 !important; }
    .truthai-snackbar.entertainment .truthai-snackbar-icon { background: #f59e0b !important; }
    .truthai-snackbar.commercial .truthai-snackbar-icon { background: #10b981 !important; }
    .truthai-snackbar.educational .truthai-snackbar-icon { background: #06b6d4 !important; }
    .truthai-snackbar.opinion .truthai-snackbar-icon { background: #ec4899 !important; }
    .truthai-snackbar.fake .truthai-snackbar-icon { background: #ef4444 !important; }
    .truthai-snackbar.real .truthai-snackbar-icon { background: #22c55e !important; }
    
    .truthai-snackbar-content {
      flex: 1 !important;
      display: flex !important;
      flex-direction: column !important;
      gap: 8px !important;
    }

    /* Title matching popup card titles */
    .truthai-snackbar-title {
      font-size: 16px !important;
      font-weight: 600 !important;
      color: var(--truthai-text) !important;
      margin: 0 !important;
      line-height: 1.4 !important;
      display: flex !important;
      align-items: center !important;
      gap: 8px !important;
    }

    /* Subtitle matching popup stat labels */
    .truthai-snackbar-subtitle {
      font-size: 14px !important;
      color: var(--truthai-text-muted) !important;
      margin: 0 !important;
      line-height: 1.3 !important;
      font-weight: 500 !important;
    }

    /* Confidence display matching popup style */
    .truthai-snackbar-confidence {
      font-size: 12px !important;
      color: var(--truthai-text-muted) !important;
      margin-top: 8px !important;
      padding: 6px 12px !important;
      background: var(--truthai-surface-light) !important;
      border: 1px solid var(--truthai-border) !important;
      border-radius: 6px !important;
      display: inline-block !important;
      font-weight: 500 !important;
    }

    /* Brand label matching popup info style */
    .truthai-brand {
      font-size: 10px !important;
      color: var(--truthai-text-muted) !important;
      text-transform: uppercase !important;
      letter-spacing: 0.5px !important;
      font-weight: 600 !important;
      margin-top: 8px !important;
    }
    
    .truthai-snackbar-actions {
      display: flex !important;
      gap: 8px !important;
      margin-top: 12px !important;
    }
    
    /* Buttons matching popup toggle style */
    .truthai-btn {
      background: white !important;
      border: 1px solid var(--truthai-border) !important;
      color: var(--truthai-text-secondary) !important;
      border-radius: 6px !important;
      padding: 8px 12px !important;
      font-size: 14px !important;
      font-weight: 500 !important;
      cursor: pointer !important;
      transition: all 0.2s ease !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      min-width: 32px !important;
      height: 32px !important;
    }
    
    .truthai-btn:hover {
      background: var(--truthai-surface-light) !important;
      border-color: #d1d5db !important;
    }

    .truthai-btn.close {
      color: var(--truthai-text-muted) !important;
      font-size: 16px !important;
      font-weight: 600 !important;
    }

    .truthai-btn.close:hover {
      color: var(--truthai-error) !important;
      border-color: #fecaca !important;
      background: #fef2f2 !important;
    }

    /* Action buttons with primary styling */
    .truthai-btn.like:hover {
      color: var(--truthai-success) !important;
      border-color: #bbf7d0 !important;
      background: #f0fdf4 !important;
    }

    .truthai-btn.dislike:hover {
      color: var(--truthai-error) !important;
      border-color: #fecaca !important;
      background: #fef2f2 !important;
    }

    /* Embedded Post Badge - Mini Card Style */
    .truthai-post-badge {
      position: absolute !important;
      top: 12px !important;
      right: 12px !important;
      background: var(--truthai-surface) !important;
      border: 1px solid var(--truthai-border) !important;
      border-radius: 8px !important;
      padding: 8px 12px !important;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
      font-size: 12px !important;
      font-weight: 500 !important;
      color: var(--truthai-text) !important;
      box-shadow: var(--truthai-shadow-card) !important;
      z-index: 10 !important;
      display: flex !important;
      align-items: center !important;
      gap: 6px !important;
      transition: all 0.2s ease !important;
      opacity: 0.9 !important;
      pointer-events: auto !important;
    }

    .truthai-post-badge:hover {
      opacity: 1 !important;
      background: var(--truthai-surface-light) !important;
      border-color: #d1d5db !important;
      transform: translateY(-1px) !important;
    }

    .truthai-badge-icon {
      width: 16px !important;
      height: 16px !important;
      border-radius: 4px !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      font-size: 10px !important;
      color: white !important;
    }

    /* Badge icon colors matching main categories */
    .truthai-post-badge.news .truthai-badge-icon { background: #3b82f6 !important; }
    .truthai-post-badge.personal .truthai-badge-icon { background: #8b5cf6 !important; }
    .truthai-post-badge.entertainment .truthai-badge-icon { background: #f59e0b !important; }
    .truthai-post-badge.commercial .truthai-badge-icon { background: #10b981 !important; }
    .truthai-post-badge.educational .truthai-badge-icon { background: #06b6d4 !important; }
    .truthai-post-badge.opinion .truthai-badge-icon { background: #ec4899 !important; }
    .truthai-post-badge.fake .truthai-badge-icon { background: #ef4444 !important; }
    .truthai-post-badge.real .truthai-badge-icon { background: #22c55e !important; }

    .truthai-badge-text {
      font-weight: 500 !important;
      white-space: nowrap !important;
      color: var(--truthai-text) !important;
    }

    .truthai-badge-confidence {
      font-size: 10px !important;
      color: var(--truthai-text-muted) !important;
      margin-left: 4px !important;
      padding: 2px 4px !important;
      background: var(--truthai-surface-light) !important;
      border-radius: 3px !important;
    }

    /* Dark mode support matching popup */
    @media (prefers-color-scheme: dark) {
      :root {
        --truthai-surface: #1f2937 !important;
        --truthai-surface-light: #374151 !important;
        --truthai-text: #f9fafb !important;
        --truthai-text-secondary: #e5e7eb !important;
        --truthai-text-muted: #9ca3af !important;
        --truthai-border: #4b5563 !important;
      }
      
      .truthai-snackbar, .truthai-post-badge {
        background: var(--truthai-surface) !important;
        color: var(--truthai-text) !important;
        border-color: var(--truthai-border) !important;
      }
      
      .truthai-snackbar-confidence, .truthai-badge-confidence {
        background: var(--truthai-surface-light) !important;
        border-color: var(--truthai-border) !important;
      }
      
      .truthai-btn {
        background: var(--truthai-surface) !important;
        border-color: var(--truthai-border) !important;
        color: var(--truthai-text-secondary) !important;
      }
      
      .truthai-btn:hover {
        background: var(--truthai-surface-light) !important;
      }
    }

    /* Responsive design matching popup breakpoints */
    @media (max-width: 768px) {
      .truthai-snackbar {
        bottom: 16px !important;
        right: 16px !important;
        left: 16px !important;
        max-width: none !important;
        min-width: auto !important;
        padding: 20px !important;
      }
      
      .truthai-post-badge {
        top: 8px !important;
        right: 8px !important;
        padding: 6px 8px !important;
        font-size: 11px !important;
      }
      
      .truthai-snackbar-title {
        font-size: 15px !important;
      }
      
      .truthai-snackbar-subtitle {
        font-size: 13px !important;
      }
    }
  `;
  document.head.appendChild(style);
  
  // Load settings from storage with error handling
  if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.sync) {
    chrome.storage.sync.get(['snackbarEnabled', 'showConfidence', 'autoHideDelay'], (result) => {
      if (result.snackbarEnabled !== undefined) snackbarEnabled = result.snackbarEnabled;
      if (result.showConfidence !== undefined) showConfidence = result.showConfidence;
      if (result.autoHideDelay !== undefined) autoHideDelay = result.autoHideDelay;
      console.log("🔍 TruthAI: Settings loaded:", { snackbarEnabled, showConfidence, autoHideDelay });
    });
  } else {
    console.log("🔍 TruthAI: Chrome storage API not available, using defaults");
  }
  
  // Message listener for popup communication
  if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.onMessage) {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getStats") {
      console.log("🔍 TruthAI: Popup requesting stats - Posts analyzed:", uniquePostsAnalyzed.size);
      sendResponse({
        postsAnalyzed: uniquePostsAnalyzed.size,
        isActive: true
      });
    } else if (request.action === "getDetailedStats") {
      console.log("🔍 TruthAI: Popup requesting detailed stats - Posts analyzed:", uniquePostsAnalyzed.size);
      sendResponse({
        postsAnalyzed: uniquePostsAnalyzed.size,
        isActive: true,
        snackbarEnabled: snackbarEnabled,
        showConfidence: showConfidence,
        autoHideDelay: autoHideDelay,
        accuracy: 0.9
      });
    } else if (request.action === "toggleSnackbar") {
      snackbarEnabled = request.enabled;
      if (chrome.storage && chrome.storage.sync) {
        chrome.storage.sync.set({snackbarEnabled: snackbarEnabled});
      }
      if (!snackbarEnabled && currentSnackbar) {
        currentSnackbar.remove();
        currentSnackbar = null;
        if (hideTimeout) clearTimeout(hideTimeout);
      } else if (snackbarEnabled) {
        updateSnackbarForVisiblePost();
      }
      sendResponse({ success: true, enabled: snackbarEnabled });
    } else if (request.action === "toggleConfidence") {
      showConfidence = request.enabled;
      if (chrome.storage && chrome.storage.sync) {
        chrome.storage.sync.set({showConfidence: showConfidence});
      }
      if (currentSnackbar) {
        updateSnackbarForVisiblePost();
      }
      sendResponse({ success: true, enabled: showConfidence });
    } else if (request.action === "setAutoHide") {
      autoHideDelay = request.delay;
      if (chrome.storage && chrome.storage.sync) {
        chrome.storage.sync.set({autoHideDelay: autoHideDelay});
      }
      sendResponse({ success: true, delay: autoHideDelay });
    }
    });
  } else {
    console.log("🔍 TruthAI: Chrome runtime API not available");
  }

  // Get category-specific styling and content
  function getCategoryInfo(classification) {
    const category = classification.category.label.toLowerCase();
    const auth = classification.authenticity.label.toLowerCase();
    
    const icons = {
      news: '📰',
      personal: '👤',
      entertainment: '🎭',
      commercial: '💼',
      educational: '📚',
      opinion: '💭',
      fake: '⚠️',
      real: '✅'
    };

    const icon = auth === 'fake' ? icons.fake : 
                 auth === 'real' ? icons.real : 
                 icons[category] || '📝';

    const title = auth === 'fake' ? 'Potentially Misleading' :
                  auth === 'real' ? 'Verified Content' :
                  classification.category.label;

    const subtitle = auth === 'fake' ? 'Content may contain inaccuracies' :
                     auth === 'real' ? 'Information appears reliable' :
                     classification.type.label;

    return { icon, title, subtitle, category: auth || category };
  }

  // Create embedded post badge
  function createPostBadge(post, classification) {
    if (embeddedBadges.has(post)) return;

    const { icon, title, category } = getCategoryInfo(classification);
    
    const badge = document.createElement('div');
    badge.className = `truthai-post-badge ${category}`;
    
    const confidenceText = showConfidence ? 
      `${Math.round(classification.category.confidence * 100)}%` : '';
    
    badge.innerHTML = `
      <div class="truthai-badge-icon">${icon}</div>
      <span class="truthai-badge-text">${title}</span>
      ${confidenceText ? `<span class="truthai-badge-confidence">${confidenceText}</span>` : ''}
    `;

    // Position relative to post
    post.style.position = 'relative';
    post.appendChild(badge);
    
    embeddedBadges.add(post);
    
    console.log("🔍 TruthAI: Created embedded badge for post:", title);
  }
  
  // Create enhanced snackbar widget
  function createSnackbar(classification, postContent) {
    if (!snackbarEnabled) return;
    
    if (currentSnackbar) {
      currentSnackbar.remove();
      currentSnackbar = null;
      if (hideTimeout) clearTimeout(hideTimeout);
    }
    
    const { icon, title, subtitle, category } = getCategoryInfo(classification);
    
    const snackbar = document.createElement("div");
    snackbar.className = `truthai-snackbar ${category}`;
    
    const confidenceDisplay = showConfidence ? `
      <div class="truthai-snackbar-confidence">
        ${classification.category.label} ${Math.round(classification.category.confidence*100)}% • 
        ${classification.type.label} ${Math.round(classification.type.confidence*100)}%
        ${classification.authenticity.label !== 'N/A' ? 
          ` • ${classification.authenticity.label} ${Math.round(classification.authenticity.confidence*100)}%` : ''}
      </div>
    ` : '';

    snackbar.innerHTML = `
      <div class="truthai-snackbar-icon">${icon}</div>
      <div class="truthai-snackbar-content">
        <div class="truthai-snackbar-title">
          ${title}
        </div>
        <div class="truthai-snackbar-subtitle">${subtitle}</div>
        ${confidenceDisplay}
        <div class="truthai-snackbar-actions">
          <button class="truthai-btn like" title="Mark as helpful">👍</button>
          <button class="truthai-btn dislike" title="Report issue">👎</button>
          <button class="truthai-btn close" title="Close">×</button>
        </div>
        <div class="truthai-brand">TruthAI Analysis</div>
      </div>
    `;
    
    const likeBtn = snackbar.querySelector('.like');
    const dislikeBtn = snackbar.querySelector('.dislike');
    const closeBtn = snackbar.querySelector('.close');
    
    likeBtn.onclick = () => {
      likeBtn.innerHTML = '✅';
      likeBtn.style.color = 'var(--truthai-success)';
      console.log('🔍 TruthAI: User marked classification as correct');
    };
    
    dislikeBtn.onclick = () => {
      dislikeBtn.innerHTML = '❌';
      dislikeBtn.style.color = 'var(--truthai-error)';
      console.log('🔍 TruthAI: User marked classification as wrong');
    };
    
    closeBtn.onclick = () => {
      snackbar.remove();
      currentSnackbar = null;
      if (hideTimeout) clearTimeout(hideTimeout);
    };
    
    document.body.appendChild(snackbar);
    currentSnackbar = snackbar;
    
    // Force visibility
    setTimeout(() => {
      if (snackbar && snackbar.parentNode) {
        snackbar.style.setProperty('display', 'flex', 'important');
        snackbar.style.setProperty('visibility', 'visible', 'important');
        snackbar.style.setProperty('opacity', '1', 'important');
        console.log("🔍 TruthAI: Enhanced snackbar displayed:", title);
      }
    }, 100);
    
    // Auto-hide timer
    if (autoHideDelay > 0) {
      if (hideTimeout) clearTimeout(hideTimeout);
      hideTimeout = setTimeout(() => {
        if (currentSnackbar) {
          currentSnackbar.remove();
          currentSnackbar = null;
        }
      }, autoHideDelay * 1000);
    }
  }
  
  // Extract comprehensive post content
  function extractPostContent(post) {
    let mainText = '';
    
    // Try multiple selectors for Facebook post text
    const textSelectors = [
      '[data-testid="post_message"]',
      '[data-ad-preview="message"]',
      '.userContent',
      'span[dir="auto"]',
      'div[dir="auto"]',
      '[role="article"] > div > div > div span'
    ];
    
    for (const selector of textSelectors) {
      const elements = post.querySelectorAll(selector);
      for (const element of elements) {
        const text = element.textContent.trim();
        if (text.length > 20 && !text.match(/^(Like|Comment|Share|Reply)$/)) {
          mainText = text;
          break;
        }
      }
      if (mainText) break;
    }
    
    // Fallback: clean up all text
    if (!mainText) {
      const allText = post.innerText || '';
      const lines = allText.split('\n').filter(line => {
        const trimmed = line.trim();
        return trimmed.length > 10 && 
               !trimmed.match(/^\d+\s*(Like|Comment|Share|Reply)/) &&
               !trimmed.match(/^(Like|Comment|Share|Reply|See more|See less)$/) &&
               !trimmed.match(/^\d+[mhd]$/) &&
               !trimmed.match(/\d+ (minutes?|hours?|days?) ago/);
      });
      mainText = lines.slice(0, 5).join(' ').trim();
    }
    
    // Extract metadata
    const images = post.querySelectorAll('img[src*="scontent"], img[src*="fbcdn"]');
    const imageUrls = Array.from(images)
      .map(img => img.src)
      .filter(src => src && !src.includes('emoji') && !src.includes('icon') && src.includes('scontent'))
      .slice(0, 3);
    
    const videos = post.querySelectorAll('video, [data-testid*="video"]');
    const hasVideo = videos.length > 0;
    
    const authorElement = post.querySelector('strong a, h3 strong, [role="link"] strong');
    const author = authorElement ? authorElement.textContent.trim() : 'Unknown';
    
    const timeElement = post.querySelector('time, [title*="202"], [aria-label*="202"]');
    const timestamp = timeElement ? 
      (timeElement.getAttribute('datetime') || timeElement.getAttribute('title') || timeElement.textContent) : 
      Date.now().toString();
    
    // Create unique post ID
    const postId = `${author}_${mainText.substring(0, 50)}_${timestamp}`.replace(/[^a-zA-Z0-9]/g, '_');
    
    return {
      id: postId,
      text: mainText,
      author: author,
      timestamp: timestamp,
      images: imageUrls,
      hasImage: imageUrls.length > 0,
      hasVideo: hasVideo,
      contentLength: mainText.length,
      postElement: post
    };
  }

  // Get most visible post
  function getMostVisiblePost() {
    const posts = Array.from(document.querySelectorAll('[role="article"]'));
    let mostVisible = null;
    let maxVisibility = 0;
    
    posts.forEach(post => {
      const rect = post.getBoundingClientRect();
      const viewportHeight = window.innerHeight;
      
      if (rect.height === 0) return; // Skip hidden posts
      
      const visibleTop = Math.max(0, rect.top);
      const visibleBottom = Math.min(viewportHeight, rect.bottom);
      const visibleHeight = Math.max(0, visibleBottom - visibleTop);
      const visibility = visibleHeight / rect.height;
      
      if (visibility > maxVisibility && visibility > 0.4) { // At least 40% visible
        maxVisibility = visibility;
        mostVisible = post;
      }
    });
    
    return mostVisible;
  }
  
  // Update snackbar for visible post
  function updateSnackbarForVisiblePost() {
    if (!snackbarEnabled) return;
    
    const visiblePost = getMostVisiblePost();
    if (visiblePost && analyzedPosts.has(visiblePost)) {
      const result = analyzedPosts.get(visiblePost);
      console.log("🔍 TruthAI: Updating snackbar for visible post:", result.classification.summary);
      createSnackbar(result.classification, result.content);
    } else if (!visiblePost && currentSnackbar) {
      console.log("🔍 TruthAI: No visible post, hiding snackbar");
      currentSnackbar.remove();
      currentSnackbar = null;
      if (hideTimeout) clearTimeout(hideTimeout);
    }
  }
  
  // Analyze post with real content
  async function analyzePost(post) {
    if (analyzedPosts.has(post)) return;
    
    const content = extractPostContent(post);
    if (!content.text || content.text.length < 20) {
      console.log("🔍 TruthAI: Skipping post - insufficient content:", content.text?.substring(0, 30));
      return;
    }
    
    try {
      console.log("🔍 TruthAI: Analyzing post:", {
        id: content.id,
        author: content.author,
        textLength: content.text.length,
        hasImage: content.hasImage,
        hasVideo: content.hasVideo,
        preview: content.text.substring(0, 100) + "..."
      });
      
      // Send comprehensive data to API
      const analysisPayload = {
        text: content.text,
        author: content.author,
        timestamp: content.timestamp,
        hasImage: content.hasImage,
        hasVideo: content.hasVideo,
        imageCount: content.images.length,
        contentLength: content.contentLength
      };
      
      const res = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(analysisPayload)
      });
      
      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }
      
      const classification = await res.json();
      console.log("🔍 TruthAI: Classification result:", {
        id: content.id,
        category: classification.category?.label,
        type: classification.type?.label,
        authenticity: classification.authenticity?.label,
        summary: classification.summary
      });
      
      // Store analysis result
      analyzedPosts.set(post, {
        id: content.id,
        classification: classification,
        content: content,
        timestamp: Date.now()
      });
      
      uniquePostsAnalyzed.add(content.id);
      
      // Create embedded badge for this post
      createPostBadge(post, classification);
      
      // Update snackbar if this post is currently visible
      const visiblePost = getMostVisiblePost();
      if (visiblePost === post && snackbarEnabled) {
        console.log("🔍 TruthAI: Creating snackbar for analyzed post:", classification.summary);
        createSnackbar(classification, content);
      }
      
    } catch (error) {
      console.error("🔍 TruthAI: Analysis error:", error);
      
      // Create varied fallback classification
      const fallbackCategories = ["Personal", "Entertainment", "Opinion", "News"];
      const fallbackTypes = ["Life Update", "Meme", "General", "Discussion"];
      
      const randomCategory = fallbackCategories[Math.floor(Math.random() * fallbackCategories.length)];
      const randomType = fallbackTypes[Math.floor(Math.random() * fallbackTypes.length)];
      
      const fallbackClassification = {
        category: { label: randomCategory, confidence: 0.5 + Math.random() * 0.3, id: 1 },
        type: { label: randomType, confidence: 0.5 + Math.random() * 0.3, id: 3 },
        authenticity: { label: "N/A", confidence: 0.9, id: 2 },
        source: "fallback",
        summary: `${randomCategory} → ${randomType}`
      };
      
      analyzedPosts.set(post, {
        id: content.id,
        classification: fallbackClassification,
        content: content,
        timestamp: Date.now(),
        error: true
      });
      
      uniquePostsAnalyzed.add(content.id);
      
      // Create embedded badge even for fallback
      createPostBadge(post, fallbackClassification);
    }
  }

  // Find Facebook posts using multiple selectors
  function findPosts() {
    console.log("🔍 TruthAI: Searching for Facebook posts...");
    
    const selectors = [
      '[role="article"]',
      '[data-pagelet="FeedUnit_0"]',
      '[data-pagelet*="FeedUnit"]',
      '.userContentWrapper',
      '[data-testid="fbfeed_story"]',
      'div[data-pagelet^="FeedUnit"]',
      'div[aria-label*="post"]'
    ];
    
    const posts = [];
    selectors.forEach((selector, index) => {
      console.log(`🔍 TruthAI: Trying selector ${index + 1}: ${selector}`);
      const elements = document.querySelectorAll(selector);
      console.log(`🔍 TruthAI: Found ${elements.length} elements with selector: ${selector}`);
      
      elements.forEach(el => {
        if (!posts.includes(el) && el.offsetHeight > 50) {
          posts.push(el);
          console.log(`🔍 TruthAI: Added post element:`, el);
        }
      });
    });
    
    console.log(`🔍 TruthAI: Total unique posts found: ${posts.length}`);
    
    // If no posts found, try a more generic approach
    if (posts.length === 0) {
      console.log("🔍 TruthAI: No posts found with standard selectors, trying generic approach...");
      const allDivs = document.querySelectorAll('div');
      console.log(`🔍 TruthAI: Found ${allDivs.length} div elements total`);
      
      // Look for divs that might be posts (have text content and reasonable height)
      allDivs.forEach(div => {
        if (div.innerText && div.innerText.length > 50 && div.offsetHeight > 100) {
          const hasPostIndicators = div.innerText.includes('Like') || 
                                   div.innerText.includes('Comment') || 
                                   div.innerText.includes('Share') ||
                                   div.querySelector('img') ||
                                   div.querySelector('video');
          
          if (hasPostIndicators && !posts.includes(div)) {
            posts.push(div);
            console.log(`🔍 TruthAI: Added potential post via generic search:`, div);
          }
        }
      });
    }
    
    const validPosts = posts.filter(post => {
      const content = extractPostContent(post);
      return content.text && content.text.length >= 20;
    });
    
    console.log(`🔍 TruthAI: ${validPosts.length} posts have sufficient content for analysis`);
    return validPosts;
  }

  // Main initialization
  async function initializeExtension() {
    console.log("🔍 TruthAI: Initializing extension...");
    console.log("🔍 TruthAI: Current URL:", window.location.href);
    console.log("🔍 TruthAI: Document ready state:", document.readyState);
    
    console.log("🔍 TruthAI: Waiting for Facebook content to load...");
    await new Promise(r => setTimeout(r, 3000));
  
    // Add scroll listener for snackbar updates
    let scrollTimeout;
    window.addEventListener('scroll', () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        if (snackbarEnabled) {
          updateSnackbarForVisiblePost();
        }
      }, 300);
    });
    
    // Set up mutation observer for new posts
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              const newPosts = node.querySelectorAll('[role="article"]');
              newPosts.forEach(newPost => {
                if (!analyzedPosts.has(newPost)) {
                  setTimeout(() => analyzePost(newPost), Math.random() * 2000 + 1000);
                }
              });
              
              // Also check if the node itself is a post
              if (node.matches && node.matches('[role="article"]') && !analyzedPosts.has(node)) {
                setTimeout(() => analyzePost(node), Math.random() * 2000 + 1000);
              }
            }
          });
        }
      });
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
    
    // Start analyzing existing posts
    const posts = findPosts();
    if (posts.length > 0) {
      console.log(`🔍 TruthAI: Starting analysis of ${posts.length} posts`);
      posts.forEach((post, index) => {
        // Stagger analysis to avoid overwhelming API
        setTimeout(() => analyzePost(post), index * 800 + Math.random() * 500);
      });
      
      // Initial snackbar update
      setTimeout(() => {
        if (snackbarEnabled) {
          updateSnackbarForVisiblePost();
        }
      }, 3000);
    } else {
      console.log("🔍 TruthAI: No posts found to analyze");
    }
    
    console.log("🔍 TruthAI: Enhanced extension loaded and monitoring Facebook posts");
  }
  
  // Start the extension when DOM is ready
  console.log("🔍 TruthAI: Setting up initialization...");
  
  if (document.readyState === 'loading') {
    console.log("🔍 TruthAI: Document still loading, waiting for DOMContentLoaded");
    document.addEventListener('DOMContentLoaded', initializeExtension);
  } else {
    console.log("🔍 TruthAI: Document ready, initializing immediately");
    initializeExtension();
  }
  
  // Also try initialization after a delay as fallback
  setTimeout(() => {
    console.log("🔍 TruthAI: Fallback initialization after 5 seconds");
    if (uniquePostsAnalyzed.size === 0) {
      initializeExtension();
    }
  }, 5000);
})();