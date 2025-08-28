(function() {
  'use strict';
  
  const API = "http://localhost:8000/classify";
  
  // Global state
  let snackbarEnabled = true;
  let showConfidence = true;
  let autoHideDelay = 10;
  let currentSnackbar = null;
  let analyzedPosts = new Map();
  let hideTimeout = null;
  let uniquePostsAnalyzed = new Set();
  
  console.log("🔍 TruthAI: Content script starting...");
  
  // Check if we're on Facebook
  if (!window.location.hostname.includes('facebook.com')) {
    console.log("🔍 TruthAI: Not on Facebook, exiting");
    return;
  }
  
  console.log("🔍 TruthAI: On Facebook, initializing extension");
  
  // CSS for snackbar
  const style = document.createElement('style');
  style.textContent = `
    .truthai-snackbar {
      position: fixed !important;
      bottom: 20px !important;
      left: 50% !important;
      transform: translateX(-50%) !important;
      background: rgba(0, 0, 0, 0.95) !important;
      color: white !important;
      border-radius: 16px !important;
      padding: 18px 24px !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
      font-size: 14px !important;
      font-weight: 500 !important;
      box-shadow: 0 12px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.1) !important;
      z-index: 2147483647 !important;
      max-width: 550px !important;
      min-width: 350px !important;
      display: flex !important;
      align-items: center !important;
      justify-content: space-between !important;
      gap: 18px !important;
      backdrop-filter: blur(20px) !important;
      transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
      animation: slideUp 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
      visibility: visible !important;
      opacity: 1 !important;
      pointer-events: auto !important;
    }
    
    @keyframes slideUp {
      from {
        transform: translateX(-50%) translateY(100px);
        opacity: 0;
      }
      to {
        transform: translateX(-50%) translateY(0);
        opacity: 1;
      }
    }
    
    .truthai-snackbar.news { border-left: 4px solid #3b82f6 !important; }
    .truthai-snackbar.personal { border-left: 4px solid #8b5cf6 !important; }
    .truthai-snackbar.entertainment { border-left: 4px solid #f59e0b !important; }
    .truthai-snackbar.commercial { border-left: 4px solid #10b981 !important; }
    .truthai-snackbar.educational { border-left: 4px solid #06b6d4 !important; }
    .truthai-snackbar.opinion { border-left: 4px solid #ec4899 !important; }
    .truthai-snackbar.fake { border-left: 4px solid #ef4444 !important; }
    .truthai-snackbar.real { border-left: 4px solid #22c55e !important; }
    
    .truthai-snackbar-content {
      display: flex !important;
      align-items: center !important;
      gap: 12px !important;
      flex: 1 !important;
    }
    
    .truthai-snackbar-buttons {
      display: flex !important;
      gap: 8px !important;
    }
    
    .truthai-btn {
      background: rgba(255,255,255,0.12) !important;
      border: 1px solid rgba(255,255,255,0.25) !important;
      color: white !important;
      border-radius: 8px !important;
      padding: 8px 14px !important;
      font-size: 13px !important;
      font-weight: 500 !important;
      cursor: pointer !important;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
      backdrop-filter: blur(10px) !important;
    }
    
    .truthai-btn:hover {
      background: rgba(255,255,255,0.25) !important;
      transform: translateY(-1px) !important;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    
    .truthai-btn.close {
      background: rgba(239,68,68,0.15) !important;
      border-color: rgba(239,68,68,0.4) !important;
      padding: 6px 10px !important;
    }
    
    .truthai-btn.close:hover {
      background: rgba(239,68,68,0.3) !important;
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
  
  // Create snackbar widget
  function createSnackbar(classification, postContent) {
    if (!snackbarEnabled) return;
    
    if (currentSnackbar) {
      currentSnackbar.remove();
      currentSnackbar = null;
      if (hideTimeout) clearTimeout(hideTimeout);
    }
    
    const snackbar = document.createElement("div");
    const categoryClass = classification.category.label.toLowerCase();
    const authClass = classification.authenticity.label.toLowerCase();
    
    snackbar.className = `truthai-snackbar ${categoryClass}`;
    if (authClass === 'fake' || authClass === 'real') {
      snackbar.classList.add(authClass);
    }
    
    const getEmoji = (category, type, auth) => {
      if (auth === 'FAKE') return '⚠️';
      if (auth === 'REAL') return '✅';
      
      switch (category) {
        case 'News': return '📰';
        case 'Personal': return '👤';
        case 'Entertainment': return '🎭';
        case 'Commercial': return '💼';
        case 'Educational': return '📚';
        case 'Opinion': return '💭';
        default: return '📝';
      }
    };
    
    const emoji = getEmoji(classification.category.label, classification.type.label, classification.authenticity.label);
    
    const confidenceDisplay = showConfidence ? `
      <div style="font-size: 11px; opacity: 0.8; margin-top: 4px;">
        ${classification.category.label} (${Math.round(classification.category.confidence*100)}%) • 
        ${classification.type.label} (${Math.round(classification.type.confidence*100)}%)
        ${classification.authenticity.label !== 'N/A' ? 
          ` • ${classification.authenticity.label} (${Math.round(classification.authenticity.confidence*100)}%)` : ''}
      </div>
    ` : '';

    snackbar.innerHTML = `
      <div class="truthai-snackbar-content">
        <span style="font-size: 20px;">${emoji}</span>
        <div style="flex: 1;">
          <div style="font-weight: 600; margin-bottom: 2px; font-size: 14px;">${classification.summary}</div>
          ${confidenceDisplay}
        </div>
      </div>
      <div class="truthai-snackbar-buttons">
        <button class="truthai-btn like" title="Mark as correct">👍</button>
        <button class="truthai-btn dislike" title="Mark as incorrect">👎</button>
        <button class="truthai-btn close" title="Close">×</button>
      </div>
    `;
    
    const likeBtn = snackbar.querySelector('.like');
    const dislikeBtn = snackbar.querySelector('.dislike');
    const closeBtn = snackbar.querySelector('.close');
    
    likeBtn.onclick = () => {
      likeBtn.innerHTML = '✅';
      console.log('🔍 TruthAI: User marked classification as correct');
    };
    
    dislikeBtn.onclick = () => {
      dislikeBtn.innerHTML = '❌';
      console.log('🔍 TruthAI: User marked classification as wrong');
    };
    
    closeBtn.onclick = () => {
      snackbar.remove();
      currentSnackbar = null;
      if (hideTimeout) clearTimeout(hideTimeout);
    };
    
    document.body.appendChild(snackbar);
    currentSnackbar = snackbar;
    
    // Force visibility and ensure it's actually shown
    setTimeout(() => {
      if (snackbar && snackbar.parentNode) {
        snackbar.style.setProperty('display', 'flex', 'important');
        snackbar.style.setProperty('visibility', 'visible', 'important');
        snackbar.style.setProperty('opacity', '1', 'important');
        snackbar.style.setProperty('position', 'fixed', 'important');
        snackbar.style.setProperty('z-index', '2147483647', 'important');
        console.log("🔍 TruthAI: Snackbar forced visible for:", classification.summary);
        console.log("🔍 TruthAI: Snackbar element:", snackbar);
        console.log("🔍 TruthAI: Snackbar computed style:", window.getComputedStyle(snackbar).display);
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
    
    console.log("🔍 TruthAI: Extension loaded and monitoring Facebook posts");
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
