(async function () {
    console.log("🔍 TruthAI: Content script starting...");
    const API = "http://localhost:8000/predict";
    const FEEDBACK_API = "http://localhost:8000/feedback";
    
    // Global variables
    let snackbarEnabled = true;
    let currentSnackbar = null;
    let analyzedPosts = new Map(); // Store post analysis results
    let visiblePosts = new Set(); // Track currently visible posts
    
    // CSS for snackbar
    const style = document.createElement('style');
    style.textContent = `
      .truthai-snackbar {
        position: fixed !important;
        bottom: 20px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        background: rgba(0, 0, 0, 0.9) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 16px 24px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
        z-index: 999999 !important;
        max-width: 500px !important;
        min-width: 300px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 16px !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        transition: all 0.3s ease !important;
      }
      
      .truthai-snackbar.fake {
        border-left: 4px solid #ef4444 !important;
      }
      
      .truthai-snackbar.real {
        border-left: 4px solid #22c55e !important;
      }
      
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
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 6px 12px !important;
        font-size: 12px !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
      }
      
      .truthai-btn:hover {
        background: rgba(255,255,255,0.2) !important;
      }
      
      .truthai-btn.close {
        background: rgba(239,68,68,0.2) !important;
        border-color: rgba(239,68,68,0.3) !important;
      }
    `;
    document.head.appendChild(style);
    
    // Add message listener for popup communication
    let postsAnalyzed = 0;
    
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === "getStats") {
        sendResponse({
          postsAnalyzed: postsAnalyzed,
          isActive: true,
          snackbarEnabled: snackbarEnabled
        });
      } else if (request.action === "toggleSnackbar") {
        snackbarEnabled = request.enabled;
        if (!snackbarEnabled && currentSnackbar) {
          currentSnackbar.remove();
          currentSnackbar = null;
        } else if (snackbarEnabled) {
          updateSnackbarForVisiblePost();
        }
        sendResponse({ success: true, enabled: snackbarEnabled });
      }
    });
    
    // Update posts count when analyzing
    function incrementPostsCount() {
      postsAnalyzed++;
    }
    
    // Create snackbar widget
    function createSnackbar(verdict, postText) {
      if (!snackbarEnabled) return;
      
      // Remove existing snackbar
      if (currentSnackbar) {
        currentSnackbar.remove();
      }
      
      const snackbar = document.createElement("div");
      snackbar.className = `truthai-snackbar ${verdict.label.toLowerCase()}`;
      
      snackbar.innerHTML = `
        <div class="truthai-snackbar-content">
          <span style="font-size: 18px;">${verdict.label === "FAKE" ? "⚠️" : "✅"}</span>
          <div>
            <div style="font-weight: 600; margin-bottom: 2px;">TruthAI: ${verdict.label}</div>
            <div style="font-size: 12px; opacity: 0.8;">Confidence: ${Math.round(verdict.confidence*100)}%</div>
          </div>
        </div>
        <div class="truthai-snackbar-buttons">
          <button class="truthai-btn like">👍</button>
          <button class="truthai-btn dislike">👎</button>
          <button class="truthai-btn close">×</button>
        </div>
      `;
      
      // Add event listeners
      const likeBtn = snackbar.querySelector('.like');
      const dislikeBtn = snackbar.querySelector('.dislike');
      const closeBtn = snackbar.querySelector('.close');
      
      likeBtn.onclick = () => {
        likeBtn.innerHTML = '✅';
        console.log('🔍 TruthAI: User marked as correct');
      };
      
      dislikeBtn.onclick = () => {
        dislikeBtn.innerHTML = '❌';
        console.log('🔍 TruthAI: User marked as wrong');
      };
      
      closeBtn.onclick = () => {
        snackbar.remove();
        currentSnackbar = null;
      };
      
      document.body.appendChild(snackbar);
      currentSnackbar = snackbar;
      
      console.log("🔍 TruthAI: Snackbar created for post");
    }
    
    async function submitFeedback(verdict, postText, isCorrect, feedbackSection) {
      try {
        console.log("🔍 TruthAI: Submitting feedback:", { 
          prediction: verdict.label, 
          confidence: verdict.confidence,
          user_feedback: isCorrect ? "correct" : "incorrect"
        });
        
        feedbackSection.innerHTML = `
          <span style="color: #38a169; font-size: 12px;">
            ✅ Thank you!
          </span>
        `;
        
      } catch (error) {
        console.error("🔍 TruthAI: Error submitting feedback:", error);
        feedbackSection.innerHTML = `
          <span style="color: #e53e3e; font-size: 12px;">
            ❌ Error
          </span>
        `;
      }
    }
  
    function extractPostContent(post) {
      const textContent = post.innerText?.trim() || '';
      const images = post.querySelectorAll('img');
      const imageUrls = Array.from(images)
        .map(img => img.src)
        .filter(src => src && !src.includes('emoji') && !src.includes('icon'))
        .slice(0, 3);
      
      return {
        text: textContent,
        images: imageUrls,
        hasImage: imageUrls.length > 0
      };
    }

    // Check which post is currently most visible
    function getMostVisiblePost() {
      const posts = Array.from(document.querySelectorAll('[role="article"]'));
      let mostVisible = null;
      let maxVisibility = 0;
      
      posts.forEach(post => {
        const rect = post.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        
        // Calculate how much of the post is visible
        const visibleTop = Math.max(0, rect.top);
        const visibleBottom = Math.min(viewportHeight, rect.bottom);
        const visibleHeight = Math.max(0, visibleBottom - visibleTop);
        const visibility = visibleHeight / rect.height;
        
        if (visibility > maxVisibility && visibility > 0.3) { // At least 30% visible
          maxVisibility = visibility;
          mostVisible = post;
        }
      });
      
      return mostVisible;
    }
    
    // Update snackbar for currently visible post
    function updateSnackbarForVisiblePost() {
      if (!snackbarEnabled) return;
      
      const visiblePost = getMostVisiblePost();
      if (visiblePost && analyzedPosts.has(visiblePost)) {
        const result = analyzedPosts.get(visiblePost);
        createSnackbar(result.verdict, result.text);
      }
    }
    
    async function analyzePost(post) {
      // Skip if already analyzed
      if (analyzedPosts.has(post)) return;
      
      const content = extractPostContent(post);
      if (!content.text || content.text.length < 60) return;
      
      try {
        console.log("🔍 TruthAI: Analyzing post:", {
          textLength: content.text.length,
          hasImage: content.hasImage
        });
        
        const res = await fetch(API, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: content.text })
        });
        
        if (!res.ok) {
          console.error("🔍 TruthAI: API error", res.status);
          return;
        }
        
        const verdict = await res.json();
        console.log("🔍 TruthAI: Result:", verdict);
        
        // Store analysis result
        analyzedPosts.set(post, { verdict, text: content.text });
        incrementPostsCount();
        
        // Update snackbar if this is the currently visible post
        updateSnackbarForVisiblePost();
        
      } catch (e) { 
        console.error("🔍 TruthAI: Error:", e);
      }
    }

    // Wait for page load
    if (document.readyState === 'loading') {
      await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve));
    }
    
    // Wait for Facebook content
    console.log("🔍 TruthAI: Waiting for Facebook content to load...");
    await new Promise(r => setTimeout(r, 3000));
    
    // Add scroll listener to update snackbar
    let scrollTimeout;
    window.addEventListener('scroll', () => {
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        updateSnackbarForVisiblePost();
      }, 200);
    });
    
    // Find posts with comprehensive debugging and selectors
    function findPosts() {
      let posts = [];
      console.log("🔍 TruthAI: Starting post detection...");
      
      // Debug: Check what's available on the page
      const allDivs = document.querySelectorAll('div');
      console.log(`🔍 TruthAI: Total divs on page: ${allDivs.length}`);
      
      // Primary selector - Facebook posts with role="article"
      posts = document.querySelectorAll("[role='article']");
      console.log(`🔍 TruthAI: Found ${posts.length} posts with [role='article']`);
      
      if (posts.length === 0) {
        // Secondary selectors for different Facebook layouts
        const selectors = [
          "[data-pagelet^='FeedUnit']",
          "[data-testid*='story']", 
          "[data-testid='story-subtitle']",
          "div[class*='story']",
          "div[data-ad-preview='message']",
          ".userContentWrapper",
          "div[class*='feed'] > div[class*='clearfix']",
          // New 2024 Facebook selectors
          "div[data-pagelet='FeedUnit_0']",
          "div[data-pagelet*='FeedUnit']",
          "div[class*='x1yztbdb']", // Common Facebook post container class
          "div[class*='x1n2onr6']", // Another common container
        ];
        
        for (const selector of selectors) {
          posts = document.querySelectorAll(selector);
          console.log(`🔍 TruthAI: Found ${posts.length} posts with selector: ${selector}`);
          if (posts.length > 0) break;
        }
      }
      
      if (posts.length === 0) {
        console.log("🔍 TruthAI: No posts found with standard selectors, trying advanced detection...");
        
        // Enhanced fallback: find post-like containers with better heuristics
        const potentialPosts = Array.from(allDivs).filter(div => {
          const text = div.textContent?.trim();
          if (!text || text.length < 50) return false;
          
          // Look for Facebook-specific indicators
          const hasReactions = div.querySelector('[aria-label*="Like"], [aria-label*="Comment"], [aria-label*="Share"]');
          const hasUserInfo = div.querySelector('[data-testid*="post"], .userContent, [role="button"]');
          const hasTimestamp = div.querySelector('[data-testid="story-subtitle"], time, [title*="202"]');
          const hasProfileLink = div.querySelector('a[href*="/profile"], a[href*="/user"]');
          
          // Check for common Facebook post patterns
          const textIndicators = /\b(ago|hours?|minutes?|days?|weeks?|months?|years?)\b/i.test(text) ||
                                /\b(Like|Comment|Share)\b/i.test(text);
          
          return text.length > 50 && text.length < 8000 && 
                 (hasReactions || hasUserInfo || hasTimestamp || hasProfileLink || textIndicators) &&
                 !div.querySelector('.truthai-widget'); // Don't re-analyze
        });
        
        console.log(`🔍 TruthAI: Found ${potentialPosts.length} potential posts with advanced detection`);
        posts = potentialPosts.slice(0, 10);
        
        // Debug: Log some sample content
        posts.slice(0, 3).forEach((post, i) => {
          const preview = post.textContent?.trim().substring(0, 100);
          console.log(`🔍 TruthAI: Sample post ${i + 1}: "${preview}..."`);
        });
      }
      
      // Filter out posts that already have widgets
      const filteredPosts = Array.from(posts).filter(post => !post.querySelector('.truthai-widget'));
      console.log(`🔍 TruthAI: Final post count after filtering: ${filteredPosts.length}`);
      
      return filteredPosts;
    }
    
    const posts = findPosts();
    console.log(`🔍 TruthAI: Found ${posts.length} posts`);
    console.log(`🔍 TruthAI: URL: ${window.location.href}`);
    
    // Analyze existing posts
    for (const post of posts) {
      await analyzePost(post);
      await new Promise(r => setTimeout(r, 500)); // throttle
    }
    
    // Monitor for new posts with improved detection
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE && node.querySelector) {
            // Use multiple selectors to catch new posts
            const selectors = [
              '[role="article"]',
              '[data-pagelet^="FeedUnit"]',
              '[data-testid*="story"]',
              'div[class*="story"]'
            ];
            
            let newPosts = [];
            for (const selector of selectors) {
              const found = node.querySelectorAll(selector);
              if (found.length > 0) {
                newPosts = Array.from(found);
                break;
              }
            }
            
            // Also check if the node itself matches our selectors
            for (const selector of selectors) {
              if (node.matches && node.matches(selector) && !node.querySelector('.truthai-widget')) {
                newPosts.push(node);
                break;
              }
            }
            
            newPosts.forEach(newPost => {
              if (!newPost.querySelector('.truthai-widget')) {
                setTimeout(() => analyzePost(newPost), 1000);
              }
            });
          }
        });
      });
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
    
    console.log("🔍 TruthAI: Extension loaded and monitoring");
  })();
  