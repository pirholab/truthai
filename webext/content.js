(async function () {
    const API = "http://localhost:8000/predict"; // set your prod URL
    function labelNode(node, verdict) {
      const badge = document.createElement("div");
      badge.textContent = verdict.label + " (" + Math.round(verdict.confidence*100) + "%)";
      badge.style.cssText = `
        position: relative; display:inline-block; margin:4px 0; padding:2px 6px;
        border-radius: 6px; font-weight: 600; font-size: 12px;
        background:${verdict.label === "FAKE" ? "#ffdddd" : "#ddffdd"};
        color:#111; border:1px solid #888;
      `;
      node.prepend(badge);
    }
  
    // very naive selector for demo; you will refine
    const posts = document.querySelectorAll("[role='article']");
    for (const post of posts) {
      const text = post.innerText?.trim();
      if (!text || text.length < 60) continue; // skip short
      try {
        console.log("TruthAI: Analyzing post text:", text.substring(0, 100) + "...");
        const res = await fetch(API, {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({text})
        });
        
        if (!res.ok) {
          console.error("TruthAI: API error", res.status, res.statusText);
          return;
        }
        
        const verdict = await res.json();
        console.log("TruthAI: Got verdict:", verdict);
        labelNode(post, verdict);
        // throttle to avoid spamming API
        await new Promise(r => setTimeout(r, 300));
      } catch (e) { 
        console.error("TruthAI: Error analyzing post:", e);
      }
    }
  })();
  