const API = "http://localhost:8000/predict"; // change to your deployed URL

document.getElementById("check").onclick = async () => {
  const text = document.getElementById("text").value.trim();
  if (!text) return;
  const res = await fetch(API, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text})
  });
  const data = await res.json();
  document.getElementById("out").textContent = JSON.stringify(data, null, 2);
};
