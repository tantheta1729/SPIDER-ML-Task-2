document.getElementById('submit-btn').addEventListener('click', sendQuery);
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') sendQuery();
});

async function sendQuery() {
    const inputField = document.getElementById('user-input');
    const outputBox = document.getElementById('output-box');
    const safetyBadge = document.getElementById('safety-badge');
    const confidenceBadge = document.getElementById('confidence-badge');

    const question = inputField.value.trim();
    if (!question) return;

    // Set Loading Visual State
    outputBox.innerHTML = '<p style="color: #94a3b8;">🔍 Accessing secure medical vectors...</p>';
    inputField.value = '';

    try {
        // Trigger HTTP request post straight to our FastAPI backend server
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();
        
        // 🚨 THE TRAP: Print exactly what Python sent back to the F12 Console
        console.log("PYTHON SENT THIS BACK:", data);
        
        // If Python crashed and sent an error detail instead of an answer, display it!
        if (data.detail) {
             outputBox.innerHTML = `<p style="color: #ef4444; font-weight: bold;">⚠️ Backend Python Error: ${data.detail}</p>`;
             return; // Stop executing the rest of the code
        }

        // Update Interface status indicators dynamically
        if (data.confidence === "Blocked") {
            safetyBadge.textContent = "Safety: 🛑 BLOCKED";
            safetyBadge.style.backgroundColor = "#ef4444";
            
            confidenceBadge.textContent = "Confidence: None";
            confidenceBadge.style.backgroundColor = "#334155";
            
            outputBox.innerHTML = `<p style="color: #ef4444; font-weight: bold;">${data.reply}</p>`;
        } else {
            safetyBadge.textContent = "Safety: ✅ PASSED";
            safetyBadge.style.backgroundColor = "#22c55e";
            
            confidenceBadge.textContent = `Confidence: ${data.confidence}`;
            confidenceBadge.style.backgroundColor = data.confidence === "High" ? "#22c55e" : "#eab308";
            
            outputBox.innerHTML = `<p style="white-space: pre-wrap;">${data.reply}</p>`;
        }

    } catch (error) {
        outputBox.innerHTML = '<p style="color: #ef4444;">System Connection Failure. Check if the Python server is running.</p>';
        console.error("Network or parsing error:", error);
    }
}