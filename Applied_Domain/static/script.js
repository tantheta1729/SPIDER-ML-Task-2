document.addEventListener("DOMContentLoaded", () => {
    const chatWindow = document.getElementById("chat-window");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const exportBtn = document.getElementById("export-btn");

    // Holds conversation history for multi-turn context
    let chatHistory = [];

    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        // Clear welcome card on first message
        const welcomeCard = document.querySelector(".welcome-card");
        if (welcomeCard) welcomeCard.remove();

        // 1. Render User Message
        appendUserMessage(query);
        userInput.value = "";

        // 2. Prepare Assistant Message Container
        const { messageDiv, bubbleDiv, metaDiv } = createAssistantMessageContainer();

        try {
            // 3. Initiate SSE Post Stream
            const response = await fetch("/api/stream-query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    chat_history: chatHistory
                })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let assistantFullText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\n\n");

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const payload = JSON.parse(line.replace("data: ", ""));

                        // Feature 1 & 2: Emergency Alert
                        if (payload.type === "emergency") {
                            bubbleDiv.classList.add("emergency-alert");
                            bubbleDiv.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${payload.content}`;
                            assistantFullText = payload.content;
                            break;
                        }

                        // Feature 1 & 2: Metadata Render (Confidence Meter & Source Badges)
                        if (payload.type === "metadata") {
                            renderMetadata(metaDiv, payload.confidence, payload.sources);
                        }

                        // Stream LLM Tokens
                        if (payload.type === "token") {
                            assistantFullText += payload.content;
                            bubbleDiv.textContent = assistantFullText;
                            chatWindow.scrollTop = chatWindow.scrollHeight;
                        }

                        // Insufficient evidence error
                        if (payload.type === "insufficient_evidence") {
                            bubbleDiv.textContent = payload.content;
                            assistantFullText = payload.content;
                        }
                    }
                }
            }

            // Update chat history array for multi-turn follow-up queries
            chatHistory.push({ role: "user", parts: query });
            chatHistory.push({ role: "model", parts: assistantFullText });

        } catch (err) {
            console.error("Streaming error:", err);
            bubbleDiv.textContent = "❌ Error connecting to server. Please try again.";
        }
    });

    // --- Helper UI Functions ---

    function appendUserMessage(text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message user";
        msgDiv.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
        chatWindow.appendChild(msgDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function createAssistantMessageContainer() {
        const messageDiv = document.createElement("div");
        messageDiv.className = "message assistant";

        const metaDiv = document.createElement("div");
        metaDiv.className = "message-meta";

        const bubbleDiv = document.createElement("div");
        bubbleDiv.className = "message-bubble";
        bubbleDiv.textContent = "Thinking & analyzing guidelines...";

        messageDiv.appendChild(metaDiv);
        messageDiv.appendChild(bubbleDiv);
        chatWindow.appendChild(messageDiv);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        return { messageDiv, bubbleDiv, metaDiv };
    }

    // Renders Features 1 & 2 (Confidence meter + Source badges)
    function renderMetadata(container, confidence, sources) {
        container.innerHTML = "";

        // Confidence meter
        const confPercent = Math.round(confidence * 100);
        const confClass = confPercent >= 80 ? "confidence-high" : "confidence-med";
        const confLabel = confPercent >= 80 ? "High Evidence Match" : "Moderate Evidence Match";

        const confDiv = document.createElement("div");
        confDiv.className = "confidence-bar";
        confDiv.innerHTML = `
            <span>Evidence Score:</span>
            <span class="confidence-badge ${confClass}">${confLabel} (${confPercent}%)</span>
        `;
        container.appendChild(confDiv);

        // Sources badges
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement("div");
            sourcesDiv.className = "sources-container";
            sources.forEach(src => {
                const badge = document.createElement("span");
                badge.className = "source-badge";
                badge.innerHTML = `<i class="fa-solid fa-book-medical"></i> ${escapeHtml(src)}`;
                sourcesDiv.appendChild(badge);
            });
            container.appendChild(sourcesDiv);
        }
    }

    // Feature 3: Export Clinical Summary Report
    exportBtn.addEventListener("click", () => {
        if (chatHistory.length === 0) {
            alert("No conversation history to export yet!");
            return;
        }

        let reportContent = "=========================================\n";
        reportContent += "   CLINICAL CONSULTATION SUMMARY REPORT  \n";
        reportContent += `   Generated: ${new Date().toLocaleString()}\n`;
        reportContent += "=========================================\n\n";

        chatHistory.forEach((msg, idx) => {
            const speaker = msg.role === "user" ? "PATIENT / USER" : "CLINICAL ASSISTANT";
            reportContent += `[${speaker}]:\n${msg.parts}\n\n`;
            reportContent += "-----------------------------------------\n";
        });

        const blob = new Blob([reportContent], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `clinical_summary_${Date.now()}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    });

    function escapeHtml(str) {
        return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
});