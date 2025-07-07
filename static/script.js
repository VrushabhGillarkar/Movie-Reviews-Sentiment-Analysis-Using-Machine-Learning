let typingTimer;
const debounceDelay = 300; // 300ms delay after user stops typing

$('#reviewInput').on('input', function () {
    clearTimeout(typingTimer);
    const text = $(this).val();

    typingTimer = setTimeout(() => {
        if (text.trim() === "") {
            $('#feedback').html("Waiting for input...");
            return;
        }

        $.ajax({
            url: "/predict_live",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({ text: text }),
            success: function (response) {
                let sentiment = response.sentiment;
                let confidence = response.confidence;

                let emoji = sentiment === "positive" ? "😊" : sentiment === "negative" ? "😞" : "😐";
                $('#feedback').html(
                    `Sentiment: <strong>${sentiment.toUpperCase()}</strong> ${emoji}<br>Confidence: ${confidence}%`
                );
            },
            error: function () {
                $('#feedback').html("Error in sentiment prediction.");
            }
        });
    }, debounceDelay);
});
