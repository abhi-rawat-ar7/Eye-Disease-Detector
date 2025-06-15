document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const fileNameSpan = document.getElementById('fileName');
    const previewImage = document.getElementById('previewImage');
    const predictButton = document.getElementById('predictButton');
    const imagePreviewContainer = document.getElementById('imagePreview');
    const resultArea = document.getElementById('resultArea');
    const statusMessage = document.getElementById('statusMessage');
    const predictedDisease = document.getElementById('predictedDisease');
    const confidenceScore = document.getElementById('confidenceScore');
    const allProbabilitiesList = document.getElementById('probabilitiesList');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorBox = document.getElementById('errorBox');
    const errorMessage = document.getElementById('errorMessage');
    const closeErrorButton = document.getElementById('closeError');

    let selectedFile = null;

    // --- Event Listeners ---

    imageUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            fileNameSpan.textContent = selectedFile.name;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
                imagePreviewContainer.style.border = 'none'; // Remove dashed border when image is present
            };
            reader.readAsDataURL(selectedFile);
            predictButton.disabled = false; // Enable predict button
            hideResultAndError();
        } else {
            fileNameSpan.textContent = 'No file chosen';
            previewImage.src = '';
            previewImage.style.display = 'none';
            imagePreviewContainer.style.border = '2px dashed #e0e0e0'; // Restore dashed border
            predictButton.disabled = true; // Disable predict button
            hideResultAndError();
        }
    });

    predictButton.addEventListener('click', async () => {
        if (!selectedFile) {
            showError("Please select an image file first.");
            return;
        }

        hideResultAndError();
        loadingIndicator.style.display = 'flex'; // Show loading spinner

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // IMPORTANT: Ensure this URL matches your FastAPI backend address
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                body: formData,
            });

            loadingIndicator.style.display = 'none'; // Hide loading spinner

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            loadingIndicator.style.display = 'none'; // Hide loading spinner in case of error
            showError(`Error during prediction: ${error.message}`);
            console.error('Prediction Error:', error);
        }
    });

    closeErrorButton.addEventListener('click', () => {
        errorBox.style.display = 'none';
    });

    // --- Helper Functions ---

    function displayResults(data) {
        resultArea.style.display = 'block';
        statusMessage.textContent = 'Success';
        statusMessage.style.color = '#4CAF50'; // Green for success

        predictedDisease.textContent = data.prediction;
        confidenceScore.textContent = `${(parseFloat(data.confidence) * 100).toFixed(2)}%`;

        // Clear previous probabilities
        allProbabilitiesList.innerHTML = '';

        // Display all probabilities
        for (const [disease, probability] of Object.entries(data.all_probabilities)) {
            const listItem = document.createElement('li');
            listItem.innerHTML = `<span>${disease}:</span> <span>${(probability * 100).toFixed(2)}%</span>`;
            allProbabilitiesList.appendChild(listItem);
        }
    }

    function showError(message) {
        errorBox.style.display = 'block';
        errorMessage.textContent = message;
    }

    function hideResultAndError() {
        resultArea.style.display = 'none';
        errorBox.style.display = 'none';
    }
});
