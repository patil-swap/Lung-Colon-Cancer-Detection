const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileName = document.getElementById('file-name');
const runAnalysisBtn = document.getElementById('run-analysis');
const previewCard = document.getElementById('preview-card');
const previewImage = document.getElementById('preview-image');
const previewFilename = document.getElementById('preview-filename');
const removeImageBtn = document.getElementById('remove-image');
const resultCard = document.getElementById('result-card');
const resultContent = document.getElementById('result-content');
const cascadeCard = document.getElementById('cascade-card');
const cascadeStages = document.getElementById('cascade-stages');
const loadRandomBtn = document.getElementById('load-random');
const randomGrid = document.getElementById('random-grid');

let selectedFile = null;
let selectedDatasetPath = null;
let selectedImageUrl = null;

function resetState() {
    selectedFile = null;
    selectedDatasetPath = null;
    selectedImageUrl = null;
    fileInput.value = '';
    fileName.textContent = '';
    runAnalysisBtn.disabled = true;
    previewCard.hidden = true;
    resultCard.hidden = true;
    cascadeCard.hidden = true;
    randomGrid.querySelectorAll('img').forEach(img => img.classList.remove('selected'));
}

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

dropzone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }
    selectedFile = file;
    selectedDatasetPath = null;
    selectedImageUrl = URL.createObjectURL(file);
    fileName.textContent = file.name;
    previewImage.src = selectedImageUrl;
    previewFilename.textContent = file.name;
    previewCard.hidden = false;
    runAnalysisBtn.disabled = false;
    resultCard.hidden = true;
    cascadeCard.hidden = true;
}

removeImageBtn.addEventListener('click', resetState);

loadRandomBtn.addEventListener('click', async () => {
    try {
        const response = await fetch('/random');
        const data = await response.json();
        renderRandomImages(data.images);
    } catch (err) {
        showError('Could not load reference samples.');
    }
});

function renderRandomImages(images) {
    randomGrid.innerHTML = '';
    images.forEach((relativePath) => {
        const img = document.createElement('img');
        img.src = `/dataset_image/${relativePath}`;
        img.alt = 'Histopathology sample';
        img.loading = 'lazy';
        img.addEventListener('click', () => selectRandomImage(relativePath));
        randomGrid.appendChild(img);
    });
}

function selectRandomImage(datasetPath) {
    selectedDatasetPath = datasetPath;
    selectedFile = null;
    selectedImageUrl = `/dataset_image/${datasetPath}`;
    fileName.textContent = datasetPath.split('/').pop();
    previewImage.src = selectedImageUrl;
    previewFilename.textContent = fileName.textContent;
    previewCard.hidden = false;
    runAnalysisBtn.disabled = false;
    resultCard.hidden = true;
    cascadeCard.hidden = true;
    randomGrid.querySelectorAll('img').forEach(img => img.classList.remove('selected'));
    // Add selected class to the clicked image
    const selectedImg = Array.from(randomGrid.querySelectorAll('img')).find(
        img => img.src.endsWith(datasetPath)
    );
    if (selectedImg) selectedImg.classList.add('selected');
}

runAnalysisBtn.addEventListener('click', async () => {
    if (!selectedFile && !selectedDatasetPath) return;
    runAnalysisBtn.disabled = true;
    showLoading();
    try {
        const formData = new FormData();
        if (selectedFile) {
            formData.append('file', selectedFile);
        } else if (selectedDatasetPath) {
            formData.append('dataset_path', selectedDatasetPath);
        }
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (data.valid) {
            showResult(data);
        } else {
            showInvalid(data.message);
        }
    } catch (err) {
        showError('An error occurred while analyzing the image.');
    } finally {
        runAnalysisBtn.disabled = false;
    }
});

function showLoading() {
    resultCard.hidden = false;
    cascadeCard.hidden = true;
    resultContent.innerHTML = `
        <div class="loading-indicator">
            <p>Analyzing image...</p>
            <div class="spinner"></div>
        </div>
    `;
}

function showResult(data) {
    resultCard.hidden = false;
    cascadeCard.hidden = false;

    // Parse prediction label to cascade stages
    const stages = parseCascadeFromLabel(data.prediction);

    // Render result
    const resultClass = data.prediction.includes('Benign') ? 'benign' :
        data.prediction.includes('Malignant') ? 'malignant' : 'subtype';
    resultContent.innerHTML = `
        <div class="result-label ${resultClass}">${data.prediction}</div>
        <div class="confidence-wrapper">
            <div class="confidence-ring" style="background: conic-gradient(var(--teal) ${Math.round(data.confidence * 360)}deg, #E2E8F0 0deg);">
                <span>${Math.round(data.confidence * 100)}%</span>
            </div>
            <div class="confidence-text">
                <div>Model Confidence</div>
                <div style="font-weight:600; font-size:1.1rem;">${Math.round(data.confidence * 100)}%</div>
            </div>
        </div>
        <div style="margin-top:0.5rem; color:var(--slate-light); font-size:0.85rem;">
            Histopathology detector confidence: ${Math.round(data.detector_confidence * 100)}%
        </div>
    `;

    // Render cascade stages
    cascadeStages.innerHTML = stages.map(stage => `
        <div class="stage completed">
            <span class="stage-icon">✓</span>
            <span class="stage-title">${stage.title}</span>
            <span class="stage-value">${stage.value}</span>
        </div>
    `).join('');
}

function showInvalid(message) {
    resultCard.hidden = false;
    cascadeCard.hidden = true;
    resultContent.innerHTML = `
        <div class="result-label malignant" style="font-size:1.2rem;">${message}</div>
    `;
}

function showError(message) {
    resultCard.hidden = false;
    cascadeCard.hidden = true;
    resultContent.innerHTML = `
        <div class="result-label malignant" style="font-size:1.2rem;">${message}</div>
    `;
}

function parseCascadeFromLabel(label) {
    // Example labels:
    // "Lung: Benign"
    // "Lung: Malignant - ACA"
    // "Lung: Malignant - SCC"
    // "Colon: Benign"
    // "Colon: Malignant"
    const parts = label.split(': ');
    const organ = parts[0];
    const rest = parts[1];
    let diseaseState = '';
    let subtype = 'N/A';

    if (rest.includes('Benign')) {
        diseaseState = 'Benign';
    } else if (rest.includes('Malignant')) {
        diseaseState = 'Malignant';
        if (rest.includes('ACA')) subtype = 'Adenocarcinoma';
        else if (rest.includes('SCC')) subtype = 'Squamous Cell Carcinoma';
    }

    return [
        { title: 'Organ', value: organ },
        { title: 'Disease State', value: diseaseState },
        { title: 'Subtype', value: subtype }
    ];
}
