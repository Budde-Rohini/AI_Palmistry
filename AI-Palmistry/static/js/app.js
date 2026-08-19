// Main frontend interaction script for AI Palmistry
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeBtn = document.getElementById('removeBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const errorAlert = document.getElementById('errorAlert');
    const errorMessage = document.getElementById('errorMessage');

    function showError(msg) {
        if (errorAlert && errorMessage) {
            errorMessage.textContent = msg;
            errorAlert.classList.remove('d-none');
            errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            alert(msg);
        }
    }

    function hideError() {
        if (errorAlert) {
            errorAlert.classList.add('d-none');
        }
    }

    if (dropzone && fileInput) {
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        if (removeBtn) {
            removeBtn.addEventListener('click', () => {
                fileInput.value = '';
                previewContainer.classList.add('d-none');
                dropzone.classList.remove('d-none');
                hideError();
                if (analyzeBtn) analyzeBtn.disabled = true;
            });
        }
    }

    function handleFileSelect(file) {
        hideError();
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            showError('Invalid file type. Please upload a JPG, PNG, or WEBP image.');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showError('File size exceeds maximum allowed limit of 10 MB.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('d-none');
            dropzone.classList.add('d-none');
            if (analyzeBtn) analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideError();

            if (!fileInput || !fileInput.files.length) {
                showError('Please select a palm image to upload.');
                return;
            }

            const formData = new FormData();
            formData.append('palm_image', fileInput.files[0]);

            // Disable button & show spinner state
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span> Validating & Uploading...';

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    window.location.href = data.redirect_url;
                } else {
                    showError(data.error || 'Upload failed. Please check your image and try again.');
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles me-2"></i> Analyze Palm';
                }
            } catch (err) {
                showError('Network or server error occurred during upload. Please try again.');
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles me-2"></i> Analyze Palm';
            }
        });
    }
});

