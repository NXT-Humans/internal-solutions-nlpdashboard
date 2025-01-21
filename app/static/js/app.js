document.addEventListener('DOMContentLoaded', function() {
    // File upload form handling
    const uploadForm = document.querySelector('form');
    const fileInput = document.querySelector('input[type="file"]');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const file = fileInput.files[0];
            if (file && !file.name.toLowerCase().endsWith('.zip')) {
                e.preventDefault();
                alert('Please select a ZIP file');
                return;
            }
            
            // Show loading state
            const submitButton = uploadForm.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.textContent = 'Uploading...';
            }
        });
    }

    // File input styling and validation
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                if (!file.name.toLowerCase().endsWith('.zip')) {
                    alert('Please select a ZIP file');
                    this.value = '';
                }
            }
        });
    }
});