const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const resultsSection = document.getElementById('results-section');
let chartInstance = null;

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#6366f1';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

async function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert("Please upload a CSV file.");
        return;
    }

    // Show loading state (simple opac change for now)
    dropZone.style.opacity = '0.5';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Analysis failed');

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error(error);
        alert('Error processing file. Ensure the backend is running.');
    } finally {
        dropZone.style.opacity = '1';
    }
}

function displayResults(data) {
    resultsSection.classList.remove('hidden');

    // Update Counts
    document.getElementById('pos-count').textContent = data.summary.positive;
    document.getElementById('neg-count').textContent = data.summary.negative;
    document.getElementById('total-count').textContent = data.total_processed;

    // Render Chart
    renderChart(data.summary.positive, data.summary.negative);

    // Populate Table
    const tbody = document.querySelector('#results-table tbody');
    tbody.innerHTML = '';

    data.detailed_results.forEach(row => {
        const tr = document.createElement('tr');
        const sentimentClass = row.sentiment === 'Positive' ? 'tag-positive' : 'tag-negative';

        tr.innerHTML = `
            <td>${row.name}</td>
            <td>${row.text}</td>
            <td><span class="tag ${sentimentClass}">${row.sentiment}</span></td>
            <td>${(row.score * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
    });

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function renderChart(pos, neg) {
    const ctx = document.getElementById('sentimentChart').getContext('2d');

    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Positive', 'Negative'],
            datasets: [{
                data: [pos, neg],
                backgroundColor: ['#10b981', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#f8fafc' }
                }
            }
        }
    });
}
