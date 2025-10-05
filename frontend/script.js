document.addEventListener('DOMContentLoaded', () => {
    const tickerInput = document.getElementById('tickerInput');
    const predictBtn = document.getElementById('predictBtn');
    const periodButtons = document.querySelectorAll('.period-btn');
    const predictionText = document.getElementById('predictionText');
    const evaluationText = document.getElementById('evaluationText');
    const stockChartCanvas = document.getElementById('stockChart');

    const API_BASE_URL = 'http://127.0.0.1:8000/api'; // Adjust if your backend runs on a different port
    let stockChart;

    const fetchChartData = async (ticker, period) => {
        try {
            const response = await fetch(`${API_BASE_URL}/chart/${period}/?ticker=${ticker}`);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            updateChart(data);
        } catch (error) {
            console.error('Failed to fetch chart data:', error);
            alert('Failed to load chart data. Please check the ticker and try again.');
        }
    };

    const fetchPrediction = async (ticker) => {
        try {
            const response = await fetch(`${API_BASE_URL}/predict/?ticker=${ticker}`);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            const { predicted_open: predictedOpen, predicted_close: predictedClose } = data;
            predictionText.textContent = `Open: ${predictedOpen.toFixed(2)} | Close: ${predictedClose.toFixed(2)}`;
        } catch (error) {
            console.error('Failed to fetch prediction:', error);
            predictionText.textContent = 'Error';
        }
    };

    const fetchEvaluation = async (ticker) => {
        try {
            const response = await fetch(`${API_BASE_URL}/evaluate/?ticker=${ticker}`);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            // openR2 = data[0]
            // closeR2 = data[1]
            // evaluationText.textContent = `Open: ${openR2.toFixed(2)} | Close: ${closeR2.toFixed(2)}`;
            evaluationText.textContent = `${data}`;

        } catch (error) {
            console.error('Failed to fetch evaluation:', error);
            evaluationText.textContent = 'Error';
        }
    };

    const updateChart = (data) => {
        const labels = data.map(item => new Date(item.Date).toLocaleDateString());
        const closePrices = data.map(item => item.Close);

        if (stockChart) {
            stockChart.destroy();
        }

        stockChart = new Chart(stockChartCanvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Close Price',
                    data: closePrices,
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    tension: 0.1,
                    fill: true,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Price (USD)'
                        }
                    }
                }
            }
        });
    };

    const loadData = () => {
        const ticker = tickerInput.value.trim().toUpperCase();
        if (!ticker) {
            alert('Please enter a stock ticker.');
            return;
        }
        const activePeriod = document.querySelector('.period-btn.active').dataset.period;
        
        fetchChartData(ticker, activePeriod);
        fetchPrediction(ticker);
        fetchEvaluation(ticker);
    };

    predictBtn.addEventListener('click', loadData);

    periodButtons.forEach(button => {
        button.addEventListener('click', () => {
            periodButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            loadData();
        });
    });

    // Initial load
    loadData();
});
