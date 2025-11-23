let currentImageIndex = 0;
let totalImages = 0;
let annotations = [];
let hasResult = false;
let isDragging = false;
let dragStart = null;
let lastPoint = null;
let currentLabel = 1;
const POINT_INTERVAL = 50;

const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const sourceImg = document.getElementById('sourceImage');
const resultImg = document.getElementById('resultImage');
const runBtn = document.getElementById('runBtn');
const saveBtn = document.getElementById('saveBtn');
const redoBtn = document.getElementById('redoBtn');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const counter = document.getElementById('image-counter');

function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function drawClickPoint(x, y, label) {
    const radius = 10;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = label === 1 ? 'rgba(0, 255, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)';
    ctx.fill();
}

function distance(p1, p2) {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

function addPointAlongPath(x, y, label) {
    if (lastPoint === null) {
        annotations.push({ x, y, label });
        drawClickPoint(x, y, label);
        lastPoint = { x, y };
        return;
    }
    
    const dist = distance(lastPoint, { x, y });
    if (dist >= POINT_INTERVAL) {
        const numPoints = Math.floor(dist / POINT_INTERVAL);
        const dx = (x - lastPoint.x) / numPoints;
        const dy = (y - lastPoint.y) / numPoints;
        
        for (let i = 1; i <= numPoints; i++) {
            const px = lastPoint.x + dx * i;
            const py = lastPoint.y + dy * i;
            annotations.push({ x: px, y: py, label });
            drawClickPoint(px, py, label);
        }
        
        lastPoint = { x, y };
    }
}

function loadCurrentImage() {
    fetch('/api/current-image')
        .then(res => res.json())
        .then(data => {
            currentImageIndex = data.index;
            totalImages = data.total;
            counter.textContent = `Image ${currentImageIndex + 1} of ${totalImages}`;
            
            sourceImg.src = data.image;
            sourceImg.onload = () => {
                canvas.width = sourceImg.width;
                canvas.height = sourceImg.height;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(sourceImg, 0, 0);
                annotations = [];
                hasResult = false;
                resultImg.src = '/static/hero.png';
                updateButtons();
            };
        })
        .catch(err => console.error('Error loading image:', err));
}

function updateButtons() {
    saveBtn.disabled = !hasResult;
}

canvas.addEventListener('mousedown', (e) => {
    const coords = getCanvasCoordinates(e);
    const isRightClick = (e.button === 2);
    currentLabel = isRightClick ? 0 : 1;
    
    isDragging = true;
    dragStart = coords;
    lastPoint = null;
    
    addPointAlongPath(coords.x, coords.y, currentLabel);
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    
    const coords = getCanvasCoordinates(e);
    addPointAlongPath(coords.x, coords.y, currentLabel);
});

canvas.addEventListener('mouseup', (e) => {
    if (isDragging) {
        const coords = getCanvasCoordinates(e);
        addPointAlongPath(coords.x, coords.y, currentLabel);
        isDragging = false;
        lastPoint = null;
    }
});

canvas.addEventListener('mouseleave', (e) => {
    if (isDragging) {
        isDragging = false;
        lastPoint = null;
    }
});

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

runBtn.addEventListener('click', () => {
    if (annotations.length === 0) {
        alert('Please click on the image first');
        return;
    }
    
    runBtn.disabled = true;
    runBtn.textContent = 'Running...';
    
    fetch('/api/segment', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            annotations: annotations,
            imageIndex: currentImageIndex,
            canvasWidth: canvas.width,
            canvasHeight: canvas.height
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            resultImg.src = data.segmented;
            hasResult = true;
            updateButtons();
        }
        runBtn.disabled = false;
        runBtn.textContent = 'Run';
    })
    .catch(err => {
        console.error('Error segmenting:', err);
        runBtn.disabled = false;
        runBtn.textContent = 'Run';
    });
});

saveBtn.addEventListener('click', () => {
    fetch('/api/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({imageIndex: currentImageIndex})
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            alert('Saved!');
        }
    });
});

redoBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(sourceImg, 0, 0);
    annotations = [];
    hasResult = false;
    resultImg.src = '/static/hero.png';
    updateButtons();
});

prevBtn.addEventListener('click', () => {
    fetch('/api/prev', {method: 'POST'})
        .then(res => res.json())
        .then(data => {
            loadCurrentImage();
        });
});

nextBtn.addEventListener('click', () => {
    fetch('/api/next', {method: 'POST'})
        .then(res => res.json())
        .then(data => {
            loadCurrentImage();
        });
});

loadCurrentImage();

