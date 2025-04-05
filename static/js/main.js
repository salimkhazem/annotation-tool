// Main Application for Annotation Tool

// Global state
const state = {
    sessionId: null,
    currentImage: null,
    imageList: [],
    currentIndex: 0,
    inputMode: 'box', // 'box', 'positive', 'negative'
    points: {
        positive: [],
        negative: []
    },
    box: null,
    isDrawing: false,
    currentObjectId: 1,
    objectMetadata: {},
    manualCounts: {},
    legends: {},
    currentGame: null,
    classes: [],
    shapes: [],
    values: [],
    startPoint: null,
    canvasDimensions: {
        width: 0,
        height: 0,
        offsetX: 0,
        offsetY: 0,
        imgWidth: 0,
        imgHeight: 0
    }
};

// Canvas references
let imageCanvas, imageCtx, drawingCanvas, drawingCtx;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize canvases
    imageCanvas = document.getElementById('imageCanvas');
    imageCtx = imageCanvas.getContext('2d');
    drawingCanvas = document.getElementById('drawingCanvas');
    drawingCtx = drawingCanvas.getContext('2d', { willReadFrequently: true });

    // Initialize state
    state.canvasDimensions = {
        width: 0,
        height: 0,
        offsetX: 0,
        offsetY: 0,
        imgWidth: 0,
        imgHeight: 0
    };

    // Set up event listeners
    setupEventListeners();
    
    // Populate game select options
    populateGameSelect();
});

function populateGameSelect() {
    // This would be populated from the server's available games
    const gameSelect = document.getElementById('gameSelect');
    
    // Sample games - these should match your Flask app's GAMES dictionary
    const games = [
        "Poker", "Splendor", "Azul", "Concept", "Uno", "Chess", "Other"
    ];
    
    games.forEach(game => {
        const option = document.createElement('option');
        option.value = game;
        option.textContent = game;
        gameSelect.appendChild(option);
    });
    
    // Default to first game
    gameSelect.value = games[0];
}

function setupEventListeners() {
    // Session initialization
    document.getElementById('initSessionBtn').addEventListener('click', initializeSession);
    
    // Image upload
    document.getElementById('uploadBtn').addEventListener('click', () => uploadImages('files'));
    document.getElementById('uploadFolderBtn').addEventListener('click', () => uploadImages('folder'));
    
    // Navigation
    document.getElementById('prevImageBtn').addEventListener('click', () => navigateImages(-1));
    document.getElementById('nextImageBtn').addEventListener('click', () => navigateImages(1));
    document.getElementById('imageSelect').addEventListener('change', handleImageSelection);
    
    // Input mode buttons
    document.getElementById('boxModeBtn').addEventListener('click', () => setInputMode('box'));
    document.getElementById('posModeBtn').addEventListener('click', () => setInputMode('positive'));
    document.getElementById('negModeBtn').addEventListener('click', () => setInputMode('negative'));
    
    // Action buttons
    document.getElementById('segmentBtn').addEventListener('click', runSegmentation);
    document.getElementById('resetObjectBtn').addEventListener('click', resetCurrentObject);
    document.getElementById('newObjectBtn').addEventListener('click', createNewObject);
    document.getElementById('clearAllBtn').addEventListener('click', clearAllAnnotations);
    document.getElementById('manualCountBtn').addEventListener('click', showManualCountModal);
    
    // Legend buttons
    document.getElementById('saveLegendBtn').addEventListener('click', saveLegend);
    document.getElementById('generateLegendBtn').addEventListener('click', generateLegend);
    
    // Export buttons
    document.getElementById('exportMasksBtn').addEventListener('click', exportMasks);
    document.getElementById('exportJsonBtn').addEventListener('click', exportAnnotations);
    document.getElementById('downloadMasksBtn').addEventListener('click', downloadMasks);
    document.getElementById('downloadJsonBtn').addEventListener('click', downloadJson);
    document.getElementById('downloadAllBtn').addEventListener('click', downloadCombined);
    
    // Class selection
    document.getElementById('classSelect').addEventListener('change', handleClassChange);
    
    // Manual count modal
    document.getElementById('saveCountsBtn').addEventListener('click', saveManualCounts);
    
    // Canvas interactions
    drawingCanvas.addEventListener('mousedown', handleMouseDown);
    drawingCanvas.addEventListener('mousemove', handleMouseMove);
    drawingCanvas.addEventListener('mouseup', handleMouseUp);
    
    // Replace the global keyboard shortcut handler with a more focused approach
    document.addEventListener('keydown', handleKeyboardShortcut);
    
    // Game selection
    document.getElementById('gameSelect').addEventListener('change', handleGameChange);
}

// Initialize a new session
async function initializeSession() {
    const model = document.getElementById('modelSelect').value;
    const device = document.getElementById('deviceSelect').value;
    const game = document.getElementById('gameSelect').value;
    
    try {
        // Show loading state
        document.getElementById('sessionStatus').textContent = 'Initializing...';
        document.getElementById('sessionStatus').className = 'badge bg-info';
        
        // Call API to initialize session
        const response = await fetch('/api/init_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model,
                device: device,
                game: game
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Store session ID
            state.sessionId = data.session_id;
            state.currentGame = data.game;
            state.classes = data.classes;
            state.shapes = data.shapes;
            state.values = data.values;
            
            // Update UI
            document.getElementById('sessionStatus').textContent = `Session Active: ${model}`;
            document.getElementById('sessionStatus').className = 'badge bg-success';
            
            // Show upload and tools sections
            document.getElementById('imageUploadSection').style.display = 'block';
            document.getElementById('toolsSection').style.display = 'block';
            document.getElementById('objectPropsSection').style.display = 'block';
            document.getElementById('actionsSection').style.display = 'block';
            document.getElementById('legendSection').style.display = 'block';
            document.getElementById('exportSection').style.display = 'block';
            
            // Hide instruction overlay
            document.getElementById('instructionOverlay').style.display = 'none';
            
            // Populate class and shape selects
            populateSelects();
            
            console.log('Session initialized:', state.sessionId);
        } else {
            alert('Failed to initialize session: ' + data.message);
            document.getElementById('sessionStatus').textContent = 'Initialization Failed';
            document.getElementById('sessionStatus').className = 'badge bg-danger';
        }
    } catch (error) {
        console.error('Error initializing session:', error);
        alert('Error initializing session. Check console for details.');
        document.getElementById('sessionStatus').textContent = 'Error';
        document.getElementById('sessionStatus').className = 'badge bg-danger';
    }
}

// Populate select elements with available options
function populateSelects() {
    // Class select
    const classSelect = document.getElementById('classSelect');
    classSelect.innerHTML = '';
    
    state.classes.forEach(className => {
        const option = document.createElement('option');
        option.value = className;
        option.textContent = className;
        classSelect.appendChild(option);
    });
    
    // Shape select
    const shapeSelect = document.getElementById('shapeSelect');
    shapeSelect.innerHTML = '';
    
    state.shapes.forEach(shape => {
        const option = document.createElement('option');
        option.value = shape;
        option.textContent = shape;
        shapeSelect.appendChild(option);
    });
    
    // Value select (dependent on class)
    populateValueSelect(state.classes[0]);
}

function populateValueSelect(className) {
    const valueSelect = document.getElementById('valueSelect');
    valueSelect.innerHTML = '';
    
    // Get available values for this class
    const values = state.values;
    
    values.forEach(value => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        valueSelect.appendChild(option);
    });
}

// Handle class change
async function handleClassChange() {
    const selectedClass = document.getElementById('classSelect').value;
    
    try {
        // Fetch values for this class
        const response = await fetch(`/api/get_values?session_id=${state.sessionId}&class=${selectedClass}&game=${state.currentGame}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            state.values = data.values;
            populateValueSelect(selectedClass);
        }
    } catch (error) {
        console.error('Error getting class values:', error);
    }
}

// Handle game change
async function handleGameChange() {
    const selectedGame = document.getElementById('gameSelect').value;
    
    if (!state.sessionId) {
        // Just store the selection, it will be used when initializing
        return;
    }
    
    try {
        const response = await fetch('/api/change_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                game: selectedGame
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            state.currentGame = data.game;
            state.classes = data.classes;
            state.values = data.values;
            
            // Update selects
            populateSelects();
        } else {
            alert('Failed to change game: ' + data.message);
        }
    } catch (error) {
        console.error('Error changing game:', error);
        alert('Error changing game. Check console for details.');
    }
}

// Upload images to the server
async function uploadImages(source = 'files') {
    // Get the appropriate file input element based on the source
    const fileInput = source === 'folder' ? 
                      document.getElementById('folderUpload') : 
                      document.getElementById('imageUpload');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        alert(`Please select ${source === 'folder' ? 'a folder containing images' : 'at least one image file'}.`);
        return;
    }
    
    const formData = new FormData();
    formData.append('session_id', state.sessionId);
    formData.append('upload_type', source);
    
    // Filter files - only include image files with allowed extensions
    const allowedExtensions = ['.jpg', '.jpeg', '.png'];
    let fileCount = 0;
    
    for (let i = 0; i < fileInput.files.length; i++) {
        const file = fileInput.files[i];
        const extension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        
        if (allowedExtensions.includes(extension)) {
            formData.append('files', file);
            fileCount++;
        }
    }
    
    if (fileCount === 0) {
        alert('No valid image files found. Supported formats: JPG, JPEG, PNG');
        return;
    }
    
    // Show loading state - use the appropriate button
    const uploadButton = source === 'folder' ? 
                         document.getElementById('uploadFolderBtn') : 
                         document.getElementById('uploadBtn');
    
    uploadButton.disabled = true;
    uploadButton.textContent = 'Uploading...';
    
    try {
        const response = await fetch('/api/upload_images', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log('Images uploaded:', data.count);
            uploadButton.textContent = `${data.count} Images Uploaded`;
            
            // Show navigation section
            document.getElementById('imageNavSection').style.display = 'block';
            
            // Load images into the image select
            loadImageList();
        } else {
            alert('Failed to upload images: ' + data.message);
            uploadButton.textContent = 'Upload Failed';
        }
    } catch (error) {
        console.error('Error uploading images:', error);
        alert('Error uploading images. Check console for details.');
        uploadButton.textContent = 'Error';
    } finally {
        // Reset button states after a delay
        setTimeout(() => {
            document.getElementById('uploadBtn').disabled = false;
            document.getElementById('uploadBtn').textContent = 'Upload Files';
            
            document.getElementById('uploadFolderBtn').disabled = false;
            document.getElementById('uploadFolderBtn').textContent = 'Upload Folder';
        }, 3000);
    }
}

// Load the image list from the server
async function loadImageList() {
    try {
        const response = await fetch(`/api/get_images?session_id=${state.sessionId}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            state.imageList = data.images;
            
            // Populate image select
            const imageSelect = document.getElementById('imageSelect');
            imageSelect.innerHTML = '';
            
            state.imageList.forEach((image, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = image.name;
                imageSelect.appendChild(option);
            });
            
            // Load the first image
            if (state.imageList.length > 0) {
                loadImage(0);
            }
        } else {
            alert('Failed to get images: ' + data.message);
        }
    } catch (error) {
        console.error('Error getting images:', error);
        alert('Error getting images. Check console for details.');
    }
}

// Handle image selection from dropdown
function handleImageSelection() {
    const index = parseInt(document.getElementById('imageSelect').value);
    loadImage(index);
}

// Load an image by index
async function loadImage(index) {
    if (index < 0 || index >= state.imageList.length) {
        return;
    }
    
    state.currentIndex = index;
    
    // Update image select
    document.getElementById('imageSelect').value = index;
    
    try {
        // Request image data from server
        const response = await fetch('/api/change_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                index: index
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Update state
            state.currentImage = {
                path: data.image_path,
                name: data.image_name,
                metadata: data.metadata,
                manual_counts: data.manual_counts,
                legend: data.legend,
                width: data.width,
                height: data.height
            };
            
            // Reset active drawing elements
            state.points.positive = [];
            state.points.negative = [];
            state.box = null;
            state.isDrawing = false;
            
            // Reset current object ID
            state.currentObjectId = 1;
            for (const objId in data.metadata) {
                const id = parseInt(objId);
                state.currentObjectId = Math.max(state.currentObjectId, id + 1);
            }
            
            // Update legend text
            document.getElementById('legendText').value = data.legend || '';
            
            // Update image info in status bar
            document.getElementById('imageInfo').textContent = `Image: ${data.image_name} (${index + 1}/${state.imageList.length})`;
            
            // Load image and mask overlay
            if (data.display_image) {
                loadImageToCanvas(data.display_image, data.width, data.height);
            } else {
                // Load original image (no mask)
                const img = new Image();
                img.onload = () => {
                    resizeCanvasToOriginalSize(img.width, img.height);
                    imageCtx.drawImage(img, 0, 0, img.width, img.height);
                };
                img.src = `/api/image/${encodeURIComponent(data.image_path)}`;
            }
            
            // Update object info
            document.getElementById('objectInfo').textContent = `Object: ${state.currentObjectId} / Mode: ${state.inputMode}`;
        } else {
            alert('Failed to change image: ' + data.message);
        }
    } catch (error) {
        console.error('Error loading image:', error);
        alert('Error loading image. Check console for details.');
    }
}

// Load image data to canvas
function loadImageToCanvas(imgSrc, width, height) {
    const img = new Image();
    img.onload = () => {
        // Set canvas size to exact image dimensions
        setupCanvasAtNativeSize(width, height);
        
        // Draw image at native resolution (1:1 pixel mapping)
        imageCtx.clearRect(0, 0, width, height);
        imageCtx.drawImage(img, 0, 0, width, height);
        
        console.log("Image loaded at native size:", {
            width: width,
            height: height,
            canvasWidth: imageCanvas.width,
            canvasHeight: imageCanvas.height
        });
    };
    img.src = imgSrc;
}

// Set up canvas at the image's native size
function setupCanvasAtNativeSize(imgWidth, imgHeight) {
    // Create wrapper if it doesn't exist
    let wrapper = document.getElementById('canvasWrapper');
    if (!wrapper) {
        wrapper = document.createElement('div');
        wrapper.id = 'canvasWrapper';
        document.getElementById('canvasContainer').appendChild(wrapper);
        
        // Move canvases into wrapper if not already there
        if (imageCanvas.parentElement.id !== 'canvasWrapper') {
            wrapper.appendChild(imageCanvas);
            wrapper.appendChild(drawingCanvas);
        }
    }
    
    // Set canvas dimensions to match image exactly
    imageCanvas.width = imgWidth;
    imageCanvas.height = imgHeight;
    drawingCanvas.width = imgWidth;
    drawingCanvas.height = imgHeight;
    
    // Store dimensions for coordinate conversion
    state.canvasDimensions = {
        width: imgWidth,
        height: imgHeight,
        offsetX: 0,
        offsetY: 0,
        imgWidth: imgWidth,
        imgHeight: imgHeight
    };
    
    console.log("Canvas dimensions set to:", state.canvasDimensions);
}

// Resize canvas to match the original image dimensions
function resizeCanvasToOriginalSize(imgWidth, imgHeight) {
    const container = document.getElementById('canvasContainer');
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    console.log("Container dimensions:", { width: containerWidth, height: containerHeight });
    console.log("Original image dimensions:", { width: imgWidth, height: imgHeight });
    
    // Set canvas to the original image dimensions
    imageCanvas.width = imgWidth;
    imageCanvas.height = imgHeight;
    drawingCanvas.width = imgWidth;
    drawingCanvas.height = imgHeight;
    
    // Store the dimensions for coordinate conversion
    // Since we're using the original size, scaling is 1:1
    state.canvasDimensions = {
        width: imgWidth,
        height: imgHeight,
        drawWidth: imgWidth,
        drawHeight: imgWidth,
        offsetX: 0,
        offsetY: 0,
        imgWidth: imgWidth,
        imgHeight: imgHeight
    };
    
    // Center the canvas in the container if smaller than container
    if (imgWidth < containerWidth) {
        const leftMargin = Math.floor((containerWidth - imgWidth) / 2);
        imageCanvas.style.left = `${leftMargin}px`;
        drawingCanvas.style.left = `${leftMargin}px`;
    } else {
        imageCanvas.style.left = '0';
        drawingCanvas.style.left = '0';
    }
    
    if (imgHeight < containerHeight) {
        const topMargin = Math.floor((containerHeight - imgHeight) / 2);
        imageCanvas.style.top = `${topMargin}px`;
        drawingCanvas.style.top = `${topMargin}px`;
    } else {
        imageCanvas.style.top = '0';
        drawingCanvas.style.top = '0';
    }
    
    console.log("Canvas resized to original dimensions:", state.canvasDimensions);
}

// Set the input mode (box, positive point, negative point)
function setInputMode(mode) {
    state.inputMode = mode;
    
    // Update UI buttons
    document.getElementById('boxModeBtn').classList.remove('active');
    document.getElementById('posModeBtn').classList.remove('active');
    document.getElementById('negModeBtn').classList.remove('active');
    
    switch (mode) {
        case 'box':
            document.getElementById('boxModeBtn').classList.add('active');
            break;
        case 'positive':
            document.getElementById('posModeBtn').classList.add('active');
            break;
        case 'negative':
            document.getElementById('negModeBtn').classList.add('active');
            break;
    }
    
    // Update status bar
    document.getElementById('modeInfo').textContent = `Mode: ${mode.charAt(0).toUpperCase() + mode.slice(1)}`;
}

// Navigate between images
function navigateImages(direction) {
    const newIndex = state.currentIndex + direction;
    if (newIndex >= 0 && newIndex < state.imageList.length) {
        loadImage(newIndex);
    }
}

// Mouse event handlers for drawing
function handleMouseDown(event) {
    if (!state.currentImage) return;
    
    // Get precise mouse position
    const mousePos = getCanvasMousePosition(event);
    
    // Check if the click is within the image boundaries
    if (mousePos.x < 0 || mousePos.x >= state.currentImage.width || 
        mousePos.y < 0 || mousePos.y >= state.currentImage.height) {
        console.log("Click outside image bounds");
        return;
    }
    
    // Convert to image space (1:1 mapping with native size)
    const imagePos = canvasToImageCoords(mousePos.x, mousePos.y);
    
    if (state.inputMode === 'box') {
        // Store raw canvas coordinates for drawing
        state.isDrawing = true;
        state.startPoint = { x: mousePos.x, y: mousePos.y };
        
        // Store image coordinates for the API
        state.box = { 
            x1: imagePos.x, 
            y1: imagePos.y, 
            x2: imagePos.x, 
            y2: imagePos.y 
        };
         
        // Clear existing drawings
        drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        
    } else if (state.inputMode === 'positive' || state.inputMode === 'negative') {
        const type = state.inputMode;
        const color = type === 'positive' ? 'red' : 'green';
        
        // Store point in image coordinates
        if (type === 'positive') {
            state.points.positive.push({ x: imagePos.x, y: imagePos.y });
        } else {
            state.points.negative.push({ x: imagePos.x, y: imagePos.y });
        }
        
        // Draw point using canvas coordinates
        drawPoint(mousePos.x, mousePos.y, color);
        
        // Send to server
        addPoint(type, imagePos.x, imagePos.y).then(() => {
            runSegmentation(true);
        });
    }
}

function handleMouseMove(event) {
    if (!state.isDrawing || state.inputMode !== 'box') return;
    
    // Get precise mouse position
    const mousePos = getCanvasMousePosition(event);
    
    // Update box endpoint in canvas coordinates for drawing
    const imagePos = canvasToImageCoords(mousePos.x, mousePos.y);
    
    // Update image coordinates for API
    state.box.x2 = imagePos.x;
    state.box.y2 = imagePos.y;
    
    // Redraw box
    drawBox(state.startPoint.x, state.startPoint.y, mousePos.x, mousePos.y);
}

function handleMouseUp(event) {
    if (!state.isDrawing || state.inputMode !== 'box') return;
    
    state.isDrawing = false;
    
    // Get precise mouse position
    const mousePos = getCanvasMousePosition(event);
    
    // Update box endpoint in image coordinates
    const imagePos = canvasToImageCoords(mousePos.x, mousePos.y);
    state.box.x2 = imagePos.x;
    state.box.y2 = imagePos.y;
    
    // Send box to server with properly ordered coordinates
    setBox(
        Math.min(state.box.x1, state.box.x2),
        Math.min(state.box.y1, state.box.y2),
        Math.max(state.box.x1, state.box.x2),
        Math.max(state.box.y1, state.box.y2)
    ).then(() => {
        runSegmentation(true);
    });
    
    console.log("Final box:", state.box);
}

// Helper function to get precise mouse position relative to canvas
function getCanvasMousePosition(event) {
    // Get the bounding rectangle of the canvas
    const rect = drawingCanvas.getBoundingClientRect();
    
    // Calculate mouse position relative to the canvas element
    // This factors in page scroll position and canvas position in the viewport
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Calculate the scaling factor between canvas element and canvas coordinate space
    const scaleX = drawingCanvas.width / rect.width;
    const scaleY = drawingCanvas.height / rect.height;
    
    // Apply scaling to get accurate coordinates in the canvas coordinate space
    const canvasX = x * scaleX;
    const canvasY = y * scaleY;
    
    console.log("Mouse position calculation:", {
        mouseEvent: { clientX: event.clientX, clientY: event.clientY },
        canvasBounds: { 
            left: rect.left, 
            top: rect.top, 
            width: rect.width, 
            height: rect.height 
        },
        elementCoords: { x, y },
        canvasScaling: { scaleX, scaleY },
        canvasCoords: { canvasX, canvasY }
    });
    
    return { x: canvasX, y: canvasY };
}

// Convert canvas coordinates to image coordinates
function canvasToImageCoords(canvasX, canvasY) {
    if (!state.currentImage) {
        return { x: 0, y: 0 };
    }
    
    // With native size, coordinates should be the same, just need to clamp to image boundaries
    return {
        x: Math.max(0, Math.min(Math.floor(canvasX), state.currentImage.width - 1)),
        y: Math.max(0, Math.min(Math.floor(canvasY), state.currentImage.height - 1))
    };
}

// Convert image coordinates to canvas coordinates
function imageToCanvasCoords(imageX, imageY) {
    if (!state.currentImage) {
        return { x: 0, y: 0 };
    }
    
    // With original size, simply return the same coordinates
    return {
        x: imageX,
        y: imageY
    };
}

// Draw a point on the canvas
function drawPoint(x, y, color) {
    // Use precise coordinates
    drawingCtx.beginPath();
    drawingCtx.arc(x, y, 5, 0, 2 * Math.PI);
    drawingCtx.fillStyle = color;
    drawingCtx.fill();
    drawingCtx.lineWidth = 2;
    drawingCtx.strokeStyle = 'white';
    drawingCtx.stroke();
}

// Draw a box on the canvas
function drawBox(x1, y1, x2, y2) {
    // Clear previous drawings
    drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    
    // Draw the box with pixel-perfect precision
    drawingCtx.beginPath();
    drawingCtx.rect(
        Math.min(x1, x2),
        Math.min(y1, y2),
        Math.abs(x2 - x1),
        Math.abs(y2 - y1)
    );
    drawingCtx.lineWidth = 2;
    drawingCtx.strokeStyle = 'yellow';
    drawingCtx.setLineDash([5, 5]);
    drawingCtx.stroke();
    drawingCtx.setLineDash([]);
}

// Add a point to the server
async function addPoint(type, x, y) {
    try {
        const response = await fetch('/api/add_point', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                type: type,
                x: x,
                y: y
            }),
        });
        
        const data = await response.json();
        
        if (data.status !== 'success') {
            console.error('Failed to add point:', data.message);
        }
        
        return data;
    } catch (error) {
        console.error('Error adding point:', error);
        throw error;
    }
}

// Set a box in the server
async function setBox(x1, y1, x2, y2) {
    try {
        const response = await fetch('/api/set_box', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                x1: x1,
                y1: y1,
                x2: x2,
                y2: y2
            }),
        });
        
        const data = await response.json();
        
        if (data.status !== 'success') {
            console.error('Failed to set box:', data.message);
        }
        
        return data;
    } catch (error) {
        console.error('Error setting box:', error);
        throw error;
    }
}

// Add color name mapping utility function after the existing constants/initialization
function getColorNameFromHex(hex) {
    const colorMap = {
        "#FF0000": "Red",
        "#FFFF00": "Yellow",
        "#0000FF": "Blue",
        "#800080": "Purple",
        "#FFC0CB": "Pink",
        "#FFFFFF": "White",
        "#808080": "Gray",
        "#000000": "Black",
        "#008000": "Green",
        "#FFA500": "Orange",
        "#00FFFF": "Cyan",
        "#A52A2A": "Brown"
    };
    
    // Return the matching color name or the hex if no match
    return colorMap[hex.toUpperCase()] || hex;
}

// Run segmentation using the current prompts
async function runSegmentation(isPreview = false) {
    if (!state.currentImage) return;
    
    // Get selected properties
    const objectClass = document.getElementById('classSelect').value;
    const objectShape = document.getElementById('shapeSelect').value;
    const objectValue = document.getElementById('valueSelect').value;
    const objectColor = document.getElementById('colorSelect').value;
    const colorName = getColorNameFromHex(objectColor);
    
    try {
        const response = await fetch('/api/run_segmentation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                class: objectClass,
                shape: objectShape,
                value: objectValue,
                color: colorName,       // Send color name
                color_hex: objectColor, // Send hex color code
                is_preview: isPreview
            }),
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Load the updated image with mask
            loadImageToCanvas(data.display_image, state.currentImage.width, state.currentImage.height);
            
            // Only clear drawing canvas and reset points if this is a final segmentation (not a preview)
            if (!isPreview) {
                // Clear drawing canvas but keep the displayed masks
                drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
                
                // Reset points and box for the current object only
                state.points.positive = [];
                state.points.negative = [];
                state.box = null;
            } else {
                // For preview, we keep the drawing canvas content
                // This allows users to continue refining their box or adding more points
            }
            
            // Update object info
            document.getElementById('objectInfo').textContent = `Object: ${data.object_id} / Area: ${data.area || 0}`;
            
            // If this is a final segmentation, store the object's metadata locally
            if (!isPreview && data.metadata) {
                if (!state.objectMetadata[state.currentIndex]) {
                    state.objectMetadata[state.currentIndex] = {};
                }
                state.objectMetadata[state.currentIndex][data.object_id] = data.metadata;
            }
            
            return data;
        } else {
            // Don't show alert for preview failures, just log them
            const errorMsg = 'Segmentation failed: ' + data.message;
            if (!isPreview) {
                alert(errorMsg);
            } else {
                console.error('Preview segmentation failed:', data.message);
            }
            throw new Error(errorMsg);
        }
    } catch (error) {
        console.error('Error running segmentation:', error);
        if (!isPreview) {
            alert('Error running segmentation: ' + (error.message || 'Unknown error'));
        }
        throw error;
    }
}

// Reset the current object annotation
async function resetCurrentObject() {
    if (!state.currentImage) return;
    
    try {
        const response = await fetch('/api/reset_current', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Load the updated image with mask
            loadImageToCanvas(data.display_image, state.currentImage.width, state.currentImage.height);
            
            // Clear drawing canvas
            drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            
            // Reset points and box
            state.points.positive = [];
            state.points.negative = [];
            state.box = null;
        } else {
            alert('Reset failed: ' + data.message);
        }
    } catch (error) {
        console.error('Error resetting object:', error);
        alert('Error resetting object. Check console for details.');
    }
}

// Create a new object annotation
async function createNewObject() {
    if (!state.currentImage) return;
    
    try {
        // First finalize the current segmentation if needed
        let segmentationApplied = false;
        if (state.points.positive.length > 0 || 
            state.points.negative.length > 0 || 
            state.box !== null) {
            try {
                await runSegmentation(false); // Not a preview, finalize the segmentation
                segmentationApplied = true;
            } catch (err) {
                console.warn("Failed to apply segmentation before creating new object:", err);
                // Continue with object creation even if segmentation fails
            }
        }
        
        // Short delay to ensure server has processed the segmentation
        if (segmentationApplied) {
            await new Promise(resolve => setTimeout(resolve, 300));
        }
        
        const response = await fetch('/api/create_new_object', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId
            }),
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Update object ID in state
            state.currentObjectId = data.new_object_id;
            
            // Clear drawing canvas but keep the image with all masks
            drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            
            // Reset points and box for the new object
            state.points.positive = [];
            state.points.negative = [];
            state.box = null;
            
            // Update object info
            document.getElementById('objectInfo').textContent = `Object: ${data.new_object_id} / Mode: ${state.inputMode}`;
            
            console.log("New object created successfully, ID:", data.new_object_id);
        } else {
            console.error('Failed to create new object:', data.message);
            alert('Failed to create new object: ' + data.message);
        }
    } catch (error) {
        console.error('Error creating new object:', error);
        
        // More detailed error message to help debugging
        let errorMessage = 'Error creating new object. ';
        if (error.message) {
            errorMessage += error.message;
        }
        
        // Check if session expired
        if (error.message && error.message.includes('Invalid session')) {
            errorMessage += ' Session may have expired. Please reinitialize.';
            document.getElementById('sessionStatus').textContent = 'Session Expired';
            document.getElementById('sessionStatus').className = 'badge bg-danger';
        }
        
        alert(errorMessage);
    }
}

// Clear all annotations for the current image
async function clearAllAnnotations() {
    if (!state.currentImage) return;
    
    if (!confirm('Are you sure you want to clear all annotations for this image?')) {
        return;
    }
    
    try {
        const response = await fetch('/api/clear_all_annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Load the updated image (no masks)
            loadImageToCanvas(data.display_image, state.currentImage.width, state.currentImage.height);
            
            // Clear drawing canvas
            drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
            
            // Reset points and box
            state.points.positive = [];
            state.points.negative = [];
            state.box = null;
            
            // Reset to object 1
            state.currentObjectId = 1;
            document.getElementById('objectInfo').textContent = `Object: 1 / Mode: ${state.inputMode}`;
        } else {
            alert('Failed to clear annotations: ' + data.message);
        }
    } catch (error) {
        console.error('Error clearing annotations:', error);
        alert('Error clearing annotations. Check console for details.');
    }
}

// Show manual count dialog
function showManualCountModal() {
    if (!state.currentImage) return;
    
    // Generate count fields for each class
    const countContainer = document.getElementById('countContainer');
    countContainer.innerHTML = '';
    
    state.classes.forEach(cls => {
        const countRow = document.createElement('div');
        countRow.className = 'count-row';
        
        const label = document.createElement('label');
        label.textContent = cls;
        countRow.appendChild(label);
        
        const input = document.createElement('input');
        input.type = 'number';
        input.min = '0';
        input.className = 'form-control';
        input.dataset.class = cls;
        
        // Set current count if available
        if (state.currentImage && 
            state.currentImage.manual_counts && 
            state.currentImage.manual_counts[cls] !== undefined) {
            input.value = state.currentImage.manual_counts[cls];
        } else {
            input.value = '0';
        }
        
        countRow.appendChild(input);
        countContainer.appendChild(countRow);
    });
    
    // Show the modal
    const manualCountModal = new bootstrap.Modal(document.getElementById('manualCountModal'));
    manualCountModal.show();
}

// Save manual counts
async function saveManualCounts() {
    if (!state.currentImage) return;
    
    // Collect counts from form
    const counts = {};
    const inputs = document.querySelectorAll('#countContainer input');
    
    inputs.forEach(input => {
        counts[input.dataset.class] = parseInt(input.value) || 0;
    });
    
    try {
        const response = await fetch('/api/set_manual_count', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                counts: counts
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Update state
            if (!state.currentImage.manual_counts) {
                state.currentImage.manual_counts = {};
            }
            state.currentImage.manual_counts = counts;
            
            // Hide modal
            bootstrap.Modal.getInstance(document.getElementById('manualCountModal')).hide();
        } else {
            alert('Failed to save counts: ' + data.message);
        }
    } catch (error) {
        console.error('Error saving counts:', error);
        alert('Error saving counts. Check console for details.');
    }
}

// Save legend
async function saveLegend() {
    if (!state.currentImage) return;
    
    const legend = document.getElementById('legendText').value;
    
    try {
        const response = await fetch('/api/set_legend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                legend: legend
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Update state
            state.currentImage.legend = legend;
            alert('Legend saved successfully.');
        } else {
            alert('Failed to save legend: ' + data.message);
        }
    } catch (error) {
        console.error('Error saving legend:', error);
        alert('Error saving legend. Check console for details.');
    }
}

// Generate legend using AI
async function generateLegend() {
    if (!state.currentImage) return;
    
    try {
        document.getElementById('generateLegendBtn').disabled = true;
        document.getElementById('generateLegendBtn').textContent = 'Generating...';
        
        const response = await fetch('/api/generate_legend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Update text area with generated legend
            document.getElementById('legendText').value = data.legend;
            
            // Update state
            state.currentImage.legend = data.legend;
        } else {
            alert('Failed to generate legend: ' + data.message);
        }
    } catch (error) {
        console.error('Error generating legend:', error);
        alert('Error generating legend. Check console for details.');
    } finally {
        document.getElementById('generateLegendBtn').disabled = false;
        document.getElementById('generateLegendBtn').textContent = 'Generate Legend';
    }
}

// Export masks
async function exportMasks() {
    try {
        const response = await fetch('/api/export_masks', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            alert(`${data.count} masks exported successfully.`);
        } else {
            alert('Failed to export masks: ' + data.message);
        }
    } catch (error) {
        console.error('Error exporting masks:', error);
        alert('Error exporting masks. Check console for details.');
    }
}

// Export annotations
async function exportAnnotations() {
    try {
        const response = await fetch('/api/export_annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            alert(`${data.count} annotation files exported successfully.`);
        } else {
            alert('Failed to export annotations: ' + data.message);
        }
    } catch (error) {
        console.error('Error exporting annotations:', error);
        alert('Error exporting annotations. Check console for details.');
    }
}

// Download annotations as a ZIP file
async function downloadJson() {
    const preserveStructure = document.getElementById('preserveStructureCheck').checked;
    
    try {
        // Show loading state
        const button = document.getElementById('downloadJsonBtn');
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Preparing...';
        
        // Add timestamp to prevent caching issues
        const timestamp = Date.now();
        
        // Request the server to create a downloadable archive
        const response = await fetch(`/api/download_annotations?t=${timestamp}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                preserve_structure: preserveStructure
            }),
        });
        
        // Check for errors in the response
        if (!response.ok) {
            throw new Error(`Server error: ${response.status} - ${response.statusText}`);
        }
        
        // Get the download URL from the response
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log(`Annotations download ready at: ${data.download_url}`);
            
            // Open the download URL in a new window/tab
            // This is more reliable than an iframe or anchor for large downloads
            window.open(data.download_url, '_blank');
            
            // Reset button after a delay
            setTimeout(() => {
                button.disabled = false;
                button.textContent = 'Download JSON';
            }, 1000);
        } else {
            alert('Failed to download annotations: ' + data.message);
            button.disabled = false;
            button.textContent = 'Download JSON';
        }
    } catch (error) {
        console.error('Error downloading annotations:', error);
        alert('Error downloading annotations: ' + (error.message || 'Unknown error'));
        const button = document.getElementById('downloadJsonBtn');
        button.disabled = false;
        button.textContent = 'Download JSON';
    }
}

// Download masks as a ZIP file
async function downloadMasks() {
    const preserveStructure = document.getElementById('preserveStructureCheck').checked;
    
    try {
        // Show loading state
        const button = document.getElementById('downloadMasksBtn');
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Preparing...';
        
        // Add timestamp to prevent caching issues
        const timestamp = Date.now();
        
        // Request the server to create a downloadable archive
        const response = await fetch(`/api/download_masks?t=${timestamp}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                preserve_structure: preserveStructure
            }),
        });
        
        // Check for errors in the response
        if (!response.ok) {
            throw new Error(`Server error: ${response.status} - ${response.statusText}`);
        }
        
        // Get the download URL from the response
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log(`Masks download ready at: ${data.download_url}`);
            
            // Open the download URL in a new window/tab
            // This is more reliable than an iframe or anchor for large downloads
            window.open(data.download_url, '_blank');
            
            // Reset button after a delay
            setTimeout(() => {
                button.disabled = false;
                button.textContent = 'Download Masks';
            }, 1000);
        } else {
            alert('Failed to download masks: ' + data.message);
            button.disabled = false;
            button.textContent = 'Download Masks';
        }
    } catch (error) {
        console.error('Error downloading masks:', error);
        alert('Error downloading masks: ' + (error.message || 'Unknown error'));
        const button = document.getElementById('downloadMasksBtn');
        button.disabled = false;
        button.textContent = 'Download Masks';
    }
}

// Download both annotations and masks in a single request
async function downloadCombined() {
    const preserveStructure = document.getElementById('preserveStructureCheck').checked;
    
    try {
        // Show loading state
        const button = document.getElementById('downloadAllBtn');
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Preparing...';
        
        // Add timestamp to prevent caching issues
        const timestamp = Date.now();
        
        // Request the server to create a combined downloadable archive
        const response = await fetch(`/api/download_combined?t=${timestamp}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: state.sessionId,
                preserve_structure: preserveStructure
            }),
        });
        
        // Check for errors in the response
        if (!response.ok) {
            throw new Error(`Server error: ${response.status} - ${response.statusText}`);
        }
        
        // Get the download URL from the response
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log(`Combined download ready at: ${data.download_url}`);
            console.log(`Contains ${data.annotation_count} annotations and ${data.mask_count} masks`);
            
            // Open the download URL in a new window/tab to start the download
            window.open(data.download_url, '_blank');
            
            // Reset button after a delay
            setTimeout(() => {
                button.disabled = false;
                button.textContent = 'Download All';
            }, 1000);
        } else {
            console.error("Combined download failed:", data.message);
            alert('Failed to download: ' + data.message);
            button.disabled = false;
            button.textContent = 'Download All';
        }
    } catch (error) {
        console.error('Error downloading combined export:', error);
        alert('Error downloading: ' + (error.message || 'Unknown error'));
        const button = document.getElementById('downloadAllBtn');
        button.disabled = false;
        button.textContent = 'Download All';
    }
}

// Handle keyboard shortcuts - modified to ignore form inputs
function handleKeyboardShortcut(event) {
    if (!state.sessionId) return;
    
    // Skip keyboard shortcuts when focus is on input elements
    if (event.target.tagName === 'INPUT' || 
        event.target.tagName === 'TEXTAREA' || 
        event.target.tagName === 'SELECT') {
        return;
    }
    
    switch (event.key) {
        case 'b':
            setInputMode('box');
            break;
        case 'p':
            setInputMode('positive');
            break;
        case 'n':
            setInputMode('negative');
            break;
        case 'r':
            resetCurrentObject();
            break;
        case 'Enter':
            // Only trigger if not inside a form element
            if (event.target.tagName !== 'BUTTON') {
                createNewObject(); // Save current object and create a new one
            }
            break;
        case 'Backspace':
            // Prevent accidental clearing when typing
            if (document.activeElement === document.body) {
                clearAllAnnotations();
            }
            break;
        case 's':
            // Only trigger if not in a text field
            if (!event.ctrlKey) { // Regular 's' without Ctrl
                runSegmentation(false); // Run final segmentation (not preview)
            }
            break;
        case 'ArrowLeft':
            // Only navigate if not editing text
            if (document.activeElement === document.body) {
                navigateImages(-1);
                // Prevent scrolling
                event.preventDefault();
            }
            break;
        case 'ArrowRight':
            // Only navigate if not editing text
            if (document.activeElement === document.body) {
                navigateImages(1);
                // Prevent scrolling
                event.preventDefault();
            }
            break;
    }
    
    // Ctrl+S for save
    if (event.ctrlKey && event.key === 's') {
        event.preventDefault();
        exportAnnotations();
    }
}
