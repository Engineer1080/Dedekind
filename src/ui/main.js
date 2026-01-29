const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

console.log("Electron script started.");

let mainWindow;
let pythonProcess;

function createWindow() {
    console.log("Creating window...");
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false, // For prototype simplicity
        },
        // Use the dark theme background color to avoid white flash
        backgroundColor: '#0f111a',
        autoHideMenuBar: true,
    });

    // In production, load the built file. In dev, load localhost.
    const startUrl = process.env.ELECTRON_START_URL || 'http://localhost:5173';
    mainWindow.loadURL(startUrl);

    mainWindow.on('closed', function () {
        mainWindow = null;
    });
}

function startPythonBackend() {
    // Path to the python executable and script
    // Assuming running from root project dir context or src/ui depending on launch
    // We'll aim to run this from the project root for dev consistency

    const scriptPath = path.join(__dirname, '../../src/server.py');

    // Spawn python process
    // Using 'python' command - requirements: python in PATH and deps installed
    pythonProcess = spawn('python', [scriptPath]);

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Backend: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Backend Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Backend exited with code ${code}`);
    });
}

app.on('ready', () => {
    startPythonBackend();
    // Give python a second to start (simple approach) or wait-on in scripts
    setTimeout(createWindow, 1000);
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

app.on('activate', function () {
    if (mainWindow === null) {
        createWindow();
    }
});
