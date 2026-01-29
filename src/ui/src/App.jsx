import { useState, useEffect, useRef } from 'react'
import Prism from 'prismjs'
import './fourier-syntax'
import './index.css'

// Using Node.js modules via Electron integration
const fs = window.require('fs');
const path = window.require('path');

function App() {
  const [projectRoot, setProjectRoot] = useState('');
  const [files, setFiles] = useState([]);
  const [openTabs, setOpenTabs] = useState([]);
  const [activeTabId, setActiveTabId] = useState(null);
  const [output, setOutput] = useState('Fourier Studio v0.1 Ready...')
  const [isRunning, setIsRunning] = useState(false)
  const [fileContents, setFileContents] = useState({});
  const editorRef = useRef(null);
  const highlightRef = useRef(null);

  const [expandedFolders, setExpandedFolders] = useState({});

  useEffect(() => {
    // Initial setup: Find project root
    const root = path.join(window.process.cwd(), '..', '..');
    setProjectRoot(root);
  }, []);

  const toggleFolder = (folderPath) => {
    setExpandedFolders(prev => ({
      ...prev,
      [folderPath]: !prev[folderPath]
    }));
  };

  const openFile = (file) => {
    if (file.isDir) {
      toggleFolder(file.path);
      return;
    }

    if (!openTabs.find(t => t.path === file.path)) {
      try {
        const content = fs.readFileSync(file.path, 'utf8');
        setFileContents(prev => ({ ...prev, [file.path]: content }));
        setOpenTabs(prev => [...prev, file]);
      } catch (err) {
        console.error("Failed to read file:", err);
        return;
      }
    }
    setActiveTabId(file.path);
  };

  const getFolderItems = (dir) => {
    try {
      const items = fs.readdirSync(dir);
      return items
        .filter(item => !item.startsWith('.') && item !== 'node_modules' && item !== 'dist')
        .map(item => {
          const fullPath = path.join(dir, item);
          const stats = fs.statSync(fullPath);
          return {
            name: item,
            path: fullPath,
            isDir: stats.isDirectory()
          };
        })
        .sort((a, b) => b.isDir - a.isDir || a.name.localeCompare(b.name));
    } catch (err) {
      console.error("Failed to read dir:", dir, err);
      return [];
    }
  };

  const FileTreeItem = ({ file, level = 0 }) => {
    const isExpanded = expandedFolders[file.path];
    const isActive = activeTabId === file.path;

    return (
      <div className="file-tree-node">
        <div
          className={`file-item ${isActive ? 'active' : ''}`}
          style={{ paddingLeft: `${20 + level * 12}px` }}
          onClick={() => openFile(file)}
        >
          <span className="file-icon">
            {file.isDir ? (isExpanded ? '📂' : '📁') : '📄'}
          </span>
          <span className="file-name">{file.name}</span>
        </div>
        {file.isDir && isExpanded && (
          <div className="folder-children">
            {getFolderItems(file.path).map(child => (
              <FileTreeItem key={child.path} file={child} level={level + 1} />
            ))}
          </div>
        )}
      </div>
    );
  };

  const closeTab = (e, filePath) => {
    e.stopPropagation();
    const newTabs = openTabs.filter(t => t.path !== filePath);
    setOpenTabs(newTabs);
    if (activeTabId === filePath) {
      setActiveTabId(newTabs.length > 0 ? newTabs[newTabs.length - 1].path : null);
    }
  };

  const handleRun = async () => {
    if (!activeTabId) return;

    setIsRunning(true);
    setOutput('Compiling and Running...');
    try {
      const response = await fetch('http://localhost:5000/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: fileContents[activeTabId] }),
      });
      const data = await response.json();
      if (data.status === 'success') {
        setOutput(data.output);
      } else {
        setOutput(`Error:\n${data.error}\n\nTraceback:\n${data.traceback}`);
      }
    } catch (err) {
      setOutput('Failed to connect to Compiler Backend.\nIs server.py running?');
    }
    setIsRunning(false);
  }

  const handleScroll = (e) => {
    if (highlightRef.current) {
      highlightRef.current.scrollTop = e.target.scrollTop;
      highlightRef.current.scrollLeft = e.target.scrollLeft;
    }
  };

  const updateContent = (content) => {
    setFileContents(prev => ({ ...prev, [activeTabId]: content }));
    // Auto-save? For now just keep in memory
  };

  useEffect(() => {
    if (activeTabId && highlightRef.current) {
      Prism.highlightElement(highlightRef.current);
    }
  }, [activeTabId, fileContents]);

  const activeContent = activeTabId ? fileContents[activeTabId] : '';

  return (
    <div className="ide-container">
      <div className="sidebar">
        <div className="sidebar-header">FOURIER</div>
        <div className="explorer-section">
          <div className="explorer-title">Project Explorer</div>
          {projectRoot && getFolderItems(projectRoot).map(file => (
            <FileTreeItem key={file.path} file={file} />
          ))}
        </div>
      </div>

      <div className="main-area">
        <div className="tabs-bar">
          {openTabs.map(tab => (
            <div
              key={tab.path}
              className={`tab ${activeTabId === tab.path ? 'active' : ''}`}
              onClick={() => setActiveTabId(tab.path)}
            >
              {tab.name}
              <span className="tab-close" onClick={(e) => closeTab(e, tab.path)}>×</span>
            </div>
          ))}
        </div>

        <div className="toolbar">
          <button className="btn btn-primary" onClick={handleRun} disabled={isRunning || !activeTabId}>
            {isRunning ? 'Running...' : '▶ Run active file'}
          </button>
          <button className="btn" onClick={() => {
            if (activeTabId) fs.writeFileSync(activeTabId, fileContents[activeTabId]);
            setOutput("File saved.");
          }}>
            💾 Save
          </button>
        </div>

        <div className="editor-container">
          {activeTabId ? (
            <div className="editor-wrapper">
              <textarea
                className="code-textarea"
                value={activeContent}
                onChange={(e) => updateContent(e.target.value)}
                onScroll={handleScroll}
                spellCheck="false"
              />
              <pre className="code-highlight">
                <code ref={highlightRef} className="language-fourier">
                  {activeContent}
                </code>
              </pre>
            </div>
          ) : (
            <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
              Select a file to edit
            </div>
          )}
        </div>

        <div className="console-container">
          <div className="console-header">
            <span>Terminal Output</span>
            <button style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer' }} onClick={() => setOutput('')}>Clear</button>
          </div>
          <div className="console-output">
            {output}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
