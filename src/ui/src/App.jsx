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
  const [output, setOutput] = useState('Fourier Studio v0.2 Ready...')
  const [isRunning, setIsRunning] = useState(false)
  const [fileContents, setFileContents] = useState({});
  const editorRef = useRef(null);
  const highlightRef = useRef(null);
  const lineNumbersRef = useRef(null);
  const editorWrapperRef = useRef(null);
  const isResizingRef = useRef(false);
  const isResizingSidebarRef = useRef(false);

  const [expandedFolders, setExpandedFolders] = useState({});
  const [sidebarWidth, setSidebarWidth] = useState(260);
  const [consoleHeight, setConsoleHeight] = useState(200);
  const [isResizing, setIsResizing] = useState(false);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);
  const [isBuilding, setIsBuilding] = useState(false);
  const [plots, setPlots] = useState([]);

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
        setPlots(data.plots || []);
      } else {
        setPlots([]);
        setOutput(`Error:\n${data.error}\n\nTraceback:\n${data.traceback}`);
      }
    } catch (err) {
      setOutput('Failed to connect to Compiler Backend.\nIs server.py running?');
      setPlots([]);
    }
    setIsRunning(false);
  };

  const handleBuild = async () => {
    if (!activeTabId) return;

    setIsBuilding(true);
    setOutput('Starting AOT Compilation (MLIR/LLVM)...');
    try {
      const response = await fetch('http://localhost:5000/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: fileContents[activeTabId] }),
      });
      const data = await response.json();
      if (data.status === 'success') {
        setOutput(`Build Success!\nArtifacts:\n- ${data.path}\n\n${data.message}`);
      } else {
        setOutput(`Build Error:\n${data.error}\n\nTraceback:\n${data.traceback}`);
      }
    } catch (err) {
      setOutput('Failed to connect to AOT Build Backend.');
    }
    setIsBuilding(false);
  };

  const handleScroll = (e) => {
    /* Syntax-Highlight liegt im <pre>, nicht im <code> – das <pre> ist das scrollbare Element */
    const highlightPre = highlightRef.current?.parentElement;
    if (highlightPre) {
      highlightPre.scrollTop = e.target.scrollTop;
      highlightPre.scrollLeft = e.target.scrollLeft;
    }
    if (lineNumbersRef.current) {
      lineNumbersRef.current.scrollTop = e.target.scrollTop;
    }
  };

  /* Mausrad über Zeilennummern: Scrollen an die Textarea weiterleiten */
  const handleLineNumbersWheel = (e) => {
    if (editorRef.current) {
      editorRef.current.scrollTop += e.deltaY;
      editorRef.current.scrollLeft += e.deltaX;
      handleScroll({ target: editorRef.current });
    }
  };

  /* Mausrad über dem Code-Bereich: Scrollen erzwingen (falls Textarea das Rad nicht verarbeitet) */
  const handleEditorWheel = (e) => {
    if (!editorRef.current) return;
    const el = editorRef.current;
    const { scrollTop, scrollLeft, scrollHeight, scrollWidth, clientHeight, clientWidth } = el;
    const maxScrollTop = scrollHeight - clientHeight;
    const maxScrollLeft = scrollWidth - clientWidth;
    let didScroll = false;
    if (maxScrollTop > 0 && e.deltaY !== 0) {
      el.scrollTop = Math.max(0, Math.min(maxScrollTop, scrollTop + e.deltaY));
      handleScroll({ target: el });
      didScroll = true;
    }
    if (maxScrollLeft > 0 && e.deltaX !== 0) {
      el.scrollLeft = Math.max(0, Math.min(maxScrollLeft, scrollLeft + e.deltaX));
      handleScroll({ target: el });
      didScroll = true;
    }
    if (didScroll || maxScrollTop > 0 || maxScrollLeft > 0) e.preventDefault();
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

  /* Mausrad-Scroll im Editor: Listener mit passive:false (damit preventDefault wirkt) */
  useEffect(() => {
    const wrapper = editorWrapperRef.current;
    if (!wrapper) return;
    const handleWheel = (e) => {
      const el = editorRef.current;
      if (!el) return;
      const { scrollTop, scrollLeft, scrollHeight, scrollWidth, clientHeight, clientWidth } = el;
      const maxScrollTop = scrollHeight - clientHeight;
      const maxScrollLeft = scrollWidth - clientWidth;
      if (maxScrollTop > 0 && e.deltaY !== 0) {
        el.scrollTop = Math.max(0, Math.min(maxScrollTop, scrollTop + e.deltaY));
        const highlightPre = highlightRef.current?.parentElement;
        if (highlightPre) {
          highlightPre.scrollTop = el.scrollTop;
          highlightPre.scrollLeft = el.scrollLeft;
        }
        if (lineNumbersRef.current) lineNumbersRef.current.scrollTop = el.scrollTop;
        e.preventDefault();
      }
      if (maxScrollLeft > 0 && e.deltaX !== 0) {
        el.scrollLeft = Math.max(0, Math.min(maxScrollLeft, scrollLeft + e.deltaX));
        const highlightPre = highlightRef.current?.parentElement;
        if (highlightPre) highlightPre.scrollLeft = el.scrollLeft;
        e.preventDefault();
      }
    };
    wrapper.addEventListener('wheel', handleWheel, { passive: false });
    return () => wrapper.removeEventListener('wheel', handleWheel);
  }, [activeTabId]);

  // Terminal-Resizer (Editor / Output)
  useEffect(() => {
    isResizingRef.current = isResizing;
  }, [isResizing]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizingRef.current) return;
      const newHeight = window.innerHeight - e.clientY;
      if (newHeight > 50 && newHeight < window.innerHeight * 0.8) {
        setConsoleHeight(newHeight);
      }
    };

    const handleMouseUp = () => {
      isResizingRef.current = false;
      setIsResizing(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    if (isResizing) {
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // Sidebar-Resizer (Projektstruktur / Editor)
  useEffect(() => {
    isResizingSidebarRef.current = isResizingSidebar;
  }, [isResizingSidebar]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizingSidebarRef.current) return;
      const w = Math.max(150, Math.min(500, e.clientX));
      setSidebarWidth(w);
    };

    const handleMouseUp = () => {
      isResizingSidebarRef.current = false;
      setIsResizingSidebar(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    if (isResizingSidebar) {
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizingSidebar]);

  const activeContent = activeTabId ? fileContents[activeTabId] : '';

  return (
    <div className="ide-container">
      <div className="sidebar" style={{ width: sidebarWidth }}>
        <div className="sidebar-header">FOURIER</div>
        <div className="explorer-section">
          <div className="explorer-title">Project Explorer</div>
          {projectRoot && getFolderItems(projectRoot).map(file => (
            <FileTreeItem key={file.path} file={file} />
          ))}
        </div>
      </div>

      <div
        className={`resizer-vertical ${isResizingSidebar ? 'active' : ''}`}
        onMouseDown={(e) => {
          e.preventDefault();
          e.stopPropagation();
          isResizingSidebarRef.current = true;
          setIsResizingSidebar(true);
        }}
        role="separator"
        aria-label="Projektstruktur-Breite anpassen"
      />

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
          <button className="btn btn-primary" onClick={handleRun} disabled={isRunning || isBuilding || !activeTabId}>
            {isRunning ? 'Running...' : '▶ Run active file'}
          </button>
          <button className="btn btn-accent" onClick={handleBuild} disabled={isRunning || isBuilding || !activeTabId}>
            {isBuilding ? 'Building...' : '⚙ Build Native (AOT)'}
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
            <div className="editor-with-linenumbers">
              <div
                ref={lineNumbersRef}
                className="line-numbers"
                aria-hidden="true"
                onWheel={handleLineNumbersWheel}
              >
                {(activeContent.split('\n').length || 1) >= 1
                  ? Array.from({ length: Math.max(1, activeContent.split('\n').length) }, (_, i) => i + 1).map((n) => (
                      <div key={n} className="line-number">{n}</div>
                    ))
                  : <div className="line-number">1</div>}
              </div>
              <div className="editor-wrapper" ref={editorWrapperRef} onWheel={handleEditorWheel}>
                <div className="editor-content-area">
                  <pre className="code-highlight">
                    <code ref={highlightRef} className="language-fourier">
                      {activeContent}
                    </code>
                  </pre>
                  <textarea
                    ref={editorRef}
                    className="code-textarea"
                    value={activeContent}
                    onChange={(e) => updateContent(e.target.value)}
                    onScroll={handleScroll}
                    spellCheck="false"
                  />
                </div>
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#666' }}>
              Select a file to edit
            </div>
          )}
        </div>

        <div
          className={`resizer ${isResizing ? 'active' : ''}`}
          onMouseDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
            isResizingRef.current = true;
            setIsResizing(true);
          }}
          role="separator"
          aria-label="Output-Bereich in der Höhe anpassen"
        ></div>

        {plots.length > 0 && (
          <div className="plots-panel">
            <div className="plots-header">Plots</div>
            <div className="plots-grid">
              {plots.map((base64, i) => (
                <div key={i} className="plot-item">
                  <img src={`data:image/png;base64,${base64}`} alt={`Plot ${i + 1}`} />
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="console-container" style={{ height: `${consoleHeight}px` }}>
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
