import { useState } from 'react'
import './index.css'

function App() {
  const [code, setCode] = useState(`fn main() {
    print("Welcome to Fourier Studio v0.1")
    
    // Matrix Operation on GPU
    vec = [1.0, 2.0, 3.0]
    print("Vector:")
    print(vec)
    
    val = 100
    print("Calculation:")
    print(val * 2)
}

main()`)
  const [output, setOutput] = useState('Ready...')
  const [isRunning, setIsRunning] = useState(false)

  const handleRun = async () => {
    setIsRunning(true)
    setOutput('Compiling and Running...')
    try {
      const response = await fetch('http://localhost:5000/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ source: code }),
      })
      const data = await response.json()
      if (data.status === 'success') {
        setOutput(data.output)
      } else {
        setOutput(`Error:\n${data.error}\n\nTraceback:\n${data.traceback}`)
      }
    } catch (err) {
      setOutput('Failed to connect to Compiler Backend.\nIs server.py running?')
    }
    setIsRunning(false)
  }

  return (
    <div className="ide-container">
      <div className="sidebar">
        <h2>FOURIER</h2>
        <div style={{ marginTop: '20px', fontSize: '0.9rem', color: '#888' }}>
          <div>EXPLORER</div>
          <div style={{ marginTop: '10px', color: '#ccc', paddingLeft: '10px', borderLeft: '2px solid #646cff' }}>
            main.fourier
          </div>
          <div style={{ marginTop: '5px', color: '#666', paddingLeft: '10px' }}>
            lib.fourier
          </div>
        </div>
      </div>

      <div className="main-area">
        <div className="toolbar">
          <button className="btn btn-primary" onClick={handleRun} disabled={isRunning}>
            {isRunning ? 'Running...' : '▶ Run Code'}
          </button>
          <button className="btn">
            Save
          </button>
        </div>

        <div className="editor-container">
          <textarea
            className="code-editor"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            spellCheck="false"
          />
        </div>

        <div className="console-container">
          <div className="console-header">Terminal Output</div>
          <div className="console-output">
            {output}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
