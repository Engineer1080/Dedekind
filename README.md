# Fourier Programming Language

![Fourier Studio](https://img.shields.io/badge/Status-Prototype-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

**Fourier** is a modern, high-performance programming language designed specifically for compute-intensive workloads in **Machine Learning** and **Graphics Rendering**.

Unlike general-purpose languages retrofitted with parallel computing capabilities, Fourier is built from the ground up with **GPU/TPU acceleration** and **Automatic Parallelization** as core features.

---

## 🚀 Features

- **Parallel by Default**: Built-in support for vector and matrix operations.
- **Hardware Acceleration**: Use `.gpu()` or `.cpu()` to control execution device via PyTorch.
- **ML Framework**: Native `Dense` and `Sequential` primitives for Neural Networks.
- **Differentiable Programming**: First-class `grad()` keyword and `.with_grad()` support.
- **Tensor Inspection**: Direct property access like `.shape`.
- **Modern IDE**: "Fourier Studio" - v0.2 with resizable terminal and file explorer.
- **Fast Development**: Transpiles directly to highly optimized PyTorch code.

## 🧠 Machine Learning Example

```fourier
fn main() {
    // Define a Neural Network
    model = Sequential([
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    
    // Create data on GPU
    input = [[1.0, 2.0, 3.0]].gpu()
    
    // Run inference
    output = model.forward(input)
    print(output)
}
main()
```

```fourier
fn main() {
    model = Sequential([
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    
    // Daten werden automatisch als Tensor verarbeitet
    input = [[1.0, 2.0, 3.0]].gpu()
    
    print("Prediction:")
    print(model.forward(input))
}
main()
```

## 🏗️ Architecture

The project consists of two main components:

1.  **Fourier Compiler Backend (`src/compiler`)**
    *   Implemented in Python (Prototype Phase).
    *   Transpiles Fourier source code (`.fourier`) into optimized high-performance Python/NumPy code (future target: MLIR/LLVM).
    *   Exposes a REST API (`src/server.py`) for the IDE.

2.  **Fourier Studio (`src/ui`)**
    *   A modern, dark-themed Integrated Development Environment.
    *   Built with **React**, **Vite**, and **Vanilla CSS**.
    *   Features a code editor, real-time console output, and syntax highlighting.

## 🛠️ Installation & Setup

### Prerequisites
*   **Python 3.10+**
*   **Node.js 18+** & `npm`

### 1. Clone the Repository
```bash
git clone https://github.com/Engineer1080/FourierLanguage.git
cd FourierLanguage
```

### 2. Setup Compiler Backend
The backend requires Python and Flask to serve the compiler API.

```bash
# Install dependencies
pip install flask flask-cors numpy

# Run the Backend Server
python src/server.py
```
*The server will start on `http://localhost:5000`.*

### 3. Setup Fourier Studio (Native App)
Open a new terminal window for the frontend.

```bash
cd src/ui

# Install Node dependencies
npm install

# Run the Application (Electron + Vite)
npm run electron:dev
```
*This will launch the native Fourier Studio window.*

## 💻 Usage

1.  Open **Fourier Studio** in your browser.
2.  Write your Fourier code in the editor.
    ```fourier
    fn main() {
        print("Hello, Fourier!")
        
        // Automatic matrix operations
        data = [1.0, 2.0, 3.0, 4.0]
        result = data * 2.0
        
        print("Result:")
        print(result)
    }
    main()
    ```
3.  Click the **▶ Run Code** button.
4.  View the compilation result and program output in the terminal panel.

## 🗺️ Roadmap

### Phase 1: Foundation ✅
*   [x] Language Specification & Design
*   [x] Proof-of-Concept Compiler (Python Backend)
*   [x] Fourier Studio IDE (React Frontend)

### Phase 2: Core Development ✅
*   [x] Build-in Core Algorithms (FFT, Conv, Linalg)
*   [x] Robust Lexer & Parser (Windows Support, Unary Ops)
*   [x] Resizable Studio Terminal & Tabs

### Phase 3: Hardware Acceleration ✅
*   [x] Integration with **PyTorch** for GPU execution.
*   [x] Implementation of `.gpu()` and `.cpu()` modifiers.

### Phase 4: Production (v0.2) ✅
*   [x] **Native Performance**: Integration with MLIR/Inductor via `.fast()`.
*   [x] **MLIR Prototype**: Fourier-Dialect IR generation.
*   [x] **Studio Upgrade**: Resizable terminal and UI polish.

### Phase 5: Advanced Mathematics ✅
*   [x] **Autograd**: Native `grad()` operator for automatic differentiation.
*   [x] **Property Access**: Native `.shape` support for tensors and models.

### Phase 6: Tensor Contraction & Logic (v0.3) 🚧
*   [ ] **Einsum**: High-level elective tensor contraction syntax.
*   [ ] **Complex/Quaternion**: Built-in support for rotational math.
*   [ ] **AOT Compilation**: Static LLVM binary generation.

## 📄 License

This project is licensed under the MIT License.
