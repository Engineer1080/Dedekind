# Fourier Programming Language

![Version](https://img.shields.io/badge/Version-0.6.0-blue) ![Fourier Studio](https://img.shields.io/badge/Status-Prototype-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

**Fourier** is a modern, high-performance programming language designed specifically for compute-intensive workloads in **Machine Learning** and **Graphics Rendering**.

Unlike general-purpose languages retrofitted with parallel computing capabilities, Fourier is built from the ground up with **GPU/TPU acceleration** and **Automatic Parallelization** as core features.

---

- **Ricci Calculus**: Native index notation (`A^ij * B_jk`) for Einstein summation.
- **Sparse Tensors**: Efficient `.sparse()` support for FEM/CFD simulations.
- **Fundamental Constants**: Native access to `c`, `G`, `h`, `k_B`, and `k_e` as **Quantity** with SI units (e.g. `c` in m/s).
- **Physical Units**: Literals with units (`10[m]`, `5[m/s]`, `1.0[kg]`); add/sub require same unit; multiply/divide combine units; `^` for powers (`r^2`); display simplified to J, N where applicable.
- **4D Rotational Math**: Native Quaternion support (`i`, `j`, `k` suffixes) and `.rotate()` method.
- **AOT Compilation**: Truly native binary generation via MLIR and LLVM.
- **Modern IDE**: "Fourier Studio" – v0.6 with resizable terminal and file explorer.

### What's New in v0.6
- **Physical Units (Option B)**: Constants `c`, `G`, `h`, `k_B`, `k_e` are now `Quantity` values with SI units; expressions like `m * c^2` and `G * m1 * m2 / r^2` yield results with correct dimensions; output simplified to **J** (Joule) and **N** (Newton) where applicable.
- **Quantity**: Full arithmetic including `__pow__` (e.g. `c^2`, `r^2`) and `__neg__`; unary minus for literals and Quaternions fixed in codegen.
- **Quaternion**: `__neg__` support so expressions such as `-1.0 + 0i` and signal lists with negative imaginary parts work correctly (e.g. `signal_physics.fourier`).

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
The backend requires Python, Flask, and PyTorch (for tensor/ML runtime).

```bash
# Install dependencies (ein Befehl)
pip install -r requirements.txt

# Run the Backend Server
python src/server.py
```
*The server will start on `http://localhost:5000`.*

**Abhängigkeiten im Überblick:** `flask`, `flask-cors` (API), `torch` (PyTorch für Tensoren, FFT, ML), `matplotlib` (für `plot()`-Visualisierung).

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

### Examples
Example programs are in `examples/fourier/`, including:
- `hello.fourier` – basic I/O and tensors  
- `universal_constants.fourier` – physical constants and units (E = mc², gravitation, Coulomb)  
- `signal_physics.fourier` – complex numbers (Quaternions) and FFT  
- `conditional_logic.fourier`, `basic_loops.fourier` – control flow  
- `mnist_classifier.fourier` – neural network with `Sequential`/`Dense`  

From the `src/` directory: `python -m compiler.compiler ../examples/fourier/hello.fourier`

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

### Phase 6: Tensor Contraction & Logic (v0.3) ✅
*   [x] **Einsum**: High-level elective tensor contraction syntax.
*   [x] **Complex/Quaternion**: Built-in support for rotational math.

### Phase 8: AOT Compilation & LLVM Backend ✅
*   [x] **Static Binary**: Standalone `.exe` generation without Python.
*   [x] **MLIR Pipeline**: Fourier -> MLIR -> LLVM -> Binary.
*   [x] **Verification**: Native binary execution on Windows.

### Phase 9: Ricci Calculus & Universal Constants ✅
*   [x] **Index Notation**: Support for `^` and `_` suffixes.
*   [x] **Auto-Einsum**: Lowering Ricci expressions to `torch.einsum`.
*   [x] **Physics Constants**: Native high-precision constants (`c`, `G`, etc.).

### Phase 10: Sparse Tensors & CFD ✅
*   [x] **Sparse API**: `.sparse()` modifier for COO/CSR formats.
*   [x] **Item Assignment**: `T[i][j] = val` for grid manipulation.
*   [x] **CFD Simulation**: Heat diffusion on 10,000 node grids.

### Phase 11: Quaternion & Rotational Math ✅
*   [x] **Hamilton Notation**: Support for `i`, `j`, `k` quaternion components.
*   [x] **Hamilton Product**: Native 4D arithmetic.
*   [x] **Robotics Support**: Native `.rotate(vector)` method.

### Phase 12: Physical Units & Constants (v0.6) ✅
*   [x] **Constants as Quantity**: `c`, `G`, `h`, `k_B`, `k_e` with SI units; `Quantity.__pow__` and unit simplification (J, N).
*   [x] **Unary Minus**: Codegen emits `-expr`; `Quantity` and `Quaternion` implement `__neg__` for correct behaviour in expressions like `-1.0[C]` and `-1.0 + 0i`.

## 🔭 Beyond v1.0: Future Vision

Fourier aims to become the "Standard Language for Nature's Laws." To achieve this, we are researching the native implementation of the following concepts:

1. **Differentiable ODE Solvers**: Native support for solving Differential Equations integrated directly with the `grad()` engine for physics-informed machine learning.
2. **Physical Units**: Implemented at runtime: `10[m] / 2[s]` → `5[m/s]`, add/sub require same unit; future: compile-time unit checking.
3. **Probabilistic Programming**: Native support for random variables and Bayesian inference, allowing variables to represent probability distributions rather than discrete values.
4. **Symbolic Simplification**: A compile-time algebraic engine that simplifies complex mathematical expressions before code generation to maximize efficiency.

## 📄 License

This project is licensed under the MIT License.
