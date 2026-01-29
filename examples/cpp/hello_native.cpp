#include <iostream>
#include <vector>

// Fourier Native Runtime Stubs
int main() {
    std::cout << "--- Fourier Native AOT Runtime ---" << std::endl;
    std::cout << "Executing MLIR-lowered kernels..." << std::endl;
    // The AOT compiler links the Fourier Dialect to LLVM here.
    std::cout << "Compiled from: examples/fourier/hello.fourier" << std::endl;
    return 0;
}
