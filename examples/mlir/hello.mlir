// Fourier MLIR Dialect Prototype v0.2
module {
  func.func @main() -> tensor<*xf32> {
    %1 = arith.constant Hello from Fourier! : f32
    %2 = call @print(%1)
    %3 = arith.constant 10 : f32
    // assignment to x
    fourier.store %3, @x
    %4 = arith.constant 20 : f32
    // assignment to y
    fourier.store %4, @y
    %5 = fourier.load @x
    %6 = fourier.load @y
    %7 = arith.addf %5, %6 : f32
    %8 = call @print(%7)
    %9 = fourier.constant_tensor [1, 2, 3] : tensor<3xf32>
    // assignment to vec
    fourier.store %9, @vec
    %10 = fourier.load @vec
    %11 = call @print(%10)
  }
  %12 = call @main()
}