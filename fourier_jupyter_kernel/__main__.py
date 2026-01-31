"""Run the Fourier kernel: python -m fourier_jupyter_kernel -f <connection_file>"""
from ipykernel.kernelapp import IPKernelApp
from .kernel import FourierKernel


if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=FourierKernel)
