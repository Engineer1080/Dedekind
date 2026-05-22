"""Run the Dedekind kernel: python -m dedekind_jupyter_kernel -f <connection_file>"""
from ipykernel.kernelapp import IPKernelApp
from .kernel import DedekindKernel


if __name__ == "__main__":
    IPKernelApp.launch_instance(kernel_class=DedekindKernel)
