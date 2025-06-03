from setuptools import setup, find_packages


setup(
    name='flowbench',
    version='0.1.0',
    # author='Your Name',
    # author_email='your.email@example.com',
    description='A package for benchmarking the robustness of optical flow estimation methods.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shashankskagnihotri/benchmarking_robustness/',
    packages=find_packages(),
    install_requires=[
        "cospgd==0.1.3",
        "cupy-cuda11x==13.4.1",
        "torch==2.7.0+cu118",
        "torch-scatter==2.1.2",
        "torchaudio==2.7.0+cu118",
        "torchvision==0.22.0+cu118",
        "matplotlib==3.10.3"
    ],
    python_requires='>=3.10, <3.11',
    license='MIT',
)
