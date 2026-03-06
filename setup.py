from setuptools import setup, find_packages

setup(
    name='cydms',
    version='0.1.0',
    description='EEG + MRI source localization library — no FreeSurfer required',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'nibabel',
        'scipy',
        'scikit-image',
        'mne',
        'pandas',
    ],
    python_requires='>=3.8',
)
