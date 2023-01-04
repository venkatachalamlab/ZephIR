from setuptools import setup, find_packages
# from .zephir.__version__ import __version__
__version__ = '1.0.4'

requirements = [
    'dataclasses>=0.6'
    'docopt>=0.6.2',
    'Flask>=2.1.2',
    'gevent>=21.12.0',
    'h5py>=3.6.0',
    'matplotlib>=3.5.2',
    'numpy>=1.22.4',
    'opencv-python>=4.5.5.64'
    'pandas>=1.4.2',
    'scikit-image>=0.19.2',
    'scikit-learn>=1.0.2',
    'scipy>=1.7.3',
    'setuptools>=61.2.0',
    'torch>=1.10.0',
    'tqdm>=4.64.0',
]

setup(
    name='zephir',
    version=__version__,
    license='MIT',
    description='Multiple object tracking algorithm via image registration',
    author='James Yu, Vivek Venkatachalam',
    author_email='v.venkatachalam@northeastern.edu',
    maintainer='James Yu',
    maintainer_email='jd.ryu@icloud.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/venkatachalamlab/ZephIR',
    entry_points={'console_scripts': [
        'zephir=zephir.main:main',
        'annotator=zephir.annotator.main:main',
        'auto_annotate=zephir.methods.auto_annotate:main',
        'extract_traces=zephir.methods.extract_traces:main',
        'overwrite_checkpoint=zephir.methods.overwrite_checkpoint:main',
        'recommend_frames=zephir.methods.recommend_frames:main',
        'train_zephod=zephir.zephod.train:main',
        'zephod=zephir.zephod.main:main',
    ]},
    include_package_data=True,
    zip_safe=False,
    # keywords=[
    #     'image registration',
    #     'neuron tracking',
    #     'object tracking',
    #     'multiple object tracking',
    # ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    install_requires=requirements,
    packages=find_packages()
)
