import setuptools

setuptools.setup(
    name='trailed-source-centering',
    version='1.0.0',
    description='A Centering Algorithm for Trailed Sources',
    author='Lin-Peng Wu',
    author_email='misterwu1998@163.com',
    maintainer_email='tqfz@jnu.edu.cn',
    url='https://github.com/astrometry-jnu/TrailedSourceCentering',
    # download_url='',
    packages=setuptools.find_packages(),
    py_modules=[
        'trailed_source_centering'
    ],
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio'
    ]
)
