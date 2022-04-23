from setuptools import setup, find_packages

setup(
    name='wealth-transfer',
    version='0.0.1',
    # url='https://github.com/mypackage.git',
    author='David and Gufran',
    # author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    # install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)