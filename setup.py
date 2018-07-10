from setuptools import find_packages
from setuptools import setup


version = '0.0.0'


setup(
    name='light_head_rcnn',
    version=version,
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    description='',
    long_description=open('README.md').read(),
    author='Shingo Kitagawa',
    author_email='shingogo.5511@gmail.com',
    url='https://github.com/knorth55/chainer-light-head-rcnn',
    license='MIT',
    keywords='machine-learning',
)
