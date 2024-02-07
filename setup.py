from setuptools import setup, find_packages
from bsym.version import __version__ as VERSION

readme = 'README.md'
long_description = open( readme ).read()

config = {
    'description': 'A Basic Spacetime Symmetry Module',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'author': 'Dr. Bogdan Tanygin',
    'author_email': 'bogdan@tanygin-holding.com',
    'url': 'https://github.com/bogdan-tanygin/spacetime-sym',
    'download_url': "https://github.com/bogdan-tanygin/spacetime-sym/archive/%s.tar.gz" % (VERSION),
    'author_email': 'bogdan@tanygin-holding.com',
    'version': VERSION,
    'install_requires': open( 'requirements.txt' ).read(),
    'python_requires': '>=3.12.1',
    'license': 'MIT',
    'packages': ['bsym', 'bsym.interface', 'spacetime-sym'],
    'scripts': [],
    'name': 'spacetime-sym'
}

setup(**config)
