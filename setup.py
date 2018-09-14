from setuptools import setup, find_packages
from bsym import __version__ as VERSION

readme = 'README.md'
long_description = open( readme ).read()

config = {
    'description': 'A Basic Symmetry Module',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'author': 'Benjamin J. Morgan',
    'author_email': 'b.j.morgan@bath.ac.uk',
    'url': 'https://github.com/bjmorgan/bsym',
    'download_url': "https://github.com/bjmorgan/bsym/archive/%s.tar.gz" % (__version__),
    'author_email': 'b.j.morgan@bath.ac.uk',
    'version': __version__,
    'install_requires': [ 'numpy', 
			  'pymatgen>=v2017.10.16', 
                          'coverage',
                          'codeclimate-test-reporter' ],
    'python_requires': '>=3.5',
    'license': 'MIT',
    'packages': [ 'bsym', 'bsym.interface' ],
    'scripts': [],
    'name': 'bsym'
}

setup(**config)
