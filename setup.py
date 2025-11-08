#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
#
# This file is part of Spacetime-sym.
#
# Spacetime-sym is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spacetime-sym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from setuptools import setup, find_packages
from spacetime.version import __version__ as VERSION

readme = 'README.md'
long_description = open( readme ).read()

config = {
    'description': 'Spacetime Symmetry Module',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'author': 'Dr. Bogdan Tanygin',
    'author_email': 'bogdan@tanygin-holding.com',
    'url': 'https://github.com/bogdan-tanygin/spacetime-sym',
    'download_url': "https://github.com/bogdan-tanygin/spacetime-sym/archive/%s.tar.gz" % (VERSION),
    'author_email': 'bogdan@tanygin-holding.com',
    'version': VERSION,
    'setup_requires': 'numpy',
    'install_requires': open( 'requirements.txt' ).read(),
    'python_requires': '>=3.9.18',
    'license': 'GPL-3.0',
    'packages': ['spacetime', 'spacetime.interface'],
    'scripts': [],
    'name': 'spacetime-sym'
}

setup(**config)
