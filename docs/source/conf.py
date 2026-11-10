"""Sphinx configuration for the spacetime-sym documentation.
The documentation is built with Sphinx and published as static HTML.
"""

from pathlib import Path
import re
import sys


# -- Repository and project metadata ----------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
# Allow autodoc to import the package directly from a source checkout.
sys.path.insert(0, str(REPO_ROOT))

project = "spacetime-sym"
author = "Dr. Bogdan Tanygin"
copyright = "2024-2026, Dr. Bogdan Tanygin"
language = "en"

# Read the package version without importing its dependencies.
version_file = REPO_ROOT / "spacetime" / "version.py"
version_text = version_file.read_text(encoding="utf-8")
version_match = re.search(
    r'^__version__\s*=\s*[\'\"]([^\'\"]+)[\'\"]',
    version_text,
    re.MULTILINE,
)
if version_match is None:
    raise RuntimeError(f"Could not find __version__ in {version_file}")
release = version_match.group(1)
version = release


# -- Documentation sources and extensions -----------------------------------
# For now, keep docs/source/index.rst. Later it can become index.md.
root_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

extensions = [
    "sphinx.ext.autodoc",       # Document Python classes/functions from docstrings.
    "sphinx.ext.autosummary",   # Generate API reference entries when requested.
    "sphinx.ext.napoleon",      # Google- and NumPy-style docstrings.
    "sphinx.ext.viewcode",      # Links to highlighted Python source.
    "sphinx.ext.githubpages",   # Generate .nojekyll for GitHub Pages.
    "myst_parser",              # Hand-written Markdown and math.
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
myst_heading_anchors = 3

exclude_patterns = ["build", "_build", "Thumbs.db", ".DS_Store"]


# -- HTML output: PyData Sphinx Theme ---------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "spacetime-sym documentation"
html_short_title = "spacetime-sym"
html_theme_options = {
    "show_nav_level": 2,
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/bogdan-tanygin/spacetime-sym",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}

# Required by the PyData theme's "Edit this page" links.
html_context = {
    "github_user": "bogdan-tanygin",
    "github_repo": "spacetime-sym",
    "github_version": "main",
    "doc_path": "docs/source",
}

# Once GitHub Pages is enabled at the URL below, uncomment this to produce
# canonical links in the generated HTML and a sitemap base if configured.
# html_baseurl = "https://bogdan-tanygin.github.io/spacetime-sym/"
