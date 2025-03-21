/docs/source/conf.py
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import inspect
import os
import subprocess
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

# Add any additional paths needed
package_dir = os.path.abspath('../..')
sys.path.insert(0, package_dir)

# -- Project information -----------------------------------------------------
project = 'osiris_utils'
copyright = '2025, João Biu, João Cândido, Diogo Carvalho'
author = 'João Biu, João Cândido, Diogo Carvalho'
version = 'v1.1.2'
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx_github_style',
]

# Options for sphinx_github_style
top_level = 'OSIRIS Utils'
linkcode_blob = 'head'
linkcode_url = r'https://github.com/joaopedrobiu6/osiris_utils/'
linkcode_link_text = 'Source'

# Autodoc configuration
autodoc_default_options = {
    'member-order': 'bysource',
    'exclude-members': '__init__',
    'undoc-members': False,
}

# Set to False to allow builds even with warnings
autodoc_warningiserror = False

# Mock imports if needed for modules that might not be available during doc building
autodoc_mock_imports = ['numpy', 'matplotlib', 'tqdm']

# Napoleon settings for docstring parsing
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.rst']

# -- Options for intersphinx -------------------------------------------------
# Link to other projects' documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme.
html_theme_options = {
    'logo_only': True,
    'prev_next_buttons_location': 'both',
    'style_external_links': False,
    'style_nav_header_background': '#3c4142',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 2,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here.
html_static_path = ['_static']
html_css_files = ['custom.css']

# The name of an image file (relative to this directory) to place at the top of the sidebar.
# html_logo = '_static/images/logo_small_clear.png'

# The name of an image file (within the static path) to use as favicon.
# html_favicon = '_static/images/desc_icon.ico'

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to typographically correct entities.
html_use_smartypants = True

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, 'Created using Sphinx' is shown in the HTML footer.
html_show_sphinx = True

# If true, '(C) Copyright ...' is shown in the HTML footer.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = 'osiris_utils'

# -- Debug path setup --------------------------------------------------------
# Function to print sys.path for debugging import errors
def setup(app):
    app.connect('builder-inited', lambda app: print(f"sys.path: {sys.path}"))