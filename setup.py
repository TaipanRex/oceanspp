from setuptools import setup

with open('requirements.txt', 'r') as fr:
  required = fr.read().splitlines()
  
setup(
  name = 'pyvisgraph',
  packages = ['pyvisgraph'],
  version = '0.2.1',
  description = 'Given a set of simple obstacle polygons, build a visibility graph and find the shortest path between two points.',
  author = 'Christian Reksten-Monsen',
  author_email = 'christian@reksten-monsen.com',
  url = 'https://github.com/TaipanRex/pyvisgraph',
  download_url = 'https://github.com/TaipanRex/pyvisgraph/tarball/0.2.1',
  install_requires = required,
  keywords = ['visibility', 'graph', 'shortest'],
  classifiers = [],
)
