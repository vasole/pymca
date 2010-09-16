#!/bin/sh
#
# By Jerome Kieffer (Jerome.Kieffer@esrf.fr) 
#
# For building Debian / Ubuntu packages you will need the package python-stdeb
rm -rf deb_dist
python setup.py --command-packages=stdeb.command bdist_deb
sudo dpkg -i deb_dist/python-pymca*.deb
