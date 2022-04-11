#!/bin/sh
#

project=PyMca5
source_project=pymca
version=$(python3 -c"import version; print(version.version)")
strictversion=$(python3 -c"import version; print(version.strictversion)")
debianversion=$(python3 -c"import version; print(version.debianversion)")

deb_name=$(echo "$source_project" | tr '[:upper:]' '[:lower:]')

#clean up build and dist directory
/bin/rm -rf build
/bin/rm -rf dist

# create upstream source package
python3 setup.py sdist

# convert PyMca5_x.y.z.xxxx into pymca_x.y.z.xxxx
cd dist
tar -xvzf ${project}-${strictversion}.tar.gz
mv ${project}-${strictversion} ${source_project}-${strictversion}
tar -cvzf ${source_project}-${strictversion}.tar.gz ${source_project}-${strictversion}
/bin/rm -rf ${project}-${strictversion}.tar.gz

# create .orig file and debian directory
cd ${source_project}-${strictversion}
dh_make --python --yes -f ../${source_project}-${strictversion}.tar.gz

# copy the control and rules
cp ../../package/debian11/control ./debian/
cp ../../package/debian11/rules ./debian/

# actually build the package
pwd
dpkg-buildpackage -uc -us


# remove the intermediate directory
cd ..
/bin/rm -rf ${source_project}-${strictversion}

# everything is under ./dist
cd ..
ls ./dist/*.deb

