import os
# this will be filled by the setup
PYMCA_DATA_DIR = 'DATA_DIR_FROM_SETUP'
# This is to be filled by the setup
PYMCA_DOC_DIR = 'DOC_DIR_FROM_SETUP'
# what follows is only used in frozen versions
if not os.path.exists(PYMCA_DATA_DIR):
    tmp_dir = os.path.dirname(__file__)
    basename = os.path.basename(PYMCA_DATA_DIR)
    PYMCA_DATA_DIR = os.path.join(tmp_dir,basename)
    while len(PYMCA_DATA_DIR) > 14:
        if os.path.exists(PYMCA_DATA_DIR):
            break
        tmp_dir = os.path.dirname(tmp_dir)
        PYMCA_DATA_DIR = os.path.join(tmp_dir, basename)

# this is used in build directory
if not os.path.exists(PYMCA_DATA_DIR):
    tmp_dir = os.path.dirname(__file__)
    basename = os.path.basename(PYMCA_DATA_DIR)
    PYMCA_DATA_DIR = os.path.join(tmp_dir, "PyMca", basename)
    while len(PYMCA_DATA_DIR) > 14:
        if os.path.exists(PYMCA_DATA_DIR):
            break
        tmp_dir = os.path.dirname(tmp_dir)
        PYMCA_DATA_DIR = os.path.join(tmp_dir, "PyMca", basename)

if not os.path.exists(PYMCA_DATA_DIR):
    raise IOError('%s directory not found' % basename)
# do the same for the directory containing HTML files
if not os.path.exists(PYMCA_DOC_DIR):
    tmp_dir = os.path.dirname(__file__)
    basename = os.path.basename(PYMCA_DOC_DIR)
    PYMCA_DOC_DIR = os.path.join(tmp_dir,basename)
    while len(PYMCA_DOC_DIR) > 14:
        if os.path.exists(PYMCA_DOC_DIR):
            break
        tmp_dir = os.path.dirname(tmp_dir)
        PYMCA_DOC_DIR = os.path.join(tmp_dir, basename)
if not os.path.exists(PYMCA_DOC_DIR):
    raise IOError('%s directory not found' % basename)

if not os.path.exists(PYMCA_DOC_DIR):
    tmp_dir = os.path.dirname(__file__)
    basename = os.path.basename(PYMCA_DOC_DIR)
    PYMCA_DOC_DIR = os.path.join(tmp_dir, "PyMca", basename)
    while len(PYMCA_DOC_DIR) > 14:
        if os.path.exists(PYMCA_DOC_DIR):
            break
        tmp_dir = os.path.dirname(tmp_dir)
        PYMCA_DOC_DIR = os.path.join(tmp_dir, "PyMca", basename)
if not os.path.exists(PYMCA_DOC_DIR):
    raise IOError('%s directory not found' % basename)
