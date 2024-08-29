import h5py

from PyMca5.PyMcaGui import PyMcaQt as qt
from PyMca5.PyMcaCore.NexusTools import getStartingPositionersGroup

from . import HDF5Info


class NexusMotorInfoWidget(qt.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)

        self.label = qt.QLabel(self)
        self.label.setText("Number of motors: 0")

        column_names = ["Name", "Value", "Units"]
        self._column_names = column_names

        self.table = qt.QTableWidget(self)
        self.table.setColumnCount(len(column_names))
        for i in range(len(column_names)):
            item = self.table.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(column_names[i], qt.QTableWidgetItem.Type)
            item.setText(column_names[i])
            self.table.setHorizontalHeaderItem(i, item)
        self.table.setSortingEnabled(True)

        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.table)

    def setInfoDict(self, ddict):
        if "motors" in ddict:
            self._setInfoDict(ddict["motors"])
        else:
            self._setInfoDict(ddict)

    def _setInfoDict(self, ddict):
        nrows = len(ddict.get(self._column_names[0], []))
        self.label.setText("Number of motors: %d" % nrows)
        self.table.setRowCount(nrows)

        if not nrows:
            self.hide()
            return

        for row in range(nrows):
            for col, label in enumerate(self._column_names):
                text = str(ddict[label][row])
                item = self.table.item(row, col)
                if item is None:
                    item = qt.QTableWidgetItem(text, qt.QTableWidgetItem.Type)
                    item.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
                    self.table.setItem(row, col, item)
                else:
                    item.setText(text)

        for col in range(len(self._column_names)):
            self.table.resizeColumnToContents(col)


class NexusInfoWidget(HDF5Info.HDF5InfoWidget):

    def _build(self):
        super()._build()
        self.motorInfoWidget = NexusMotorInfoWidget(self)
        self.addTab(self.motorInfoWidget, "Motors")

    def setInfoDict(self, ddict):
        super().setInfoDict(ddict)
        self.motorInfoWidget.setInfoDict(ddict)


def getInfo(hdf5File, node):
    """
    hdf5File is and HDF5 file-like insance
    node is the posix path to the node
    """
    info = HDF5Info.getInfo(hdf5File, node)
    info["motors"] = get_motor_positions(hdf5File, node)
    return info


def get_motor_positions(hdf5File, node):
    node = hdf5File[node]

    nxentry_name = node.name.split("/")[1]
    if not nxentry_name:
        return dict()

    nxentry = hdf5File[nxentry_name]
    if not isinstance(nxentry, h5py.Group):
        return dict()

    nxpositioners = getStartingPositionersGroup(hdf5File, nxentry_name)
    if nxpositioners is None or not isinstance(nxpositioners, h5py.Group):
        return dict()

    positions = [
        (name, dset[()], dset.attrs.get("units", ""))
        for name, dset in nxpositioners.items()
        if isinstance(dset, h5py.Dataset) and dset.ndim == 0
    ]
    column_names = "Name", "Value", "Units"

    return dict(zip(column_names, zip(*positions)))
