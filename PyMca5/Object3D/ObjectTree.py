#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
#
# This file is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This file is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license
# is a problem for you.
#
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "LGPL2+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
"""
Provides the class 'ObjectTree'.
"""

class ObjectTree(object):
    """
    Implements a simple tree of objects with a _name attribute.
    Number of children is illimited.
    Recursive structure
    """
    def __init__(self, item, name=None):
        """
        Init an empty tree.
        PARAMETERS :
            item : the object of the tree
            name : the name of the object.
                   If ignored it will be set to object.name()
                   or to Unnamed.
        """
        self.root = [item]
        if name is None:
            if hasattr(item, "name"):
                name = item.name()
            else:
                name = "Unnamed"
        self.__name = name

    def name(self):
        """
        Return its name.
        """
        #if type(self.root[0]) == type(''):
        #    return self.root[0]
        #return self.root[0]._name
        return self.__name

    def addChild(self, item, name=None):
        """
        Add a child to the current tree or sub-tree.
        PARAMETERS :
            item : the item of the child tree
        RETURNS :
            the child
        """
        child = ObjectTree(item, name=name)
        self.root.append(child)
        return child

    def addChildTree(self, childTree):
        """
        Add a child tree to the tree.
        PARAMETERS :
             childTree : the child tree to add
        """
        self.root.append(childTree)

    def childList(self):
        """
        Compute the list of children.
        """
        return self.root[1:]

    def childNumber(self):
        """
        Compute the number of children.
        """
        return len(self.root) - 1

    def delDirectChild(self, name):
        """
        Remove a direct child.
        PARAMETERS :
            name : the name of the child to remove
        RETURNS :
            1 if deleted, else 0
        """
        nbChild = self.childNumber()+1
        for i in range(1,nbChild):
            if self.root[i].name() == str(name):
                del self.root[i]
                return 1
        return 0

    def delChild(self, name):
        """
        Remove a child in the tree.
        PARAMETERS :
            name : the name of the child to remove
        RETURNS :
            1 if deleted, else 0
        """
        nbChild = self.childNumber()+1
        for i in range(1,nbChild):
            if self.root[i].name() == str(name):
                del self.root[i]
                return 1

        for child in self.childList():
            child.delChild(name)

    def erase(self):
        """
        Full erase of the tree.
        Only the name stays.
        """
        del self.root[1:]

    def getList(self):
        """
        Compute a plane list of the elements.
        RETURNS :
            this list of names.
        """
        result = [self.name()]
        childList = self.childList()

        for child in childList:
            result += child.getList()

        return result

    def find(self, childName):
        """
        Find a child - or sub-child, given its name.
        PARAMETERS :
            childName : the name of the child to find
        RETURNS :
            the child, or 'None' if not found
        """
        if self.name() == str(childName):
            return self

        for child in self.childList():
            f = child.find(childName)
            if f!=None:
                return f
        return None


    def parent(self, childName):
        """
        Return the parent of the child in the tree.
        PARAMETERS :
            childName : the name of the child to find the parent
        RETURNS :
            the parent, or 'None' if not found
        """
        nbChild = self.childNumber()+1
        for i in range(1,nbChild):
            if self.root[i].name() == str(childName):
                return self

        for child in self.childList():
            parent = child.parent(childName)
            if parent!=None:
                return parent

        return None


    def getLine(self, childName):
        """
        Extract the line of parents from root to child.
        PARAMETERS :
            childName : the name of the child to extract the line
        RETURNS :
            a list of name, from root to child, or [] if not found
        """
        name = self.name()
        if name == str(childName):
            return [name]

        for child in self.childList():
            line = child.getLine(childName)
            if line!=[]:
                return [name] + line
        return []


    def __str__(self):
        """
        Return a printable string representing the tree.
        This is only a convinience mask for 'self.str2()'.
        """
        return "*" + self.str2(1)


    def str2(self, indent):
        """
        Return a printable string representing the tree.
        PARAMETERS :
            indent : the number of space to indent the tree string
        """
        result = "-> " + self.name() + "\n"
        newIndent = indent+3

        for subTree in self.childList():
            result += newIndent * " " + subTree.str2(newIndent)

        return result


    def html(self, fileName, openMode="w"):
        """
        Similar to str, but in a HTML file .
        """
        ffile = open(fileName, openMode)
        self.htmlTable(ffile)
        ffile.close()

    def htmlTable(self, ffile):
        """
        Create table here.
        """
        ffile.write("<table width=100%>\n")
        name = self.name()
        nb = self.childNumber()

        if nb <= 0:
            ffile.write("<tr><td bgcolor=skyblue><center><b>%s</b></center></td></tr>\n" %name)
        else:
            ffile.write("<tr><td colspan=%d bgcolor=skyblue><center><b>%s</b></center></td></tr>\n" %(nb, name))
            for child in self.childList():
                ffile.write("<td valign=top>\n")
                child.htmlTable(ffile)
                ffile.write("</td>\n")

        ffile.write("</table>\n")


if __name__ == "__main__":
    import Object3DQt as qt
    import Object3DBase
    app = qt.QApplication([])
    o0 = Object3DBase.Object3D("DummyObject0")
    o1 = Object3DBase.Object3D("DummyObject1")
    o01 = Object3DBase.Object3D("DummyObject01")
    w = ObjectTree('__Scene__', name='root')
    t0=w.addChild(o0)
    t1=w.addChild(o1)
    if 0:
        t0.addChild(o01)
    else:
        #append DummyObject01 to DummyObject0'
        tree = w.find("DummyObject0")
        tree.addChild(o01)
    print(w)
    print("LIST")
    print(w.getList())
    print("Now I am going to paste DummyObject0 into DummyObject1")
    t0 = w.find("DummyObject0")
    w.delChild("DummyObject0")
    t1=w.find("DummyObject1")
    t1.addChildTree(t0)
    print(w)

    print("LIST")
    print(w.getList())
