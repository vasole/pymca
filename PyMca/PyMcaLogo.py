#/*##########################################################################
# Copyright (C) 2004-2009 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem for you.
#############################################################################*/
PyMcaLogo = [
"55 68 8 1",
"  c blue",
". c #07070707fcfc0000",
"X c #3b3b3b3bfdfd0000",
"o c #8c8c8c8cfdfd0000",
"O c #babababafbfb0000",
"+ c #e0e0e0e0fdfd0000",
"@ c #f7f7f7f7fdfd0000",
"# c white",
"#######################################################",
"#######################################################",
"#######################################################",
"###########################O+##########################",
"#####################OO####.o###@o@####################",
"#####################oX####+@###+.+####################",
"################@@#########oO#########+################",
"################Xo####XX##+ X##+XO###@.o###############",
"################O+###@..###o+##O o####O+###############",
"#################OXO##++###O+###+###oX#################",
"###########@O@###o o##@Xo@o .@OXo###X.@###+O###########",
"###########O O####+@+#O  OO .@o .#@+@+####XX###########",
"############+#oO###X o@XX##+@#OXo@X O##@oO@@###########",
"#############+ X###X X##@+###@+##@. o##O X#############",
"##############O++o++o@#O. X#+. X@#+O#oo@O+#############",
"########+o@#####X X####X   OX   o###+  O#####Oo########",
"########O.+Oo###o.o@OO#X   +o   O@O+@XX+##+o@oX########",
"###########X +@O@##X  o+X.o#@X.o+.  O##+O#o o##########",
"###########Oo#X o#+   .#########o   X#@. o+o@##########",
"##############X o##.  X#########+   o#@. O#############",
"#######@######@O###OXo+##########OXo@##++######@#######",
"#######.o@Xo#Oo+#OXo@##############OXo@#Oo+@Xo#Xo######",
"#######o++.X#. o+.  X#############+.  X#. o+ X#oO######",
"##########+@#X.OO   X#############O   X#o.O#+@#########",
"################@X  o#############@X  o################",
"#############oX+#@OO###############@O+##oX+############",
"#######O@+Xo#. o##oXO#############@oo+##. o+.o#+@######",
"#######.o+.X#oX+#O   O############X  X##oXO+.X#.o######",
"#######+@#@######o   o############.   +#####@##+@######",
"##############o.O+. .O############X  X#X.O#############",
"###########++#X o#+o+OXX+######OXo@Oo@@. o@O###########",
"###########X.++o+###@.  X#@OO#+   o####Oo@o o##########",
"########OX@oX@##OXO#+   .@X  oO   X##oX@##OX+OX########",
"########+X@#####X X##X  oO   X@X .O#+  O#####OX########",
"##############+@OXO@+#O+#+.  X#@O+@+#XX++@#############",
"#############+.X###X o####OXX+###@. O##+.X#############",
"#############@Xo###X o#oo##@##+o+@. o##@Xo#############",
"###########+X+#####@o@O  +O.X#o X#+O######oX###########",
"###########+o+###O.O##+XX@o .@O.o###XX@###Oo###########",
"#################o o######@oO#######X.@################",
"################+@+##@XX###+@##+.o##@@+################",
"################Xo###@.X##+ X##O.o###@.o###############",
"################O+####@@##@Xo###+#####O@###############",
"#####################oo#########+X+#####@@@@@@@@@@@@@@#",
"#o             X#####oo####Xo###+X+####+              @",
"#OXXXXXXXXXXXXXX###########oO##########+XXXXXXXXXXXXXX@",
"#######################################################",
"#######################################################",
"#+OOOOOOOOOOO@####OoooO#O@#OOOOOOOOO+#####OOOOOOOOOOOO#",
"##@o  .ooXX  O###o.O+OXX.@#@o   ooo. XO####O   XooX. X#",
"###O  .####O.O##O o####o @##@   +##+. .+####.  O###@XX#",
"###O  .###@#oO##X o#####X@##@   +###X  o####.  O####@o#",
"###O  .##+O#+O##X .O####o@##@   +###o  X####.  O##++#O#",
"###O  .##oO#####o   XO######@   +###X  o####.  O##o+###",
"###O  .#+.O#####+.    X+####@   +##O. .+####.  O#O.+###",
"###O   X. O######O.    .O###@   ooX. XO#####.  XX  +###",
"###O  .#O.O#######@o.    O##@   OX  .+######.  O@o.+###",
"###O  .##oO#####+###+o   X##@   ++.  X######.  O##o+###",
"###O  .##+O##O+#o#####O. .##@   +#O   o#####.  O##O+###",
"###O  .###@##X@#X+#####X X##@   +##o  .O####.  O##@@###",
"###O  .#####o.##.X@####o O##@   +##@X  X@###.  O#######",
"###o   oOOoX X##..XO##O.o###O   O###+.  X@#+   o#######",
"#+oooooooooooO##o#+oXXXO###Ooooooo@##OooooOoooooo+#####",
"#######################################################",
"#######################################################",
"#oXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#",
"#o...................................................X#",
"#######################################################"
]
