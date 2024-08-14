/* Source http://paulbourke.net/geometry/polygonmesh/
 with contribution by Alexander Motrichuk: InsidePolygonWithBounds.cpp
 to deal with points exactly on a vertex 
 Any source code found in the previous site may be freely used provided 
 credits are given to the author. Credits follow:
 */
#/*##########################################################################
#
# Copyright Paul Bourke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/

/* SOLUTION #1 (2D) */
#define MIN(x,y) (x < y ? x : y)
#define MAX(x,y) (x > y ? x : y)
#define INSIDE 1
#define OUTSIDE 0

typedef struct {
   double x,y;
} Point;

typedef struct {
   float x,y;
} PointF;

typedef struct {
   int x,y;
} PointInt;


unsigned char _InsidePolygonF(Point *polygon, int N,\
                              PointF p, unsigned char bound_value);

void PointsInsidePolygonF(double *polygon_xy, int N_xy, \
                         float *points_xy, int N_points_xy,
                         int border_value, unsigned char *output);

unsigned char _InsidePolygonInt(Point *polygon, int N,\
                              PointInt p, unsigned char bound_value);

void PointsInsidePolygonInt(double *polygon_xy, int N_xy, \
                         int *points_xy, int N_points_xy,
                         int border_value, unsigned char *output);


unsigned char _InsidePolygon(Point *polygon, int N,\
                              Point p, unsigned char bound_value);

void PointsInsidePolygon(double *polygon_xy, int N_xy, \
                         double *points_xy, int N_points_xy,
                         int border_value, unsigned char *output);

