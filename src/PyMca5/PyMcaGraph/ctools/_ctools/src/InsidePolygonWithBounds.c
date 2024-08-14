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

#include "../include/InsidePolygonWithBounds.h"

#define _INSIDE_POLYGON(NAME, TYPE) \
unsigned char NAME(Point *polygon, int N, TYPE p, unsigned char border_value)            \
{                                                                                        \
  int counter = 0;                                                                       \
  int i;                                                                                 \
  double xinters;                                                                        \
  Point p1,p2;                                                                           \
                                                                                         \
  p1 = polygon[0];                                                                       \
  for (i=1;i<=N;i++) {                                                                   \
    if ((p1.x == p.x) && (p1.y == p.y))                                                  \
        return border_value;                                                             \
    p2 = polygon[i % N];                                                                 \
    if (p.y > MIN(p1.y,p2.y)) {                                                          \
      if (p.y <= MAX(p1.y,p2.y)) {                                                       \
        if (p.x <= MAX(p1.x,p2.x)) {                                                     \
          if (p1.y != p2.y) {                                                            \
            xinters = (p.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x;                           \
            if (p1.x == p2.x || p.x <= xinters)                                          \
              counter++;                                                                 \
          }                                                                              \
        }                                                                                \
      }                                                                                  \
    }                                                                                    \
    p1 = p2;                                                                             \
  }                                                                                      \
                                                                                         \
  if (counter % 2 == 0)                                                                  \
    return(OUTSIDE);                                                                     \
  else                                                                                   \
    return(INSIDE);                                                                      \
}



_INSIDE_POLYGON(_InsidePolygon, Point)
_INSIDE_POLYGON(_InsidePolygonF, PointF)
_INSIDE_POLYGON(_InsidePolygonInt, PointInt)

void PointsInsidePolygon(double *vertices, int N_vertices, \
                         double *points_xy, int N_points_xy,
                         int border_value, unsigned char *output)
{
    int i;
    Point *polygon;
    Point *point;
    unsigned char *p;
    unsigned char border;



    polygon = (Point *) vertices;
    point = (Point *) points_xy;
    border = (unsigned char) border_value;

    p = output;
    for (i=0; i < N_points_xy; i++)
    {
        *p = _InsidePolygon(polygon, N_vertices, *point, border);
        p++;
        point++;
    }
}

void PointsInsidePolygonF(double *vertices, int N_vertices, \
                         float *points_xy, int N_points_xy,
                         int border_value, unsigned char *output)
{
    int i;
    Point *polygon;
    PointF *point;
    unsigned char *p;
    unsigned char border;



    polygon = (Point *) vertices;
    point = (PointF *) points_xy;
    border = (unsigned char) border_value;

    p = output;
    for (i=0; i < N_points_xy; i++)
    {
        *p = _InsidePolygonF(polygon, N_vertices, *point, border);
        p++;
        point++;
    }
}

void PointsInsidePolygonInt(double *vertices, int N_vertices, \
                         int *points_xy, int N_points_xy,
                         int border_value, unsigned char *output)
{
    int i;
    Point *polygon;
    PointInt *point;
    unsigned char *p;
    unsigned char border;



    polygon = (Point *) vertices;
    point = (PointInt *) points_xy;
    border = (unsigned char) border_value;

    p = output;
    for (i=0; i < N_points_xy; i++)
    {
        *p = _InsidePolygonInt(polygon, N_vertices, *point, border);
        p++;
        point++;
    }
}
