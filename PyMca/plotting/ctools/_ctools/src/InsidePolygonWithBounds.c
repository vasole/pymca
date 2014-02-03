/* Source http://paulbourke.net/geometry/polygonmesh/
 with contribution by Alexander Motrichuk: InsidePolygonWithBounds.cpp
 to deal with points exactly on a vertex */

/* SOLUTION #1 (2D) */

#include "InsidePolygonWithBounds.h"

unsigned char _InsidePolygon(Point *polygon, int N,\
                              Point p, unsigned char border_value)
{
  int counter = 0;
  int i;
  double xinters;
  Point p1,p2;

  p1 = polygon[0];
  for (i=1;i<=N;i++) {
    if ((p1.x == p.x) && (p1.y == p.y))
        return border_value;
    p2 = polygon[i % N];
    if (p.y > MIN(p1.y,p2.y)) {
      if (p.y <= MAX(p1.y,p2.y)) {
        if (p.x <= MAX(p1.x,p2.x)) {
          if (p1.y != p2.y) {
            xinters = (p.y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x;
            if (p1.x == p2.x || p.x <= xinters)
              counter++;
          }
        }
      }
    }
    p1 = p2;
  }

  if (counter % 2 == 0)
    return(OUTSIDE);
  else
    return(INSIDE);
}

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
