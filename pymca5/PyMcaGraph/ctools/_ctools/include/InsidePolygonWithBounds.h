/* Source http://paulbourke.net/geometry/polygonmesh/
 with contribution by Alexander Motrichuk: InsidePolygonWithBounds.cpp
 to deal with points exactly on a vertex */

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

