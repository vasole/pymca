/* Make sure we work with 32 bit integers */
#if defined (_MSC_VER)
    /* Microsoft Visual Studio */
    #if _MSC_VER >= 1600
        /* Visual Studio 2010 and higher */
        #include <stdint.h>
    #else
        #ifndef int32_t
            #define int32_t int
            #define uint32_t unsigned int
        #endif
    #endif
#else
    #include <stdint.h>
#endif

void getMinMaxDouble(double *data, long nValues, \
                     double *minValue, double *maxValue, double *minPositive, double highestValue);
void getMinMaxFloat(float *data, long nValues, \
                    double *minValue, double *maxValue, double *minPositive, float highestValue);
void getMinMaxChar(char *data, long nValues, \
                   double *minValue, double *maxValue, double *minPositive, char highestValue);
void getMinMaxUChar(unsigned char *data, long nValues, \
                    double *minValue, double *maxValue, double *minPositive, unsigned char highestValue);
void getMinMaxShort(short *data, long nValues, \
                    double *minValue, double *maxValue, double *minPositive, short highestValue);
void getMinMaxUShort(unsigned short *data, long nValues, \
                     double *minValue, double *maxValue, double *minPositive, unsigned short highestValue);
void getMinMaxInt32(int32_t *data, long nValues, \
                  double *minValue, double *maxValue, double *minPositive, int32_t highestValue);
void getMinMaxUInt32(uint32_t *data, long nValues, \
                   double *minValue, double *maxValue, double *minPositive, uint32_t highestValue);
void getMinMaxInt(int *data, long nValues, \
                  double *minValue, double *maxValue, double *minPositive, int32_t highestValue);
void getMinMaxUInt(unsigned int *data, long nValues, \
                   double *minValue, double *maxValue, double *minPositive, uint32_t highestValue);
void getMinMaxLong(long *data, long nValues,\
                    double *minValue, double *maxValue, double *minPositive, long highestValue);
void getMinMaxULong(unsigned long *data, long nValues, \
                    double *minValue, double *maxValue, double *minPositive, unsigned long highestValue);

void fillPixmapFromDouble(double *data, long nValues, unsigned char *colormap, long nColors, \
                          unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromFloat(float *data, long nValues, unsigned char *colormap, long nColors, \
                         unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromChar(char *data, long nValues, unsigned char *colormap, long nColors, \
                        unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromUChar(unsigned char *data, long nValues, unsigned char *colormap, long nColors, \
                         unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromShort(short *data, long nValues, unsigned char *colormap, long nColors, \
                         unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromUShort(unsigned short *data, long nValues, unsigned char *colormap, long nColors, \
                          unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromInt(int *data, long nValues, unsigned char *colormap, long nColors, \
                       unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromUInt(unsigned int *data, long nValues, unsigned char *colormap, long nColors, \
                        unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromInt32(int32_t *data, long nValues, unsigned char *colormap, long nColors, \
                       unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromUInt32(uint32_t *data, long nValues, unsigned char *colormap, long nColors, \
                        unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromLong(long *data, long nValues, unsigned char *colormap, long nColors, \
                        unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
void fillPixmapFromULong(unsigned long *data, long nValues, unsigned char *colormap, long nColors, \
                         unsigned char *pixmap, short method, short autoFlag, double *minValue, double *maxValue);
