#include "MinMax.h"

#define GET_MINMAX(NAME, TYPE) \
void NAME(TYPE * data, unsigned int length, TYPE * min, TYPE * max) \
{ \
    TYPE tmpMin = data[0]; \
    TYPE tmpMax = tmpMin; \
    TYPE * endPtr = &data[length]; \
    TYPE * curPtr; \
\
    for (curPtr = data; curPtr < endPtr; curPtr++) { \
        TYPE value = *curPtr; \
        if (value < tmpMin) { \
            tmpMin = value; \
        } \
        else if (value > tmpMax) { \
            tmpMax = value; \
        } \
    } \
    *min = tmpMin; \
    *max = tmpMax; \
}

GET_MINMAX(getMinMaxDouble, double)
GET_MINMAX(getMinMaxFloat, float)

GET_MINMAX(getMinMaxInt8, int8_t)
GET_MINMAX(getMinMaxUInt8, uint8_t)
GET_MINMAX(getMinMaxInt16, int16_t)
GET_MINMAX(getMinMaxUInt16, uint16_t)
GET_MINMAX(getMinMaxInt32, int32_t)
GET_MINMAX(getMinMaxUInt32, uint32_t)
GET_MINMAX(getMinMaxInt64, int64_t)
GET_MINMAX(getMinMaxUInt64, uint64_t)

