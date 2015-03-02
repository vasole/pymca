#include "MinMax.h"
#include "Types.h"

#define GET_MINMAX_DEFINITION(TYPE)\
static void getMinMax_ ## TYPE(TYPE * data,\
                 unsigned int length,\
                 double * min,\
                 double * max)\
{\
    TYPE tmpMin = data[0];\
    TYPE tmpMax = tmpMin;\
    TYPE * endPtr = &data[length];\
    TYPE * curPtr;\
\
    for (curPtr = data; curPtr < endPtr; curPtr++) {\
        TYPE value = *curPtr;\
        if (value < tmpMin) {\
            tmpMin = value;\
        }\
        else if (value > tmpMax) {\
            tmpMax = value;\
        }\
    }\
    *min = (double) tmpMin;\
    *max = (double) tmpMax;\
}

GET_MINMAX_DEFINITION(float)
GET_MINMAX_DEFINITION(double)

GET_MINMAX_DEFINITION(int8_t)
GET_MINMAX_DEFINITION(uint8_t)

GET_MINMAX_DEFINITION(int16_t)
GET_MINMAX_DEFINITION(uint16_t)

GET_MINMAX_DEFINITION(int32_t)
GET_MINMAX_DEFINITION(uint32_t)

GET_MINMAX_DEFINITION(int64_t)
GET_MINMAX_DEFINITION(uint64_t)


#define CALL_GET_MINMAX(TYPE)\
    getMinMax_ ## TYPE((TYPE *) data,\
        length,\
        minOut,\
        maxOut)


void
getMinMax(void * data,
          unsigned int type,
          unsigned int length,
          double * minOut,
          double * maxOut)
{
    switch (type) {
        case (FLOATING | SIZE_32): /*float*/
            CALL_GET_MINMAX(float);
            break;
        case (FLOATING | SIZE_64): /*double*/
            CALL_GET_MINMAX(double);
            break;

        case (SIZE_8): /*int8_t*/
            CALL_GET_MINMAX(int8_t);
            break;
        case (UNSIGNED | SIZE_8): /*uint8_t*/
           CALL_GET_MINMAX(uint8_t);
           break;

        case (SIZE_16): /*int16_t*/
            CALL_GET_MINMAX(int16_t);
            break;
        case (UNSIGNED | SIZE_16): /*uint16_t*/
            CALL_GET_MINMAX(uint16_t);
            break;

        case (SIZE_32): /*int32_t*/
            CALL_GET_MINMAX(int32_t);
            break;
        case (UNSIGNED | SIZE_32): /*uint32_t*/
            CALL_GET_MINMAX(uint32_t);
            break;

        case (SIZE_64): /*int64_t*/
            CALL_GET_MINMAX(int64_t);
            break;
        case (UNSIGNED | SIZE_64): /*uint64_t*/
            CALL_GET_MINMAX(uint64_t);
            break;
        default:
            break;
    }
}
