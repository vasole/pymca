cdef extern from "MinMax.h":
        void getMinMaxDouble(double * data, unsigned int length,
                         double * min, double * max)
        void getMinMaxFloat(float * data, unsigned int length,
                        float * min, float * max)
        void getMinMaxInt8(char * data, unsigned int length,
                       char * min, char * max)
        void getMinMaxUInt8(unsigned char * data, unsigned int length,
                        unsigned char * min, unsigned char * max)
