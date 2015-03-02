/** @file MinMax.h
 *  Get min and max from data.
 */
#ifndef __MinMax_H__
#define __MinMax_H__

/** Get the min and max values of data.
 *
 * Optionally get the min positive value of data.
 *
 * @param data Pointer to the data to get min and max from.
 * @param type Bit field describing the data type.
 * @param length Number of elements in data.
 * @param minOut Pointer where to store the min value.
 * @param minPositiveOut Pointer where to store the strictly positive min.
 *        If not required, set to NULL.
 *        If all values of data are < 0, set to 0.
 * @param maxOut Pointer where to store the max value.
 */
void
getMinMax(void * data,
          unsigned int type,
          unsigned int length,
          double * minOut,
          double * minPositiveOut,
          double * maxOut);

#endif /*__MinMax_H__*/
