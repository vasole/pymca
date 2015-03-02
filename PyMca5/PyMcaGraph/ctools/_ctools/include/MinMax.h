/** @file MinMax.h
 *  Get min and max from data.
 */
#ifndef __MinMax_H__
#define __MinMax_H__

/** Get the min and max from data.
 *
 * @param data Pointer to the data to get min and max from.
 * @param type Bit field describing the data type.
 * @param length Number of elements in data.
 * @param minOut Output the min of data.
 * @param maxOut Output the max of data.
 */
void
getMinMax(void * data,
          unsigned int type,
          unsigned int length,
          double * minOut,
          double * maxOut);

#endif /*__MinMax_H__*/
