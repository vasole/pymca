/****************************************************************************
*
*   Copyright (c) 1998-2010 European Synchrotron Radiation Facility (ESRF)
*
*   The software contained in this file "blissmalloc.h" is part of the set
*   of files designed to interface the shared-data structures used and defined
*   by the CSS "spec" package with other utility software.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
****************************************************************************/
#if MALLOC_DEBUG

struct pmem {
  void *data;
  long int size;
  char *file;
  int line;
  struct pmem *next;
} ;


#define malloc(N) _pmalloc(N,__FILE__,__LINE__)
#define free(N) _pfree(N,__FILE__,__LINE__)

#endif
