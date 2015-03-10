/****************************************************************************
*   @(#)spec_shm.h	6.7  11/20/13 CSS
*
*   "spec" Release 6
*
*   Copyright (c) 1995-2010 Certified Scientific Software
*
*   The software contained in this file "spec_shm.h" describes the
*   shared-data structures used and defined by the CSS "spec" package.
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

#define SHM_MAGIC       0xCEBEC000

/*
*  Difference between SHM_VERSION 3 and 4 is the increase in
*  header size from 1024 to 4096 to put the data portion
*  on a memory page boundary.
*
*  Difference between SHM_VERSION 4 and 5 is the addition of
*  the SHM_IS_FRAMES tag and the frame_size and latest_frames
*  elements of the shm_head structure.
*
*  Difference between SHM_VERSION 5 and 6 is the addition of
*  metadata tail to data and info array to shm_head structure.
*/
#define SHM_VERSION     6

/* structure flags */
#define SHM_IS_STATUS   0x0001
#define SHM_IS_ARRAY    0x0002
#define SHM_IS_MASK     0x000F  /* User can't change these bits */
#define SHM_IS_MCA      0x0010
#define SHM_IS_IMAGE    0x0020
#define SHM_IS_SCAN     0x0040
#define SHM_IS_INFO     0x0080
#define SHM_IS_FRAMES   0x0100

/* array data types */
#define SHM_DOUBLE      0
#define SHM_FLOAT       1
#define SHM_LONG        2
#define SHM_ULONG       3
#define SHM_SHORT       4
#define SHM_USHORT      5
#define SHM_CHAR        6
#define SHM_UCHAR       7
#define SHM_STRING      8
#define SHM_LONG64      9
#define SHM_ULONG64     10

#define NAME_LENGTH     32
#define INFO_LENGTH     512

/*
*  The meta_length field is not defined by the standard.
*  However, spec will use the following size for the
*  shared arrays it creates.  Applications that use
*  the meta data should always check the meta_length
*  field and not rely on SHM_META_SIZE.
*/
#define SHM_META_SIZE   8192    /* size of metadata region added to tail of array */

#define SHM_OHEAD_SIZE  1024    /* Old header size */
#define SHM_HEAD_SIZE   4096    /* Header size puts data on page boundary */

#ifndef SPEC_TYPE_DEFS
typedef int     s32_t;
typedef unsigned int    u32_t;
typedef long long       s64_t;
typedef unsigned long long      u64_t;
#endif

struct  shm_head {
	u32_t   magic;                  /* magic number (SHM_MAGIC) */
	u32_t   type;                   /* one of the array data types */
	u32_t   version;                /* version number of this struct */
	u32_t   rows;                   /* number of rows of array data */
	u32_t   cols;                   /* number of cols of array data */
	u32_t   utime;                  /* last-updated counter */
	char    name[NAME_LENGTH];      /* name of spec variable */
	char    spec_version[NAME_LENGTH];      /* name of spec process */
	s32_t   shmid;                  /* shared mem ID */
	u32_t   flags;                  /* more type info */
	u32_t   pid;                    /* process id of spec process */
	/*
	*  A frame can be a single MCA acquisition or a single image.
	*  A 2D array can be considered a succession of MCA frames or
	*  a succession of images.  Since data is stored row-wise,
	*  frames are defined by a number of rows.
	*/
	u32_t   frame_size;             /* number of rows per frame */
	u32_t   latest_frame;           /* most recently updated frame */
	/*
	*  The metadata region added in SHM_VERSION 6 is located after
	*  the array data.  The meta_start element is the offset
	*  from the start of the shared segment.
	*/
	u32_t   meta_start;             /* byte offset to metadata tail */
	u32_t   meta_length;            /* byte length of metadata tail */
	char    pad[256];               /* space to expand */
	char    info[INFO_LENGTH];      /* arbitrary info */
};

#define SHM_MAX_IDS     256

struct  shm_status {
	u32_t    spec_state;
	u32_t    utime;                 /* updated when ids[] changes */
	s32_t    ids[SHM_MAX_IDS];      /* shm ids for shared arrays */
	/* more later */
};

struct shm_oheader {
	union {
		struct  shm_head head;
		char    pad[SHM_OHEAD_SIZE];
	} head;
	void    *data;
};

struct shm_header {
	union {
		struct  shm_head head;
		char    pad[SHM_HEAD_SIZE];
	} head;
	void    *data;
};
