#/*##########################################################################
#
# This file is part of Certified Scientific Software (CSS) SPEC package.
# It is distributed with the agreement of CSS.
#
#############################################################################*/
/****************************************************************************
*   @(#)spec_shm.h	4.7  06/13/99 CSS
*
*   "Spec" Release 4
*
*   Copyright (c) 1995,1996,1997,1999
*   by Certified Scientific Software.
*   All rights reserved.
*   Copyrighted as an unpublished work.
*
****************************************************************************/

#define SHM_MAGIC       0xCEBEC000
/*
*  Difference between SHM_VERSION 3 and 4 is the increase in
*  header size from 1024 to 4096 to put the data portion
*  on a memory page boundary.
*/
#define SHM_VERSION     4

/* structure flags */
#define SHM_IS_STATUS   0x0001
#define SHM_IS_ARRAY    0x0002
#define SHM_IS_MASK     0x000F  /* User can't change these bits */
#define SHM_IS_MCA      0x0010
#define SHM_IS_IMAGE    0x0020
#define SHM_IS_SCAN     0x0040
#define SHM_IS_INFO     0x0080

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

#define NAME_LENGTH     32
#define SHM_OHEAD_SIZE  1024    /* Old header size */
#define SHM_HEAD_SIZE   4096    /* Header size puts data on page boundary */

struct  shm_head {
	long    magic;                  /* magic number (SHM_MAGIC) */
	long    type;                   /* one of the array data types */
	long    version;                /* version number of this struct */
	long    rows;                   /* number of rows of array data */
	long    cols;                   /* number of cols of array data */
	long    utime;                  /* last-updated counter */
	char    name[NAME_LENGTH];      /* name of spec variable */
	char    spec_version[NAME_LENGTH];      /* name of spec process */
	long    shmid;                  /* shared mem ID */
	long    flags;                  /* more type info */
	long    pid;                    /* process id of spec process */
};

#define SHM_MAX_IDS     128

struct  shm_status {
	unsigned long   spec_state;
	long     utime;                 /* updated when ids[] changes */
	int      ids[SHM_MAX_IDS];      /* shm ids for shared arrays */
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
