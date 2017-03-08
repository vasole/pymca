/****************************************************************************
*
*   Copyright (c) 1998-2016 European Synchrotron Radiation Facility (ESRF)
*   Copyright (c) 1998-2016 Certified Scientific Software (CSS)
*
*   The software contained in this file "sps.c" is designed to interface
*   the shared-data structures used and defined by the CSS "spec" package
*   with other utility software.
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
/****************************************************************************
*   @(#)sps.c	6.9  05/11/16 CSS
*
*   "spec" Release 6
*
*   This file was mostly an ESRF creation, but includes code from and is
*   maintained by Certified Scientific Software.
*
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <signal.h>

#include <spec_shm.h>

#ifndef IPCS
#define IPCS "LC_ALL=C ipcs -m"
#endif

#define SHM_MAX_ENTRIES 8192
#define SHM_MAX_STR_LEN 8192

#define SHM_CLEANUP 1

typedef struct shm_header SHM;

/*
   id_buffer hold all the SPEC shared memory ids in the system
   The routine init_ShmIDs is used to fill this buffer (slow!)
*/

static u32_t id_buffer[SHM_MAX_ENTRIES];
static s32_t id_no = 0;

/*
  The table SpecIDTab holds a table of all SPEC shmids for every spec
  version there is a table of array ids allocated in memory
*/

struct arrayid {
	char    *name;
	u32_t   id;
};

struct specid {
	char    *spec_version;
	u32_t   id;
	u32_t   pid;
	u32_t   ids_utime;
	struct  arrayid *array_names;
	s32_t   arrayno;
};

static struct   specid SpecIDTab[SHM_MAX_ENTRIES];
static int      SpecIDNo = 0;

/*
   The structure sps_array is used as a handle to one array in a spec
   version. It holds the current state of this connection. Either
   not found yet, not attached yet, or attached
*/

typedef struct sps_array {
	SHM     *shm;
	u32_t   utime;
	char    *spec;
	char    *array;
	int     write_flag;
	int     attached;
	int     stay_attached;
	int     pointer_got_count;
	u32_t   id;
	void    *private_data_copy;
	size_t  buffer_len;
	void    *private_meta_copy;
	u32_t   meta_len;
	void    *private_info_copy;
} *SPS_ARRAY;

/*
   Linked list of shared memories I created or I created handles for
   Information is redundant and should be reduced as soon the interface to
   the outside world is stable
*/

struct shm_created {
	s32_t   id;
	char    *array_name;
	char    *spec_version;
	int     isstatus;
	struct  shm_created *status_shm;
	int     no_referenced;
	SHM     *shm;
	SPS_ARRAY handle;
	int     my_creation;
	struct  shm_created *next;
};

static struct shm_created *SHM_CREATED_HEAD = NULL;

static  SHM     *attachArray(char *fullname, char *array, int read_only);
static  SHM     *attachSpec(char *fullname);
static  SHM     *create_master_shm(char *name);
static  SHM     *create_shm(char *specversion, char *array, int rows, int cols, int type, int flags);
static  SPS_ARRAY add_private_shm(SHM *shm, char *fullname, char *array, int write_flag);
static  SPS_ARRAY convert_to_handle(char *spec_version, char *array_name);
static  char    *GetNextAll(int flag);
static  char    *composeVersion(char *version, u32_t pid);
static  int     DeconnectArray(SPS_ARRAY private_shm);
static  int     ReconnectToArray(SPS_ARRAY private_shm, int write_flag);
static  int     SearchSpecVersions(void);
static  int     checkSHM(SHM *shm, char *spec_version, char *name, u32_t type);
static  int     delete_handle(SPS_ARRAY handle);
static  int     extractVersion(char *fullname, char *name, u32_t *pid);
static  int     find_ArrayIDX(int tab_idx, char *array_name);
static  int     find_TabIDX(char *spec_version, u32_t pid);
static  int     find_TabIDX_composed(char *fullname);
static  int     getShmIDs(u32_t **id_ptr, char *spec_version, char *name, u32_t type);
static  int     init_ShmIDs(void);
static  int     iscomposed(char *fullname);
static  int     TypedCopy(char *fullname, char *array, void *buffer, int my_type,
			int items_in_buffer, int direction);
static  int     typedcp(void *t, int tt, void *f, int ft, int np, int rev, int offset);
static  int     typedcp_private(SPS_ARRAY private_shm, void *buffer, int my_type,
			int items_in_buffer, int direction);
static  s32_t   SearchArrayOnly(char *arrayname);
static  size_t  typedsize(int t);
static  struct  shm_created *ll_addnew_array(char *specversion, char *arrayname, int isstatus,
			struct shm_created *status, s32_t id, int my_creation, SHM *shm);
static  struct  shm_created *ll_find_array(char *specversion, char *arrayname, int isstatus);
static  struct  shm_created *ll_find_pointer(SHM *shm);
static  void    *CopyDataRC(char *fullname, char *array, int my_type, int row, int col,
			int *act_copied, int use_row, int direction, void *my_buffer);
static  void    *c_shmat(s32_t id, char *ptr, int flag);
static  void    c_shmdt(void *ptr);
static  void    *id_is_our_creation(u32_t id);
static  void    *shm_is_our_creation(void *shm);
static  void    SearchSpecArrays(char *fullname);
static  void    delete_SpecIDTab(void);
static  void    delete_id_from_list(u32_t id);
static  void    delete_id_from_status(SHM *status, s32_t id);
static  void    delete_shm(s32_t id);
static  void    ll_delete_array(struct shm_created *todel);

char    *SPS_GetEnvStr(char *spec_version, char *array_name, char *identifier);
char    *SPS_GetNextArray(char *fullname, int flag);
char    *SPS_GetNextSpec(int flag);
int     SPS_CopyFromShared(char *name, char *array, void *buffer, int my_type, int items_in_buffer);
int     SPS_CopyToShared(char *name, char *array, void *buffer, int my_type, int items_in_buffer);
int     SPS_CreateArray(char *spec_version, char *arrayname, int rows, int cols, int type, int flags);
int     SPS_FreeDataCopy(char *fullname, char *array);
int     SPS_GetArrayInfo(char *spec_version, char *array_name, int *rows, int *cols, int *type, int *flag);
int     SPS_GetFrameSize(char *spec_version, char *array_name);
int     SPS_IsUpdated(char *fullname, char *array);
int     SPS_LatestFrame(char *fullname, char *array);
int     SPS_PutEnvStr(char *spec_version, char *array_name, char *identifier, char *set_value);
int     SPS_ReturnDataPointer(void *pointer);
int     SPS_Size(int type);
int     SPS_UpdateDone(char *fullname, char *array);
s32_t   SPS_GetSpecState(char *version);
void    *SPS_GetDataCol(char *name, char *array, int my_type, int col, int row, int *act_rows);
void    *SPS_GetDataCopy(char *fullname, char *array, int my_type, int *rows_ptr, int *cols_ptr);
void    *SPS_GetDataPointer(char *fullname, char*array, int write_flag);
void    *SPS_GetDataRow(char *name, char *array, int my_type, int row, int col, int *act_cols);
void    SPS_CleanUpAll(void);

char    *SPS_GetMetaData(char *spec_version, char *array_name, u32_t *length);
char    *SPS_GetInfoString(char *spec_version, char *array_name);
int     SPS_PutMetaData(char *spec_version, char *array_name, char *data, u32_t length);
int     SPS_PutInfoString(char *spec_version, char *array_name, char *info);

#if SPS_DEBUG
static  int     SPS_AttachToArray(char *fullname, char *array, int write_flag);
static  int     SPS_DetachFromArray(char *fullname, char*array);
#endif

/*
   Our private attach and dettach to stay connected with the shared memory
   we created ourselves
*/

static void *c_shmat(s32_t id, char *ptr, int flag) {
	void *shm;

	if ((shm = id_is_our_creation(id)))
		return shm;
	return shmat(id, ptr, flag);
}

static void c_shmdt(void *ptr) {
	if (shm_is_our_creation(ptr) == NULL)
		shmdt(ptr);
}

/*
   Used internally. Finds the index in the SpecIDTab array with given
   spec_version and pid. For the moment the pid is only used to decide
   between two different versions of spec with the same name.
   Input: spec_version the name of the version to look for
          pid: the pid of the specversion or 0 if every pid.
   Returns: -1 if not found, idx otherwise
*/

static int find_TabIDX_composed(char *fullname) {
	int     i;

	if (!fullname || !*fullname)
		return(-1);

	for (i = 0; i < SpecIDNo; i++)
		if (!strcmp(fullname, SpecIDTab[i].spec_version))
			return(i);
	return(-1);
}

static int find_TabIDX(char *spec_version, u32_t pid) {
	int     i, idx = -1;
	char    *fullname;

	if (pid)
		fullname = composeVersion(spec_version, pid);
	else
		fullname = spec_version;

	if (!fullname || !*fullname)
		return(-1);

	for (i = 0; i < SpecIDNo; i++)
		if (!strcmp(fullname, SpecIDTab[i].spec_version)) {
			idx = i;
			break;
		}

	if (fullname != spec_version)
		 free(fullname);

	return(idx);
}

static int find_ArrayIDX(int tab_idx, char *array_name) {
	int     i;
	char    *s;

	if (tab_idx >= SpecIDNo)
		return -1;

	for (i = 0; i < SpecIDTab[tab_idx].arrayno; i++)
		if ((s = SpecIDTab[tab_idx].array_names[i].name))
			if (!strcmp(array_name, s))
				return i;
	return(-1);
}

/*
   Used internally to combine the version and the pid to a string
   Input: version: spec version name
	  pid: u32_t
   returns: NULL if error
            combined string of both version and pid i.e. fourc(12345)
*/
static char *composeVersion(char *version, u32_t pid) {
	int     len;
	char    *comb;

	len = (int) strlen(version) + 10;
	comb = (char *) malloc(len * sizeof(char));
	if (comb == NULL)
		return(NULL);
	sprintf(comb, "%s(%u)", version, pid);
	return(comb);
}

/*
   Used internally to extract the version and the pid from a string
   Input: fullname: i.e. fourc(12345) or just pslits
          version: pointer to char array which has enough space
                   memory for string will have to be freed
	  pid: pointer to s32_t will contain pid
   returns: 1 if fullname contained a pid
            0 if not
*/
static int extractVersion(char *fullname, char *name, u32_t *pid) {
  u32_t pid_buf;
  char version_buf[512];

  if (sscanf(fullname,"%[^(](%u)", version_buf, &pid_buf) == 2) {
    if (name) strcpy(name, version_buf);
    if (pid)  *pid = pid_buf;
    return 1;
  } else {
    if (name) strcpy(name, fullname);
    if (pid)  *pid = 0;
    return 0;
  }
}

/*
  fast test to see if there is a chance that the pid is included in the name
*/

static int iscomposed(char *fullname) {
  return (strchr(fullname, '(')? 1:0);
}

static void delete_SpecIDTab(void)
{
  int i,j;
  for (i = 0; i < SpecIDNo; i++) {
    for (j = 0; j < SpecIDTab[i].arrayno; j++)
      if (SpecIDTab[i].array_names[j].name) {
	free(SpecIDTab[i].array_names[j].name);
	SpecIDTab[i].array_names[j].name = NULL;
      }
    free(SpecIDTab[i].array_names);
    free(SpecIDTab[i].spec_version);
  }
  SpecIDNo = 0;
}

/*
   Create a list of all SpecVersions and their IDs. The result is stored
   in some internal array. This function uses ipcs to get the actual
   running version and produces the internal table.
   Fills in the array SpecIDTab[] with all the Specversions but every
   Specversion does not contain a list of all its arrays yet. To fill
   this in a call to the function SearchSpecArrays(xxx) has to be called
   for every Specversion
*/

static int SearchSpecVersions(void)
{
  u32_t *id_ptr;
  SHM *shm;
  int i, j, n;
  int found;

  delete_SpecIDTab();

  init_ShmIDs();
  if ((SpecIDNo = getShmIDs(&id_ptr, NULL, NULL, SHM_IS_STATUS)) == 0)
    return 0;

  for (n = 0, i = 0; i < SpecIDNo; i++) {
    if ((shm = (SHM *) c_shmat(id_ptr[i], NULL, SHM_RDONLY)) == (SHM *) -1)
      continue;

    /* Find if the name of the spec_version is already used */
    for (found = 0, j = 0; j < n; j++)
      if (strcmp(shm->head.head.spec_version,SpecIDTab[j].spec_version)== 0)
	found++;

    if (!found) {
      SpecIDTab[n].spec_version = (char *) strdup(shm->head.head.spec_version);
    } else {
      SpecIDTab[n].spec_version = composeVersion(shm->head.head.spec_version, shm->head.head.pid);
    }

    SpecIDTab[n].pid = shm->head.head.pid;
    SpecIDTab[n].id = id_ptr[i];
    SpecIDTab[n].arrayno = 0;
    SpecIDTab[n].array_names = NULL;
    SpecIDTab[n].ids_utime = 0;
    n++;
    c_shmdt((void *) shm);
  }

  return SpecIDNo = n;
}


/*
   Fill in the internal memory structure struct specid SpecIDTab[] for the
   specversion fullname. The routine relys on the fact that every SPEC version
   has the status array with a list of shared memory ids which belong to
   this Specversion.
   Input: fullname : Name of the Version with possible pid (fourc, spec(123)
*/

static void SearchSpecArrays(char *fullname) {
  s32_t id;
  int found;
  int redone = 0;
  int i, si, no, idx;
  SHM *shm = NULL;
  struct shm_status *st;

 redo:
  found = ((si = find_TabIDX_composed(fullname)) == -1)? 0:1;

  if (found) {
    shm = (SHM *) c_shmat(SpecIDTab[si].id, NULL, SHM_RDONLY);
  }
  if (!found || !checkSHM(shm, SpecIDTab[si].spec_version, NULL, SHM_IS_STATUS)) {
    if (found && shm && shm != (SHM *) -1)
      c_shmdt((void *) shm);

    if (!redone) {
      redone = 1;
      SearchSpecVersions();
      goto redo;
    } else
      return;
  }

  if (shm->head.head.version < 4)
    st = (struct shm_status *) &(((struct shm_oheader *)shm)->data);
  else
    st = (struct shm_status *) &(shm->data);

  /* Check if there was already an entry for all the arrays and if still
     uptodate */

  if (SpecIDTab[si].arrayno) {
    if (SpecIDTab[si].ids_utime == st->utime) {
      c_shmdt((void *) shm);
      return;
    } else {
      for (i = 0; i < SpecIDTab[si].arrayno; i++)
	if (SpecIDTab[si].array_names[i].name)
	  free(SpecIDTab[si].array_names[i].name);
      free(SpecIDTab[si].array_names);
      SpecIDTab[si].arrayno = 0;
      SpecIDTab[si].array_names = NULL;
    }
  }

  SpecIDTab[si].ids_utime = st->utime;

  for (no = 0, i = 0; i < SHM_MAX_IDS; i++) {
    if (st->ids[i] == -1)
      continue;
    no++;
  }

  SpecIDTab[si].arrayno = no;
  if (no)
    SpecIDTab[si].array_names = (struct arrayid *) malloc(sizeof(struct arrayid) * no);

  /* Make a table of all arraynames with their shm ids */
  for (idx = 0, i = 0; idx < SHM_MAX_IDS; idx++) {
    SHM *shm_array;

    id = st->ids[idx]; /* the shared memory can change while we go through */
    if (id == -1 || i >= no) /* therefore save id and check again i */
      continue;
    shm_array = (SHM *) c_shmat(id, NULL, SHM_RDONLY);
    if (!checkSHM(shm_array, SpecIDTab[si].spec_version, NULL, 0)) {
      SpecIDTab[si].array_names[i].name = NULL;
      SpecIDTab[si].array_names[i].id = 0;
      if (shm_array && shm_array != (SHM *) -1)
	c_shmdt((void *) shm_array);
      i++;
      continue;
    }
    SpecIDTab[si].array_names[i].name =
      (char *) strdup(shm_array->head.head.name);
    SpecIDTab[si].array_names[i].id = id;
    c_shmdt((void *) shm_array);
    i++;
  }

  c_shmdt((void *) shm);
}

static s32_t SearchArrayOnly(char *arrayname) {
  u32_t *id_ptr;
  int no;

  if ((no = getShmIDs(&id_ptr, NULL, arrayname, SHM_IS_ARRAY)) == 0) {
    init_ShmIDs();
    if ((no = getShmIDs(&id_ptr, NULL, arrayname, SHM_IS_ARRAY)) == 0)
      return -1;
  }

  return *id_ptr; /* Return the first array of the list of arrays */
}

/*
   Get a list of all SPEC shared memory ids. This operation uses the external
   program ipcs and is therefore very slow.
   Input: None
   Returns: 1 if error, 0 if OK

   The SHM_INFO/SHM_STAT can bypass ipcs on Linux kernels, when those
   shmctl() commands are supported.
*/

static int init_ShmIDs(void) {
	int     i, id, col = 0;
	SHM     *shm;
	struct  shmid_ds info;
	char    *p, buf[256];
	FILE    *pd;

#if defined(__linux__)
	int     id_cnt, maxid;

# ifndef SHM_STAT
#  define SHM_STAT        13
#  define SHM_INFO        14
# endif

	pd = NULL;
	id_cnt = 0;
	if ((maxid = shmctl(0, SHM_INFO, (void *) buf)) < 0)
#endif
	{
		if ((pd = (FILE*) popen(IPCS,"r")) == NULL)
			return 1;
	}
	id_no = 0;
	for (;;) {
#if defined(__linux__)
		if (maxid == 0)
			return 1;
		if (maxid > 0) {
			if (id_cnt > maxid)
				break;
			id = shmctl(id_cnt++, SHM_STAT, (void *) buf);
		} else
#endif
		{
			if (feof(pd))
				break;
			if (fgets(p = buf, sizeof(buf) - 1, pd) == NULL)
				break;
			while (isspace(*p))
				p++;
			if (col == 0) {
				/* Find column of ipcs output containing ID */
				for (col = 1; *p; col++) {
					/* "shmid" is usual column header on Linux */
					if (!strncasecmp(p, "shmid", 5))
						break;
					/* "ID" is usual column header on Solaris/Mac */
					if (!strncasecmp(p, "ID ", 3))
						break;
					while (*p && !isspace(*p))
						p++;
					while (isspace(*p))
						p++;
				}
				if (!*p)
					col = 0;
				continue;
			}
			if (col == 0)
				continue;
			for (i = 1; i < col; i++) {
				while (*p && !isspace(*p))
					p++;
				while (isspace(*p))
					p++;
			}
			if (sscanf(p, "%d", &id) != 1)
				continue;
		}
		if ((shm = (SHM *) c_shmat(id, NULL, SHM_RDONLY)) == (SHM *) -1)
			continue;

		if (shm->head.head.magic != SHM_MAGIC) {
		      c_shmdt((void *) shm);
		      continue;
		}

#if SHM_CLEANUP
		if (!id_is_our_creation(id)) {
			shmctl(id, IPC_STAT, &info);
			if (info.shm_nattch == 1) {
				c_shmdt((void *)shm);
				shmctl(id, IPC_RMID, NULL);
				continue;
			}
		}
#endif
		if (id_no < SHM_MAX_ENTRIES)
			id_buffer[id_no++] = id;

		c_shmdt((void *)shm);
	}

	if (pd)
		pclose(pd);

	return 0;
}


/*
  Get all shared memory IDs which belong to a certain class. The shared
  memory ids on the system must be filled in first with init_ShmIDs()
  Input: **id_ptr : Will be filled out to a static temporary buffer with
                    all the ids filled in. Do not free this buffer. Do not
		    keep this pointer around for concecutive calls to this
		    routine.
	 spec_version : NULL or a specversion you are interested in
	 name : NULL or an array name you are interested in
	 type : 0 or a bitmap that must be set in the type field of the
	        array (i.e. SHM_STATUS)
 */
static int getShmIDs(u32_t **id_ptr, char *spec_version,
	       char *name, u32_t type) {
  u32_t id;
  int i, ids_no;
  SHM *shm;
  static u32_t ids[SHM_MAX_ENTRIES];

  for (ids_no = 0, i = 0; i < id_no; i++) {
    id = id_buffer[i];

    if ((shm = (SHM *) c_shmat(id, NULL, SHM_RDONLY)) == (SHM *) -1)
      continue;

    if (! checkSHM(shm, spec_version, name, type)) {
      c_shmdt((void *)shm);
      continue;
    }

    c_shmdt((void *)shm);

    if (ids_no < SHM_MAX_ENTRIES) {
      ids[ids_no++] = id;
    }
  }

  *id_ptr = ids;
  return ids_no;
}

/* Delete a list ids from our */
static void delete_id_from_list(u32_t id) {
  int i, j, k, l, no;
  struct arrayid *new_arrays, *old_arrays;
  for (i = 0; i < SpecIDNo; i++) {
    if (SpecIDTab[i].id == id) {
      /* Just set it to 0 if the id is a spec status shared memory */
      SpecIDTab[i].id = 0;
      return;
    }
    for (j = 0; j < SpecIDTab[i].arrayno; j++)
      if (SpecIDTab[i].array_names[j].id == id) {
	/* Delete one id of the array shared memory */
	old_arrays = SpecIDTab[i].array_names;
    if (SpecIDTab[i].array_names[j].name)
	  free(SpecIDTab[i].array_names[j].name);
	no = SpecIDTab[i].arrayno -1;
	if (no) {
	  new_arrays = (struct arrayid *) malloc(no * sizeof(struct arrayid));
	  if (new_arrays == NULL) {
	    SpecIDTab[i].array_names[j].id = 0;
	    SpecIDTab[i].array_names[j].name = NULL;
	    return;
	  }
	  for (k = 0, l = 0; k < SpecIDTab[i].arrayno; k++) {
	    if (k != j) {
	      new_arrays[l].name = old_arrays[k].name;
	      new_arrays[l].id = old_arrays[k].id;
	      l++;
	    }
	  }
	} else
	  new_arrays = NULL;
	SpecIDTab[i].arrayno = no;
	SpecIDTab[i].array_names = new_arrays;
	free(old_arrays);
	return;
      }
  }
}

/*
   Checks if a certain shm pointer belongs to the class of shared memory
   specified
   Input: shm: shared memory pointer to test
          spec_version: The name of the spec version or NULL for all
	  name: The name of the array or NULL for all
	  type: The type of the array (see spec_shm.h)
   Returns: 0 if not passed, 1 if this shm pointer has all the conditions
   given.
*/

static int checkSHM(SHM *shm, char *spec_version, char *name, u32_t type) {
	int     id;
	struct  shmid_ds info;
	char    spec_name[512];
	u32_t   pid;

	if (shm == NULL || shm == (SHM *) -1)
		return(0);

	if (shm->head.head.magic != SHM_MAGIC)
		return(0);

	if (spec_version) {
		if (!iscomposed(spec_version)) {
			if (strcmp(shm->head.head.spec_version, spec_version))
				return(0);
		} else {
			extractVersion(spec_version, spec_name, &pid);
			if (strcmp(shm->head.head.spec_version, spec_name))
				return(0);
			if (shm->head.head.pid != pid)
				return(0);
		}
	}

	if (name && strcmp(shm->head.head.name, name))
		return(0);

	if (type && (type&shm->head.head.flags) != type)
		return(0);

	/* check that id still in system */
	id = shm->head.head.shmid;
	if (shmctl(id, IPC_STAT, &info) < 0)
		return(0);

	/* If we own process we can test if it is still running */
	if (info.shm_perm.uid == getuid() && shm->head.head.pid && kill(shm->head.head.pid, 0) < 0) {
#if SHM_CLEANUP
		if (!id_is_our_creation(id)) {
			if (info.shm_nattch == 1)
				shmctl(id, IPC_RMID, NULL);
			delete_id_from_list(id);
		}
#endif
		return(0);
	}
	return(1);
}


/*
   Attaches to shared memory with given Spec version and array name.
   Input: spec_version: Name of the spec version
	  array: Name of the array inside SPEC
   Returns: NULL if error
            pointer to SPEC shared memory shm_header
*/

static SHM *attachArray(char *fullname, char *array, int read_only) {
  int idx, arr_idx, i, id;
  SHM *shm;

  shm = NULL;
  if (fullname) {
    for (i = 0; i < 2; i++) {
      idx = find_TabIDX_composed(fullname);
      if (idx == -1) {
	SearchSpecVersions();
	if ((idx = find_TabIDX_composed(fullname)) == -1)
	  return NULL;
      }

      arr_idx = find_ArrayIDX(idx, array);
      if (arr_idx == -1) {
	SearchSpecArrays(fullname);
	if ((arr_idx = find_ArrayIDX(idx, array)) == -1)
	  return NULL;
      }

      shm = (SHM *) c_shmat(SpecIDTab[idx].array_names[arr_idx].id, NULL, read_only? SHM_RDONLY:0);

      /* We might not be attached because the id is not longer valid */
      if (shm != (SHM *) -1)
	break;

      SearchSpecVersions(); /* and retry */
    }
  } else { /* Use another method of finding the shared mem */
    if ((id = SearchArrayOnly(array)) != -1)
      shm = (SHM *) c_shmat(id, NULL, read_only? SHM_RDONLY:0);

    /* We might not be attached */
    if (shm == (SHM *) -1)
      return NULL;
    else
      return shm;
  }

  if (!checkSHM(shm, fullname, array, 0)) {
    if (shm && shm != (SHM *) -1)
      c_shmdt((void *) shm);
    return NULL;
  }

  return shm;
}

/*
   Attaches to shared memory of a given Spec version
   Input: spec_version: Name of the spec version
   Returns: NULL if error
            pointer to SPEC shared memory shm_header
*/

static SHM *attachSpec(char *fullname) {
  int idx = -1;
  SHM *shm;

  /* Search in out index - if already known */
  idx = find_TabIDX_composed(fullname);

  /* If not in our index then redo the search in the system */
  if (idx == -1) {
    SearchSpecVersions();
    if ((idx = find_TabIDX_composed(fullname)) == -1)
      return NULL;
  }

  /* Maybe we find a pointer in our table */

  /* Attach to the shared memory in read-only mode */
  shm = (SHM *) c_shmat(SpecIDTab[idx].id, NULL, SHM_RDONLY);

  /* Check that the shared memory is still valid */
  if (!checkSHM(shm, fullname, NULL, 0)) {
    if (shm && shm != (SHM *) -1)
      c_shmdt((void *) shm);
    return NULL;
  }

  return shm;
}


/* How many bytes per data type -- perhaps a table would be smarter ... */
static size_t typedsize(int t) {
	switch (t) {
	  case SHM_USHORT: return(sizeof(unsigned short));
	  case SHM_ULONG:  return(sizeof(u32_t));
	  case SHM_ULONG64: return(sizeof(u64_t));
	  case SHM_SHORT:  return(sizeof(short));
	  case SHM_LONG:   return(sizeof(s32_t));
	  case SHM_LONG64: return(sizeof(s64_t));
	  case SHM_UCHAR:  return(sizeof(unsigned char));
	  case SHM_CHAR:   return(sizeof(char));
	  case SHM_STRING: return(sizeof(char));
	  case SHM_DOUBLE: return(sizeof(double));
	  case SHM_FLOAT:  return(sizeof(float));
	  default:        return(0);
	}
}

int SPS_Size(int t) {
  return (int) typedsize(t);
}

/*
 *  If np < 0, if rev < 0 "to" pointer points to end of
 *  destination buffer and fills in reverse order, else
 *  if rev > 0 "from" pointer points to end of destination buffer.
 *  if np > 0: rev == 0 normal copy
 *             rev == 1 offset in from in every point (copy cols from shm)
 *             rev == 2 offset in to in every point  (copy cols to shm)
*/
#define ONECP(tto, tfrom) do {\
	tto     *a = (tto *)   t; \
	tfrom   *b = (tfrom *) f; \
\
	n = np; \
	offs = offset; \
	if (n > 0) { \
		if (rev == 0) \
			while (n--) \
				*a++ = (tto) *b++; \
		else if (rev == 1) \
			while (n--) { \
				*a++ = (tto) *b; \
				b += offs; \
			} \
		else if (rev == 2) \
			while (n--) { \
				*a = (tto) *b++; \
				a += offs; \
			} \
	} else if (rev < 0) \
		 while (n++) \
			*a-- = (tto) *b++; \
	else \
		while (n++) \
			*a++ = (tto) *b--; \
} while (0)

static int typedcp(void *t, int tt, void *f, int ft, int np, int rev, int offset) {
	int     n, offs;

	if (np == 0)
		return(0);

	if (ft == tt && np > 0 && rev == 0) {
		memcpy(t, f, np * typedsize(ft));
		return(0);
	}

	switch (tt) {
	 case SHM_LONG:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(s32_t, double);                  break;
	    case SHM_FLOAT:   ONECP(s32_t, float);                   break;
	    case SHM_ULONG:   ONECP(s32_t, u32_t);                   break;
	    case SHM_ULONG64: ONECP(s32_t, u64_t);                   break;
	    case SHM_USHORT:  ONECP(s32_t, unsigned short);          break;
	    case SHM_UCHAR:   ONECP(s32_t, unsigned char);           break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(s32_t, char);                    break;
	    case SHM_SHORT:   ONECP(s32_t, short);                   break;
	    case SHM_LONG:    ONECP(s32_t, s32_t);                   break;
	    case SHM_LONG64:  ONECP(s32_t, s64_t);                   break;
	   }
	   break;
	 case SHM_ULONG:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(u32_t, double);                  break;
	    case SHM_FLOAT:   ONECP(u32_t, float);                   break;
	    case SHM_ULONG:   ONECP(u32_t, u32_t);                   break;
	    case SHM_ULONG64: ONECP(u32_t, u64_t);                   break;
	    case SHM_USHORT:  ONECP(u32_t, unsigned short);          break;
	    case SHM_UCHAR:   ONECP(u32_t, unsigned char);           break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(u32_t, char);                    break;
	    case SHM_SHORT:   ONECP(u32_t, short);                   break;
	    case SHM_LONG:    ONECP(u32_t, s32_t);                   break;
	    case SHM_LONG64:  ONECP(u32_t, s64_t);                   break;
	   }
	   break;
	 case SHM_LONG64:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(s64_t, double);                  break;
	    case SHM_FLOAT:   ONECP(s64_t, float);                   break;
	    case SHM_ULONG:   ONECP(s64_t, u32_t);                   break;
	    case SHM_ULONG64: ONECP(s64_t, u64_t);                   break;
	    case SHM_USHORT:  ONECP(s64_t, unsigned short);          break;
	    case SHM_UCHAR:   ONECP(s64_t, unsigned char);           break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(s64_t, char);                    break;
	    case SHM_SHORT:   ONECP(s64_t, short);                   break;
	    case SHM_LONG:    ONECP(s64_t, s32_t);                   break;
	    case SHM_LONG64:  ONECP(s64_t, s64_t);                   break;
	   }
	   break;
	 case SHM_ULONG64:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(u64_t, double);                  break;
	    case SHM_FLOAT:   ONECP(u64_t, float);                   break;
	    case SHM_ULONG:   ONECP(u64_t, u32_t);                   break;
	    case SHM_ULONG64: ONECP(u64_t, u64_t);                   break;
	    case SHM_USHORT:  ONECP(u64_t, unsigned short);          break;
	    case SHM_UCHAR:   ONECP(u64_t, unsigned char);           break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(u64_t, char);                    break;
	    case SHM_SHORT:   ONECP(u64_t, short);                   break;
	    case SHM_LONG:    ONECP(u64_t, s32_t);                   break;
	    case SHM_LONG64:  ONECP(u64_t, s64_t);                   break;
	   }
	   break;
	 case SHM_USHORT:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(unsigned short, double);         break;
	    case SHM_FLOAT:   ONECP(unsigned short, float);          break;
	    case SHM_ULONG:   ONECP(unsigned short, u32_t);          break;
	    case SHM_ULONG64: ONECP(unsigned short, u64_t);          break;
	    case SHM_USHORT:  ONECP(unsigned short, unsigned short); break;
	    case SHM_UCHAR:   ONECP(unsigned short, unsigned char);  break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(unsigned short, char);           break;
	    case SHM_SHORT:   ONECP(unsigned short, short);          break;
	    case SHM_LONG:    ONECP(unsigned short, s32_t);          break;
	    case SHM_LONG64:  ONECP(unsigned short, s64_t);          break;
	   }
	   break;
	 case SHM_UCHAR:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(unsigned char, double);          break;
	    case SHM_FLOAT:   ONECP(unsigned char, float);           break;
	    case SHM_ULONG:   ONECP(unsigned char, u32_t);           break;
	    case SHM_ULONG64: ONECP(unsigned char, u64_t);           break;
	    case SHM_USHORT:  ONECP(unsigned char, unsigned short);  break;
	    case SHM_UCHAR:   ONECP(unsigned char, unsigned char);   break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(unsigned char, char);            break;
	    case SHM_SHORT:   ONECP(unsigned char, short);           break;
	    case SHM_LONG:    ONECP(unsigned char, s32_t);           break;
	    case SHM_LONG64:  ONECP(unsigned char, s64_t);           break;
	   }
	   break;
	 case SHM_SHORT:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(short, double);                  break;
	    case SHM_FLOAT:   ONECP(short, float);                   break;
	    case SHM_ULONG:   ONECP(short, u32_t);                   break;
	    case SHM_ULONG64: ONECP(short, u64_t);                   break;
	    case SHM_USHORT:  ONECP(short, unsigned short);          break;
	    case SHM_UCHAR:   ONECP(short, unsigned char);           break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(short, char);                    break;
	    case SHM_SHORT:   ONECP(short, short);                   break;
	    case SHM_LONG:    ONECP(short, s32_t);                   break;
	    case SHM_LONG64:  ONECP(short, s64_t);                   break;
	   }
	   break;
	 case SHM_STRING:  /*FALLTHROUGH*/
	 case SHM_CHAR:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(char, double);                   break;
	    case SHM_FLOAT:   ONECP(char, float);                    break;
	    case SHM_ULONG:   ONECP(char, u32_t);                    break;
	    case SHM_ULONG64: ONECP(char, u64_t);                    break;
	    case SHM_USHORT:  ONECP(char, unsigned short);           break;
	    case SHM_UCHAR:   ONECP(char, unsigned char);            break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(char, char);                     break;
	    case SHM_SHORT:   ONECP(char, short);                    break;
	    case SHM_LONG:    ONECP(char, s32_t);                    break;
	    case SHM_LONG64:  ONECP(char, s64_t);                    break;
	   }
	   break;
	 case SHM_FLOAT:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(float, double);                  break;
	    case SHM_FLOAT:   ONECP(float, float);                   break;
	    case SHM_ULONG:   ONECP(float, u32_t);                   break;
	    case SHM_ULONG64: ONECP(float, u64_t);                   break;
	    case SHM_USHORT:  ONECP(float, unsigned short);          break;
	    case SHM_UCHAR:   ONECP(float, unsigned char);           break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(float, char);                    break;
	    case SHM_SHORT:   ONECP(float, short);                   break;
	    case SHM_LONG:    ONECP(float, s32_t);                   break;
	    case SHM_LONG64:  ONECP(float, s64_t);                   break;
	   }
	   break;
	 case SHM_DOUBLE:
	   switch (ft) {
	    case SHM_DOUBLE:  ONECP(double, double);                 break;
	    case SHM_FLOAT:   ONECP(double, float);                  break;
	    case SHM_ULONG:   ONECP(double, u32_t);                  break;
	    case SHM_ULONG64: ONECP(double, u64_t);                  break;
	    case SHM_USHORT:  ONECP(double, unsigned short);         break;
	    case SHM_UCHAR:   ONECP(double, unsigned char);          break;
	    case SHM_STRING:  /*FALLTHROUGH*/
	    case SHM_CHAR:    ONECP(double, char);                   break;
	    case SHM_SHORT:   ONECP(double, short);                  break;
	    case SHM_LONG:    ONECP(double, s32_t);                  break;
	    case SHM_LONG64:  ONECP(double, s64_t);                  break;
	   }
	   break;
	}
	return(0);
}

/*
  char *SPS_GetNextSpec(int flag)
  Input: Flag to know if this is the first call to SPS_GetNextSpec
         1: Get first in list, 0: Get next
  Returns: Name of the SPEC version or NULL if no more in the list
*/
char *SPS_GetNextSpec(int flag) {
  static int loop_count = 0;
  if (flag == 0) {
    SearchSpecVersions();
    loop_count = 0;
  } else
    loop_count++;

  if (loop_count >= SpecIDNo) {
    loop_count = 0;
    return NULL;
  } else
    return SpecIDTab[loop_count].spec_version;
}


/*
  char *SPS_GetNextArray(char *version, int flag)
  Input: version : name of SPEC version, or NULL for all.
         Flag to know if this is the first call to SPS_GetNextArray
         1: Get first in list, 0: Get next
  Returns: Name of the SPEC array name or NULL if no more in the list
*/

char *SPS_GetNextArray(char *fullname, int flag) {
  static int loop_count = 0;
  int idx = -1;

  if (fullname == NULL)
    return GetNextAll(flag);

  if (flag == 0) {
    SearchSpecArrays(fullname);
    loop_count = 0;
  } else
    loop_count++;

  idx = find_TabIDX_composed(fullname);

  if (idx == -1 || loop_count >= SpecIDTab[idx].arrayno ||
      SpecIDTab[idx].array_names[loop_count].name == NULL) {
    loop_count = 0;
    return NULL;
  } else
    return SpecIDTab[idx].array_names[loop_count].name;
}

static char *GetNextAll(int flag) {
  static int loop_count = 0;
  static char *spec_version = NULL;
  int idx = -1;

  for (;;) {
    if (flag == 0 || spec_version == NULL) {
      loop_count = 0;
      if ((spec_version = SPS_GetNextSpec(flag)) == NULL) {
	return NULL;
      }
      SearchSpecArrays(spec_version);
    } else
      loop_count++;

    idx = find_TabIDX_composed(spec_version);

    if (idx == -1 || loop_count >= SpecIDTab[idx].arrayno ||
	SpecIDTab[idx].array_names[loop_count].name == NULL) {
      spec_version = NULL;
      flag = 1;
      continue;
    } else
      return SpecIDTab[idx].array_names[loop_count].name;
  }
}

/*
   Read the state the particular SPEC version is in for the moment
   Input: version : specversion with PID if necessary (spec(1234) or fourc)
   Returns: State
*/
s32_t SPS_GetSpecState(char *version) {
  SHM *spec_shm;
  s32_t state = 0;
  struct shm_status *st;
  SPS_ARRAY private_shm;
  int was_attached;

  if ((private_shm = convert_to_handle(version, NULL)) == NULL)
    return -1;

  /* private_shm->stay_attached = 1; Always stay attached to the status */
  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 0))
    return -1;

  spec_shm = private_shm->shm;

  if (spec_shm) {
    if (spec_shm->head.head.version < 4)
      st = (struct shm_status *) &(((struct shm_oheader *)spec_shm)->data);
    else
      st = (struct shm_status *) &(spec_shm->data);
    state = st->spec_state;
  }

  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return state;
}

static SPS_ARRAY add_private_shm(SHM *shm, char *fullname, char *array, int write_flag) {
  SPS_ARRAY private_shm;
  private_shm = (SPS_ARRAY) malloc(sizeof(struct sps_array));
  if (private_shm == NULL) {
    return NULL;
  }
  private_shm->shm        = shm;
  if (shm) {
    private_shm->attached   = 1;
    private_shm->utime      = 0xffffffff; /*Called when creating shared mem */
    private_shm->id         = shm->head.head.shmid;
    private_shm->write_flag = write_flag;
  } else {
    private_shm->attached   = 0;
    private_shm->id         = 0;
    private_shm->write_flag = 0;
    private_shm->utime      = 0xffffffff; /* Called when conv. from handle */
  }

  private_shm->spec       = fullname ? (char *) strdup(fullname) : NULL;
  private_shm->array      = array ? (char *) strdup(array) : NULL;

  private_shm->private_data_copy = NULL;
  private_shm->buffer_len = 0;
  private_shm->private_info_copy = NULL;
  private_shm->private_meta_copy = NULL;
  private_shm->meta_len = 0;
  private_shm->stay_attached   = 0;
  return private_shm;
}

static SPS_ARRAY
convert_to_handle(char *spec_version, char *array_name) {
  struct shm_created *shm_list;
  SPS_ARRAY private_shm;

  if (spec_version == NULL && array_name == NULL)
    return NULL; /* Why would somebody want to do that */

  shm_list = ll_find_array(spec_version, array_name, array_name? 0:1);

  /* If not in list then put it into the list if it is not the generic
     spec_version (NULL) */

 if (shm_list == 0) {
    private_shm = add_private_shm(NULL, spec_version, array_name, 0);
    shm_list = ll_addnew_array(spec_version, array_name,
				  array_name? 0:1, NULL, 0, 0, NULL);
    shm_list->handle = private_shm;
  } else {
    private_shm = shm_list->handle;
    if (shm_list->spec_version == NULL && private_shm->spec)
      shm_list->spec_version = (char *) strdup(private_shm->spec);
  }
  return private_shm;
}

#if SPS_DEBUG
/*
   Attaches to a SPEC array. Returns a private opaque structure which user
   should not modify. To get the real data call
   Input: fullname specversion with PID if necessary (spec(1234) or fourc)
          array_name: The name of the SPEC array (i.e. MCA_DATA)
	  write_flag: One if you intend to modify the shared memory
   Output: NULL if error
	   opaque structure to attached memory
*/

static int SPS_AttachToArray(char *fullname, char *array, int write_flag) {
  SPS_ARRAY private_shm;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return 1;
  private_shm->write_flag = write_flag;
  private_shm->stay_attached = 1;
  if (private_shm)
    ReconnectToArray(private_shm, 0);
  return 0;
}

/*
   Detaches from a SPEC array. The opaque pointer private_shm stays valid.
   If it is used in further calls the functions automatically try to reattach
   to the SPEC array.
   Input: fullname: Spec name
          array name of the array
   Returns: 1 error occured
            0 everything OK
*/

static int SPS_DetachFromArray(char *fullname, char*array) {
  SPS_ARRAY private_shm;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return 1;
  private_shm->stay_attached = 0;
  return DeconnectArray(private_shm);
}
#endif /* SPS_DEBUG */

static int DeconnectArray(SPS_ARRAY private_shm) {
  if (private_shm->attached) {
    c_shmdt((void *)private_shm->shm);
    private_shm->attached = 0;
    private_shm->shm = NULL;
    private_shm->pointer_got_count = 0;
  }
  return 0;
}

/*
   Reconnects to a shared memory SPEC array. The process is attached to this
   shared memory again.
   Input: Opac pointer from ConnectToArray
   returns: 1 error
            0 success
*/

static int ReconnectToArray(SPS_ARRAY private_shm, int write_flag) {
  SHM *shm;

  if (write_flag && !private_shm->write_flag) { /*Reattach with write flag */
    private_shm->write_flag = 1;
    shm = private_shm->shm;
    if (shm) {
      c_shmdt((void *) shm);
      private_shm->attached = 0;
      private_shm->shm = NULL;
    }
  }

  if (!private_shm->attached) {
    if (private_shm->id != 0) {
      shm = c_shmat(private_shm->id, NULL, private_shm->write_flag? 0:SHM_RDONLY);
    } else
      shm = NULL;
  } else
    shm = private_shm->shm;

  /*  Check shm and try to get a new one if old one not longer valid */
  if (!checkSHM(shm, private_shm->spec, private_shm->array, 0)) {
    if (shm && shm != (SHM *) -1) {
      c_shmdt((void *) shm);
      private_shm->attached = 0;
      private_shm->shm = NULL;
    }

    if (private_shm->array) {
      if ((shm = attachArray(private_shm->spec, private_shm->array,
			      private_shm->write_flag? 0:1)) == NULL)
	return 1;
    } else {
      if ((shm = attachSpec(private_shm->spec)) == NULL)
	return 1;
    }
  }

  private_shm->shm        = shm;
  private_shm->attached  = (shm == NULL)? 0:1;
  private_shm->id         = shm->head.head.shmid;
  /* If we are attached and the SPEC version is still unknown - update now */
  if (shm && private_shm->spec == NULL)
    private_shm->spec = (char *) strdup(shm->head.head.spec_version);

  return 0;
}

/*
   Gives you the pointer to the data area of the SPEC array. If the process
   is not currently attached it will be attached after the call.
   Input: fullname : Spec version
          array : Name of the array
   Returns: NULL error
            void * to the data area. Do not remember this pointer as most of
	    the SPS_ functions can change this pointer (In case the other
	    party quit and recreated the shared memory)

*/
void *SPS_GetDataPointer(char *fullname, char *array, int write_flag) {
  SPS_ARRAY private_shm;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return NULL;
  if (ReconnectToArray(private_shm, write_flag))
    return NULL;

  private_shm->pointer_got_count++;
  if (private_shm->shm->head.head.version < 4)
    return &(((struct shm_oheader *)(private_shm->shm))->data);
  return &(private_shm->shm->data);
}

/*
   You should return the pointer to the library if you do not use it anymore.
   If you returned the pointer as many times as you got it you will be
   detached from the shared memory and the pointer is not valid anymore.
   Input: fullname : Spec version
          array : Name of the array
   Returns: 0 success
            1 error
*/

int SPS_ReturnDataPointer(void *data) {
  SPS_ARRAY private_shm;
  struct shm_created *created;
  struct shm_head *sh;
  SHM *shm;

  /* Try old header size first, since it is smaller */
  sh = (struct shm_head *) (((char *) data) - SHM_OHEAD_SIZE);
  if (sh->magic != SHM_MAGIC)
     sh = (struct shm_head *) (((char *) data) - SHM_HEAD_SIZE);
  if (sh->magic != SHM_MAGIC)
     return(1);

  shm = (SHM *) sh;

  if ((created = ll_find_pointer(shm)) == NULL)
    return 1;

  if ((private_shm = created->handle) == NULL)
    return 1;

  private_shm->pointer_got_count--;

  if (private_shm->pointer_got_count <= 0) {
    private_shm->pointer_got_count = 0;
    return DeconnectArray(private_shm);
  }
  return 0;
}

/*
   Copies the data from the shared memory SPEC array to the user's buffer.
   The type of the data in the shared array and the type the user wants
   in his buffer is taken into account. The routine should work efficient
   even if both types are equal. The user has to provide the number of
   items in the buffer. This number is used to check for buffer overflow.
   If a possible buffer overflow is detected only the items_in_buffer
   are copied.
   Input: fullname : name of the specversion
          array: name of the array in SPEC
          buffer: pointer to our buffer
	  my_type: In which format do we want the results
	  items_in_buffer: buffersize in number of items of type my_type
	                   in buffer
   Returns: 0 success
	    1 overflow, but copy done
           -1 error Nothing done
*/

int SPS_CopyFromShared(char *fullname, char *array, void *buffer, int my_type,
		      int items_in_buffer) {
  return TypedCopy(fullname, array, buffer, my_type, items_in_buffer, 0);
}

/*
   Copies the data to the shared memory SPEC array from the user's buffer.
   The type of the data in the shared array and the type the user wants
   in his buffer is taken into account. The routine should work efficient
   even if both types are equal. The user has to provide the number of
   items in the buffer. This number is used to check for buffer overflow.
   If a possible buffer overflow is detected only the items_in_buffer
   are copied.
   Input: fullname : name of the specversion
          array: name of the array in SPEC
          buffer: pointer to our buffer
	  my_type: In which format do we want the results
	  items_in_buffer: buffersize in number of items of type my_type
	                   in buffer
   Returns: 0 success
	    1 overflow, but copy done
           -1 error Nothing done
*/

int SPS_CopyToShared(char *fullname, char *array, void *buffer, int my_type,
		      int items_in_buffer) {
  return TypedCopy(fullname, array, buffer, my_type, items_in_buffer, 1);
}

/*
   Copies the data from the shared memory SPEC array to the user's buffer.
   The type of the data in the shared array and the type the user wants
   in his buffer is taken into account. The routine should work efficient
   even if both types are equal. The user has to provide the number of
   items in the buffer. This number is used to check for buffer overflow.
   If a possible buffer overflow is detected only the items_in_buffer
   are copied.
   Input: fullname : name of the specversion
          array: name of the array in SPEC
          buffer: pointer to our buffer
	  my_type: In which format do we want the results
	  items_in_buffer: buffersize in number of items of type my_type
	                   in buffer
   Returns: 0 success
	    1 overflow, but copy done
           -1 error Nothing done
*/

static int TypedCopy(char *fullname, char *array, void *buffer, int my_type,
		      int items_in_buffer, int direction) {
  SPS_ARRAY private_shm;
  int was_attached, overflow;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return -1;
  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, direction))
    return -1;

  overflow = typedcp_private(private_shm, buffer, my_type, items_in_buffer,
		     direction);

  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return overflow;
}

static int typedcp_private(SPS_ARRAY private_shm, void *buffer, int my_type,
			int items_in_buffer, int direction) {
  void *data_ptr;
  int rows, cols, type, n_to_copy, overflow;

  /* Do the data copy taking type and my_type into account */
  rows = private_shm->shm->head.head.rows;
  cols = private_shm->shm->head.head.cols;
  type = private_shm->shm->head.head.type;
  if (private_shm->shm->head.head.version < 4)
    data_ptr = &(((struct shm_oheader *)(private_shm->shm))->data);
  else
    data_ptr = &(private_shm->shm->data);
  if (rows * cols > items_in_buffer) {
    overflow = 1;
    n_to_copy = items_in_buffer;
  } else {
    overflow = 0;
    n_to_copy = rows*cols;
  }

  if (direction) {
    typedcp(data_ptr, type, buffer, my_type, n_to_copy, 0, 0);
    private_shm->shm->head.head.utime++; /* Updated */
  } else {
    typedcp(buffer, my_type, data_ptr, type, n_to_copy, 0, 0);
  }
  return overflow;
}

/* Can be used to copy rows and columns from an to shared memory.
   The first parameter are standard see SPS_GetDataRow.
   use_row is 1 if it should copy rows. (One block of memory)
   direction is 1 if it should write to shared memory and 0 to read from shm
   my_buffer is NULL if it should copy to the internal buffer otherwise the
             buffer to copy from or to
*/
static void *CopyDataRC(char *fullname, char *array, int my_type,
			 int row, int col, int *act_copied,
			 int use_row, int direction, void *my_buffer) {
  int rows, cols, type;
  void *buffer = NULL;
  SPS_ARRAY private_shm;
  int was_attached;
  size_t size;
  void *data_ptr;
  int n_to_copy = 0;

  if (act_copied)
    *act_copied = 0;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return NULL;

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, direction))
    return NULL;

  rows = private_shm->shm->head.head.rows;
  cols = private_shm->shm->head.head.cols;
  type = private_shm->shm->head.head.type;

  if ((use_row && (row < 0 || row >= rows)) ||
      (!use_row && (col < 0 || col >= cols)))
    return NULL;

  size  = (use_row? cols:rows) * (int) typedsize(my_type); /* full row/column */

  if (my_buffer == NULL) {
    if (private_shm->private_data_copy == NULL ||
	 size > private_shm->buffer_len) {
      if (size > private_shm->buffer_len) {
	free(private_shm->private_data_copy);
	private_shm->private_data_copy = NULL;
	private_shm->buffer_len = 0;
      }

      if ((buffer = (void *) malloc(size)) == NULL)
	goto error;
      private_shm->private_data_copy = buffer;
      private_shm->buffer_len = size;
    } else
      buffer = private_shm->private_data_copy;
  } else
    buffer = my_buffer;

  if (private_shm->shm->head.head.version < 4)
    data_ptr = &(((struct shm_oheader *)(private_shm->shm))->data);
  else
    data_ptr = &(private_shm->shm->data);


  /* Do the data copy taking type and my_type into account */
  if (use_row) {
    data_ptr = (void *) ((char*) data_ptr + cols * row * typedsize(my_type));
    if (col == 0 || col > cols)
      n_to_copy = cols;
    else
      n_to_copy = col;

    if (direction) {
      typedcp(data_ptr, type, buffer, my_type, n_to_copy, 0, 0);
      private_shm->shm->head.head.utime++; /* Updated */
    } else {
      typedcp(buffer, my_type, data_ptr, type, n_to_copy, 0, 0);
    }

  } else {
    data_ptr = (void *) ((char*) data_ptr + col * typedsize(my_type));
    if (row == 0 || row > rows)
      n_to_copy = rows;
    else
      n_to_copy = row;

    if (direction) {
      typedcp(data_ptr, type, buffer, my_type, n_to_copy, 2, cols);
      private_shm->shm->head.head.utime++; /* Updated */
    } else {
      typedcp(buffer, my_type, data_ptr, type, n_to_copy, 1, cols);
    }

  }

 error:
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  if (act_copied)
    *act_copied = n_to_copy;

  return buffer;
}

/*
  Copy a row of data from the shared memory array to a private buffer. Only
  one private buffer per shared memory array is allowed. You can call
  SPS_FreeDataCopy to free the memory by the buffer but it is not necessary.
  If the buffer is not freed it is reused the next time you call the
  function SPS_GetDataRow for fullname-array.
  Input:
         name : The name of the specversion
	 array : The name of the array in this SPEC version
	 my_type : The data type how you would like to have the buffer. This
	           data type does not have to be the same as the data type
		   of the shared memory array. In this case it will be
		   transformed in the data type you asked for (fast)
	 row : integer. Which row do you want.
	 col : integer. How many columns to you want. If this is 0 then all
	       the columns are copied.
	 act_cols : pointer to int. If not NULL then this is filled with the
               number of columns actually copied.
  Returns: Pointer to the data array (DO NOT USE free() to free this buffer)
*/

void *SPS_GetDataRow(char *name, char *array, int my_type,
			int row, int col, int *act_cols) {
  return CopyDataRC(name, array, my_type, row, col, act_cols, 1, 0, NULL);
}

/*
  Copy a column of data from the shared memory array to a private buffer.
  Only one private buffer per shared memory array is allowed. You can call
  SPS_FreeDataCopy to free the memory by the buffer but it is not necessary.
  If the buffer is not freed it is reused the next time you call the
  function SPS_GetDataCol, SPS_GetDataRow or SPS_GetData for fullname-array.
  Input:
         name : The name of the specversion
	 array : The name of the array in this SPEC version
	 my_type : The data type how you would like to have the buffer. This
	           data type does not have to be the same as the data type
		   of the shared memory array. In this case it will be
		   transformed in the data type you asked for (fast)
	 col : integer. Which column do you want.
	 row : integer. How many rows you want. If this is 0 then all
	       the rows are copied.
	 act_rows : pointer to int. If not NULL then this is filled with the
               number of rows actually copied.
  Returns: Pointer to the data array (DO NOT USE free() to free this buffer)
*/

void *SPS_GetDataCol(char *name, char *array, int my_type,
			int col, int row, int *act_rows) {
  return CopyDataRC(name, array, my_type, row, col, act_rows, 0, 0, NULL);
}

/*
  Copy a row of data from the shared memory array to a buffer.
  Input:
         name : The name of the specversion
	 array : The name of the array in this SPEC version
	 buf : A pointer to the buffer.
         my_type : The data type how you would like to have the buffer. This
	           data type does not have to be the same as the data type
		   of the shared memory array. In this case it will be
		   transformed in the data type you asked for (fast)
	 row : integer. Which row do you want.
	 col : integer. How many columns you want to copy. If this is 0 then
	       all the columns are copied.
	 act_cols : pointer to int. If not NULL then this is filled with the
               number of columns actually copied.
  Returns: 0 success
           1 error
*/

int SPS_CopyRowFromShared(char *name, char *array, void *buf,
     int my_type, int row, int col, int *act_cols) {
  return
   (CopyDataRC(name, array, my_type, row, col, act_cols, 1, 0, buf)? 0:1);
}

/*
  Copy a column of data from the shared memory array to a buffer.
  Input:
         name : The name of the specversion
	 array : The name of the array in this SPEC version
	 buf : A pointer to the buffer.
         my_type : The data type how you would like to have the buffer. This
	           data type does not have to be the same as the data type
		   of the shared memory array. In this case it will be
		   transformed in the data type you asked for (fast)
	 col : integer. Which column do you want.
	 row : integer. How many rows you want to copy. If this is 0 then
	       all the columns are copied.
	 act_rows : pointer to int. If not NULL then this is filled with the
               number of rows actually copied.
  Returns: 0 success
           1 error
*/

int SPS_CopyColFromShared(char *name, char *array, void *buf,
     int my_type, int col, int row, int *act_rows) {
  return
   (CopyDataRC(name, array, my_type, row, col, act_rows, 0, 0, buf)? 0:1);
}

/*
  Copy a row of data to the shared memory array from a buffer.
  Input:
         name : The name of the specversion
	 array : The name of the array in this SPEC version
	 buf : A pointer to the buffer.
         my_type : The data type how you would like to have the buffer. This
	           data type does not have to be the same as the data type
		   of the shared memory array. In this case it will be
		   transformed in the data type you asked for (fast)
	 row : integer. Which row do you want.
	 col : integer. How many columns you want to copy. If this is 0 then
	       all the columns are copied.
	 act_cols : pointer to int. If not NULL then this is filled with the
               number of columns actually copied.
  Returns: 0 success
           1 error
*/

int SPS_CopyRowToShared(char *name, char *array, void *buf,
     int my_type, int row, int col, int *act_cols) {
  return
   (CopyDataRC(name, array, my_type, row, col, act_cols, 1, 1, buf)? 0:1);
}

/*
  Copy a column of data to the shared memory array from a buffer.
  Input:
         name : The name of the specversion
	 array : The name of the array in this SPEC version
	 buf : A pointer to the buffer.
         my_type : The data type how you would like to have the buffer. This
	           data type does not have to be the same as the data type
		   of the shared memory array. In this case it will be
		   transformed in the data type you asked for (fast)
	 col : integer. Which column do you want.
	 row : integer. How many rows you want to copy. If this is 0 then
	       all the columns are copied.
	 act_rows : pointer to int. If not NULL then this is filled with the
               number of rows actually copied.
  Returns: 0 success
           1 error
*/

int SPS_CopyColToShared(char *name, char *array, void *buf,
     int my_type, int col, int row, int *act_rows) {
  return
   (CopyDataRC(name, array, my_type, row, col, act_rows, 0, 1, buf)? 0:1);
}

/*
  Copy the data in the shared memory array to a private buffer. Only one
  private buffer per shared memory array is allowed. You can call
  SPS_FreeDataCopy to free the memory by the buffer but it is not necessary.
  If the buffer is not freed it is reused the next time you call the
  function SPS_GetDataCopy for fullname-array.
  Input:
         fullname : The name of the specversion
	 array : The name of the array in this SPEC version
	 my_type : The data type how you would like to have the buffer. This
	           data type does not have to be the same as the data type
		   of the shared memory array. In this case it will be
		   transformed in the data type you asked for (fast)
	 rows : pointer to integer. Will be filled with number of rows
	        if not NULL
	 cols : pointer to integer. Will be filled with number of cols
	        if not NULL
  Returns: Pointer to the data array (DO NOT USE free() to free this buffer)
*/

void *SPS_GetDataCopy(char *fullname, char *array, int my_type,
			int *rows_ptr, int *cols_ptr) {
  int rows, cols;
  void *buffer = NULL;
  SPS_ARRAY private_shm;
  int allocated = 0;
  int was_attached;
  size_t size;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return NULL;

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 0))
    return NULL;

  rows = private_shm->shm->head.head.rows;
  cols = private_shm->shm->head.head.cols;

  if (rows_ptr)
    *rows_ptr = rows;
  if (cols_ptr)
    *cols_ptr = cols;

  size  = rows * cols * typedsize(my_type);

  if (private_shm->private_data_copy == NULL ||
       size > private_shm->buffer_len) {
    if (size > private_shm->buffer_len) {
      free(private_shm->private_data_copy);
      private_shm->private_data_copy = NULL;
      private_shm->buffer_len = 0;
    }

    if ((buffer = (void *) malloc(size)) == NULL)
      goto error;
    allocated = 1;
    private_shm->private_data_copy = buffer;
    private_shm->buffer_len = size;
  }

  if (typedcp_private(private_shm, private_shm->private_data_copy,
		       my_type, rows * cols, 0)) {
    if (allocated) {
      free(buffer);
      buffer = NULL;
    }
  } else
    buffer = private_shm->private_data_copy;

 error:
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return buffer;
}

int SPS_FreeDataCopy(char *fullname, char *array) {
  SPS_ARRAY private_shm;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return 1;

  if (private_shm && private_shm->private_data_copy != NULL) {
    free(private_shm->private_data_copy);
    private_shm->private_data_copy = NULL;
    private_shm->buffer_len = 0;
  }
  return 0;
}

int SPS_PutMetaData(char *fullname, char *array, char *meta, u32_t length) {
  SPS_ARRAY private_shm;
  int was_attached, ret = 0, len;
  struct shm_head *sh;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return(-1);

  if (meta == NULL)
    return(-1);

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 1))
    return(-1);

  sh = &(private_shm->shm->head.head);

  if (sh->version < 6) {
    ret = -1;
    goto error;
  }
  if (length > sh->meta_length)
	len = sh->meta_length;
  else
	len = length;

  memcpy((char *) (private_shm->shm) + sh->meta_start, meta, len);

error:
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return(ret);
}

char *SPS_GetMetaData(char *fullname, char *array, u32_t *length) {
  void *buffer = NULL;
  SPS_ARRAY private_shm;
  int was_attached;
  u32_t size;
  struct shm_head *sh;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return NULL;

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 0))
    return NULL;

  sh = &(private_shm->shm->head.head);

  if (sh->version < 6)
    goto error;

  size = sh->meta_length;

  if (private_shm->private_meta_copy == NULL || size > private_shm->meta_len) {
    if (size > private_shm->meta_len) {
      if (private_shm->private_meta_copy)
	free(private_shm->private_meta_copy);
      private_shm->private_meta_copy = NULL;
      private_shm->meta_len = 0;
    }

    if ((buffer = (void *) malloc(size ? size : 1)) == NULL)
      goto error;
    private_shm->private_meta_copy = buffer;
    private_shm->meta_len = size;
    ((char*)buffer)[0] = '\0';
  } else
     buffer = private_shm->private_meta_copy;

  memcpy(buffer, (char *) (private_shm->shm) + sh->meta_start, size);
  *length = size;

 error:
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return buffer;
}

int SPS_PutInfoString(char *fullname, char *array, char *info) {
  SPS_ARRAY private_shm;
  int was_attached, ret = 0;
  struct shm_head *sh;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return(-1);

  if (info == NULL)
    return(-1);

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 1))
    return(-1);

  sh = &(private_shm->shm->head.head);

  if (sh->version < 6) {
    ret = -1;
    goto error;
  }
  strncpy(sh->info, info, sizeof(sh->info));

 error:
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return(ret);
}

char *SPS_GetInfoString(char *fullname, char *array) {
  void *buffer = NULL;
  SPS_ARRAY private_shm;
  int was_attached;
  struct shm_head *sh;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return NULL;

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 0))
    return NULL;

  sh = &(private_shm->shm->head.head);

  if (sh->version < 6)
    goto error;

  if (private_shm->private_info_copy == NULL) {
    if ((buffer = (void *) malloc(sizeof(sh->info))) == NULL)
      goto error;
    private_shm->private_info_copy = buffer;
  }

  memcpy(private_shm->private_info_copy, sh->info, sizeof(sh->info));
  buffer = private_shm->private_info_copy;

 error:
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return buffer;
}

/*
  Used to read a shared string array in where every line is in the
  format identifier=value
    Input: spec_version : The spec version
           array_name : the shared memory string array which holds the
                        lines id=value
	   identifier : the identifier
    Result: NULL if we could not connect to the array or if the identifier
                 did not exist
            value : Do not free this pointer. Do not just keep this pointer
                    around but make a copy of the contents as the storage
                    space is reused at every call to this function.
*/

char *
SPS_GetEnvStr(char *spec_version, char *array_name, char *identifier) {
  int rows, cols;
  char *data;
  char id[SHM_MAX_STR_LEN + 1];
  static char value[SHM_MAX_STR_LEN + 1];
  SPS_ARRAY private_shm;
  char *res = NULL;
  int was_attached, i;
  char strange[SHM_MAX_STR_LEN + 1];

  if ((private_shm = convert_to_handle(spec_version, array_name)) == NULL)
    return NULL;

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 0))
    return NULL; /* Can not attach */

  if (private_shm->shm->head.head.type != SHM_STRING)
    goto back; /* Must be string type */

  if (private_shm->shm->head.head.version < 4)
    data = (char *) &(((struct shm_oheader *)(private_shm->shm))->data);
  else
    data = (char *) &(private_shm->shm->data);
  rows = private_shm->shm->head.head.rows;
  cols = private_shm->shm->head.head.cols;

  if (cols > SHM_MAX_STR_LEN)
    goto back; /* We better give up, our buffer might be too small */

  for (i =0; i < rows; i++) {
    strcpy(strange, data + cols *i); /* sscanf core-dumps if in shared mem*/
    if (sscanf(strange,"%[^=]=%[^\n]", id, value) == 2) {
      /* OK we have a pair */
      if (strcmp(id, identifier) == 0) {
	res = value;
	break;
      }
    }
  }
 back:
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return res;
}

/*
  Used to read all the keys from the enviroment string

    Input: spec_version : The spec version
           array_name : the shared memory string array which holds the
                        lines id=value
    Result: NULL if we could not connect to the array or if the identifier
                 did not exist
            value : Do not free this pointer. Do not just keep this pointer
                    around but make a copy of the contents as the storage
                    space is reused at every call to this function.
*/
char *
SPS_GetNextEnvKey(char *spec_version, char *array_name, int flag) {
  int rows, cols;
  char *data;
  char id[SHM_MAX_STR_LEN + 1];
  SPS_ARRAY private_shm;
  char *res = NULL;
  int was_attached, i, parsed;
  char strange[SHM_MAX_STR_LEN + 1];
  static int loop_count = 0;
  static int keyNO = 0;
  static char **keys=NULL;

  if (flag != 0) {
    if (loop_count >= keyNO) {
      loop_count = 0;
      if (keys != NULL) {
	for (i = 0; i < keyNO; i++)
	  free(keys[i]);
	free(keys);
	keys = NULL;
      }
      return NULL;
    } else {
      res=keys[loop_count];
      loop_count++;
      return res;
    }
  }

  if (keys != NULL) {
    for (i = 0; i < keyNO; i++)
      free(keys[i]);
    free(keys);
    keys = NULL;
  }

  loop_count = 0;
  keyNO = 0;

  if ((private_shm = convert_to_handle(spec_version, array_name)) == NULL)
    return NULL;

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 0))
    return NULL; /* Can not attach */

  if (private_shm->shm->head.head.type != SHM_STRING)
    goto back; /* Must be string type */

  if (private_shm->shm->head.head.version < 4)
    data = (char *) &(((struct shm_oheader *)(private_shm->shm))->data);
  else
    data = (char *) &(private_shm->shm->data);
  rows = private_shm->shm->head.head.rows;
  cols = private_shm->shm->head.head.cols;

  if (cols > SHM_MAX_STR_LEN)
    goto back; /* We better give up, our buffer might be too small */

  keys = malloc(sizeof(char *) * rows);
  for (i = 0; i < rows; i++) {
    char dummy[2];
    strcpy(strange, data + cols *i); /* sscanf core-dumps if in shared mem*/
    parsed = sscanf(strange,"%[^=]=%1[^\n]", id, dummy);
    if (parsed == 2) {
      /* OK we have a pair */
      keys[i] = (char*) strdup(id);
      keyNO++;
    } else if (parsed == 1) {
      keys[i] = (char*) strdup(id);     /* empty string */
    }
  }
 back:
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  if (keyNO) {
    loop_count = 1;
    return keys[0];
  } else {
    free(keys);
    keys = NULL;
    return NULL;
  }
}

/*
  Used to write a shared string array in where every line is in the
  format identifier=value
    Input: spec_version : The spec version
           array_name : the shared memory string array which holds the
                        lines id=value
	   identifier : the identifier
	   set_value : The value you would like to put in the identifier
    Result: 1 error
            0 success

*/

int
SPS_PutEnvStr(char *spec_version, char *array_name,
	       char *identifier, char *set_value) {
  int rows, cols;
  char *data;
  SPS_ARRAY private_shm;
  int res = 1;
  int was_attached, i, use_row;
  char id[SHM_MAX_STR_LEN + 1], value[SHM_MAX_STR_LEN + 1],
    strange[SHM_MAX_STR_LEN + 1];

  if ((private_shm = convert_to_handle(spec_version, array_name)) == NULL)
    return 1;

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 1))
    return 1;

  if (private_shm->shm->head.head.type != SHM_STRING)
    goto back; /* Must be string type */

  if (!private_shm->write_flag)
    goto back; /* We can not write to this array */

  if (private_shm->shm->head.head.version < 4)
    data = (char *) &(((struct shm_oheader *)(private_shm->shm))->data);
  else
    data = (char *) &(private_shm->shm->data);
  rows = private_shm->shm->head.head.rows;
  cols = private_shm->shm->head.head.cols;

  if ((int)(strlen(identifier) + strlen (value) + 2) > cols ||
      cols > SHM_MAX_STR_LEN)
    goto back; /* We will no be able to fit that in */

  for (i =0, use_row = -1; i < rows; i++) {
    strcpy(strange, data + cols *i); /* sscanf core-dumps if in shared mem*/
    if (sscanf(strange,"%[^=]=%[^\n]", id, value) == 2) {
      /* OK we have a pair */
      if (strcmp(id, identifier) == 0) {
	use_row = i; /* We can reuse this row */
	break;
      }
    } else {
      use_row = i; /* Delete the entry which doesn't have the correct format */
      break;
    }
  }

  if (use_row == -1)
    goto back; /* Sorry no more space in the array */

  strcpy(data + cols * use_row, identifier);
  strcat(data + cols * use_row, "=");
  strcat(data + cols * use_row, set_value);

  private_shm->shm->head.head.utime++; /* Updated */
  res = 0; /* Success - tell everybody*/

 back:
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return res;
}

/*
   Tells you if the data in the shared memory array changed since you last
   called this function with these parameters or
   if this is your first call to this function.
   If you are not currently attached to the memory the function will
   attach, read and detach after.

   Input: fullname and array
   Returns: 0 No change
           -1 Error - memory not longer updated
            1 Contents changed - new valid data
*/
int SPS_IsUpdated(char *fullname, char *array) {
  u32_t utime;
  int updated;
  int was_attached;
  u32_t id;
  SPS_ARRAY private_shm;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return -1;

  id = private_shm->id;
  utime = private_shm->utime;
  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 0))
    return -1;

  private_shm->utime = private_shm->shm->head.head.utime;

  updated = (private_shm->id == id)? 0:1;
  if (!updated)
    updated = (private_shm->shm->head.head.utime == utime)? 0:1;
  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return updated;
}

int SPS_LatestFrame(char *fullname, char *array) {
#if SHM_VERSION >= 5
	int     frame;
	int     was_attached;
	SPS_ARRAY private_shm;

	if ((private_shm = convert_to_handle(fullname, array)) == NULL)
		return(-1);

	was_attached = private_shm->attached;

	if (ReconnectToArray(private_shm, 0))
		return(-1);

	frame = private_shm->shm->head.head.latest_frame;

	if (was_attached == 0 && private_shm->stay_attached == 0)
		DeconnectArray(private_shm);

	return(frame);
#else
	return(0);
#endif
}

/*
   Returns the current update counter for this array or -1
   if not longer updated. This function is used in some cases where the
   simpler function SPS_IsUpdated is not sufficient (For example if you
   have multiple subroutines in your program which access the shared memory
   independently. The SPS routines can not distinguish between calls from
   the same subroutine twice or single calls from different subroutines)
   You should keep the old counter value and compare the returned value to
   the old. If it is changed you have to update your array.
   Input: fullname and array
   Returns: update counter value
           -1 Error - memory not longer updated
*/

int SPS_UpdateCounter(char *fullname, char *array) {
  int updated;
  int was_attached;
  SPS_ARRAY private_shm;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return -1;

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 0))
    return -1;

  private_shm->utime = private_shm->shm->head.head.utime;

  updated = private_shm->utime;

  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return updated;
}

/*
   Tells all the readers of the SPEC array that you updated its contents
   If you are not currently attached to the memory the function will
   attach, write and detach after.
   Input: fullname : Spec version
          array : Array in SPEC version
   Returns: 1 Error - memory not longer updated or no write permission
            0 success
*/
int SPS_UpdateDone(char *fullname, char *array) {
  int was_attached;
  SPS_ARRAY private_shm;

  if ((private_shm = convert_to_handle(fullname, array)) == NULL)
    return 1;
  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 1))
    return 1;

  if (private_shm->write_flag == 0)
    return 1;

  private_shm->utime = ++(private_shm->shm->head.head.utime);

  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return 0;
}

/*
  Input: version : name of SPEC version.
         array_name : Name of this spec array
         rows : Pointer to integer to return number of rows in this array
         cols : Pointer to integer to return number of columns in this array
         type : Pointer - Which data type does the data in this array have.
	        Possible values are:
         flag : Pointer More information about the contents of this array
	      (does it contain data for MCA, CCD cameras other info ..)
	      Returns: Error code : 0 == no error
*/

int
SPS_GetArrayInfo(char *spec_version, char *array_name, int *rows,
		  int *cols, int *type, int *flag) {
  int was_attached;
  SPS_ARRAY private_shm;

  if ((private_shm = convert_to_handle(spec_version, array_name)) == NULL)
    return 1;

  was_attached = private_shm->attached;

  if (ReconnectToArray(private_shm, 0)) {
    if (rows) *rows = 0;
    if (cols) *cols = 0;
    if (type) *type = 0;
    if (flag) *flag = 0;
    return 1;
  }

  if (rows) *rows = private_shm->shm->head.head.rows;
  if (cols) *cols = private_shm->shm->head.head.cols;
  if (type) *type = private_shm->shm->head.head.type;
  if (flag) *flag = private_shm->shm->head.head.flags;

  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return 0;
}

/*
  Input: version : name of SPEC version.
	 array_name : Name of this spec array
  Returns: shared memory identifier
*/

int
SPS_GetShmId(char *spec_version, char *array_name) {
  int was_attached;
  int shmid;
  SPS_ARRAY private_shm;

  if ((private_shm = convert_to_handle(spec_version, array_name)) == NULL)
    return -1;

  was_attached = private_shm->attached;

  shmid = (int) private_shm->id;

  if (was_attached == 0 && private_shm->stay_attached == 0)
    DeconnectArray(private_shm);

  return shmid;
}

int
SPS_GetFrameSize(char *spec_version, char *array_name) {
#if SHM_VERSION >= 5
	int     frame_size;
	int     was_attached;
	SPS_ARRAY private_shm;

	if ((private_shm = convert_to_handle(spec_version, array_name)) == NULL)
		return(-1);

	was_attached = private_shm->attached;

	if (ReconnectToArray(private_shm, 0))
		return(-1);

	frame_size = private_shm->shm->head.head.frame_size;

	if (was_attached == 0 && private_shm->stay_attached == 0)
		DeconnectArray(private_shm);

	return(frame_size);
#else
	return(0);
#endif
}

static struct shm_created *
ll_addnew_array(char *specversion, char *arrayname, int isstatus,
		 struct shm_created *status, s32_t id, int my_creation, SHM *shm) {
  struct shm_created *created, **new_created;
  struct shm_created *new_array;

  for (created = SHM_CREATED_HEAD, new_created = & SHM_CREATED_HEAD;
       created; new_created = &(created->next), created = created->next) {
  }

  new_array = (struct shm_created *) malloc(sizeof(struct shm_created));
  if (new_array == NULL)
    return NULL;

  new_array->next = NULL;
  new_array->no_referenced = 0;
  new_array->isstatus = isstatus;
  new_array->status_shm = status;
  new_array->id = id;
  new_array->my_creation = my_creation;
  new_array->handle = NULL;
  new_array->shm = shm;

  if (specversion) {
    if ((new_array->spec_version = (char *) strdup(specversion)) == NULL) {
      free(new_array);
      return NULL;
    }
  } else
    new_array->spec_version = NULL;

  if (arrayname) {
    if ((new_array->array_name = (char *) strdup(arrayname)) == NULL) {
      if (new_array->spec_version)
	free(new_array->spec_version);
      free(new_array);
      return NULL;
    }
  } else
    new_array->array_name = NULL;

  *new_created = new_array;
  return new_array;
}

static struct shm_created *ll_find_pointer(SHM *shm) {
  struct shm_created *created;

  for (created = SHM_CREATED_HEAD; created; created = created->next) {
    if (created->handle && created->handle->shm == shm)
      return created;
  }
  return NULL;
}

static struct shm_created *
ll_find_array(char *specversion, char *arrayname, int isstatus) {
  struct shm_created *created;

  for (created = SHM_CREATED_HEAD; created; created = created->next) {
    if ((specversion == NULL || created->spec_version == NULL ||
	   strcmp(created->spec_version, specversion) == 0) &&
	 (arrayname == NULL || created->array_name == NULL ||
	   strcmp(created->array_name, arrayname) == 0) &&
	 created->isstatus == isstatus) {
      return created;
    }
  }
  return NULL;
}

static void *
id_is_our_creation(u32_t id) {
  struct shm_created *created;

  for (created = SHM_CREATED_HEAD; created; created = created->next)
    if (created->id == (int) id)
      return (created->my_creation)? created->shm:NULL;

  return NULL;
}

static void *
shm_is_our_creation(void *shm) {
  struct shm_created *created;

  for (created = SHM_CREATED_HEAD; created; created = created->next)
    if ((void *) created->shm == shm)
      return (created->my_creation)? created->shm:NULL;

  return NULL;
}

static void
ll_delete_array(struct shm_created *todel) {
  struct shm_created *created, **new_created;

  for (created = SHM_CREATED_HEAD, new_created = & SHM_CREATED_HEAD;
       created; new_created = &(created->next), created = created->next) {
    if (created == todel) {
      *new_created = created->next;
      if (created->spec_version)
	free(created->spec_version);
      if (created->array_name)
	free(created->array_name);
      free(created);
      return;
    }
  }
}

static SHM *
create_shm(char *specversion, char *array, int rows, int cols, int type, int flags) {
  int sflag = 0644, key = -1;
  s32_t id;
  size_t size;
  SHM *shm;

  if (key == -1)
    key = IPC_PRIVATE;
  else
    sflag |= IPC_CREAT;

  size = rows * cols * typedsize(type) + sizeof(SHM) + SHM_META_SIZE;
  id = shmget((key_t) key, size, sflag);
  /*
   * now put the structure into the shared memory
   * Remember to attach to it first
   */

  if ((shm = (SHM *) shmat(id, (char*) 0, 0)) == (SHM *) -1) {
    return NULL;
  }

#if defined(__linux__)
  /* Will be removed after last detach */
  shmctl(id, IPC_RMID, NULL);
#endif

  /* init header */
  shm->head.head.magic = SHM_MAGIC;
  shm->head.head.type = type;
  shm->head.head.version = SHM_VERSION;
  shm->head.head.rows = rows;
  shm->head.head.cols = cols;
  shm->head.head.utime = 0;
  shm->head.head.shmid = id;
  shm->head.head.flags = flags;
  shm->head.head.pid = getpid();
  strcpy(shm->head.head.name, array);
  strcpy(shm->head.head.spec_version, specversion);
  shm->head.head.meta_start = size - SHM_META_SIZE;
  shm->head.head.meta_length = SHM_META_SIZE;
  return shm;
}

static SHM *
create_master_shm(char *name) {
  int sflag = 0644, key = -1;
  s32_t id;
  int size, i;
  SHM *shm;
  struct shm_status *st;

  if (key == -1)
    key = IPC_PRIVATE;
  else
    sflag |= IPC_CREAT;

  size = sizeof(struct shm_status) + sizeof(SHM);
  id = shmget((key_t) key, (size_t) size, sflag);

  if ((shm = (SHM *) shmat(id, NULL, 0)) == (SHM *) -1) {
    return NULL;
  }

#if defined(__linux__)
  /* Will be removed after last detach */
  shmctl(id, IPC_RMID, NULL);
#endif

  /* init header */
  shm->head.head.magic = SHM_MAGIC;
  shm->head.head.type = 0;
  shm->head.head.version = SHM_VERSION;
  shm->head.head.rows = 0;
  shm->head.head.cols = 0;
  shm->head.head.utime = 0;
  shm->head.head.shmid = id;
  shm->head.head.flags = SHM_IS_STATUS;
  shm->head.head.pid = getpid();
  *(shm->head.head.name) = '\0';
  strcpy(shm->head.head.spec_version, name);

  if (shm->head.head.version < 4)
    st = (struct shm_status *) &(((struct shm_oheader *)shm)->data);
  else
    st = (struct shm_status *) &(shm->data);
  st->spec_state = 0;
  st->utime = 0;                 /* updated when ids[] changes */
  for (i = 0; i < SHM_MAX_IDS; i++)
    st->ids[i] = -1;                 /* shm ids for shared arrays */

  return shm;
}

static void delete_shm(s32_t id) {
  shmctl(id, IPC_RMID, NULL);
}

static void delete_id_from_status(SHM *status, s32_t id) {
  struct shm_status *st;
  int i, j;
  if (status->head.head.version < 4)
    st = (struct shm_status *) &(((struct shm_oheader *)status)->data);
  else
    st = (struct shm_status *) &(status->data);
  for (i = 0; i < SHM_MAX_IDS; i++)
    if (st->ids[i] == id) {
      for (j = i; j < SHM_MAX_IDS -1; j++)
	st->ids[j] = st->ids[j+1];
      break;
    }
  st->utime++;
}

/*
   Creates a shared memory array and the shared memory structure for
   the spec version. You can only create new arrays for specversions
   which either do not exist or have been created by this process.
   After the call you are automatically attached to the shared memory.
   The process always stay attached to the shared memories he created.
   You can call all the other functions like Connect or Deconnect but you will
   always stay attached to the shared memory
   Input: spec_version: The specversion you will create the new array in
          arrayname: The name of the array
	  rows: number of rows
	  cols: number of columns
	  type : The data type of the data stored in this array.
	  flag: A flag to indicate the type of the array MCA data or
	        CCD camera. It is not necessary to specify SPS_IS_ARRAY here.
		This flag is always true for the arrays we can create with
		this function.
   Returns: 0 OK
            1 Error
*/

int
SPS_CreateArray(char *spec_version, char *arrayname, int rows, int cols, int type, int flags) {
  SHM *shm, *ashm;
  int i, idx;
  struct shm_status *st;
  struct shm_created *shm_array, *shm_status;
  SPS_ARRAY private_shm;

  flags |= SHM_IS_ARRAY; /* We can only create arrays with this function */

  if (spec_version == NULL || arrayname == NULL)
    return 1;

  if ((shm_status = ll_find_array(spec_version, NULL, 1)) == NULL) {

    /* SearchSpecArrays (spec_version); */
    /* Check if there is already a spec_version which exists */
    if ((idx = find_TabIDX(spec_version, 0)) != -1)
      return 1; /* We can not add to a spec_version we not created*/

    /* Create a spec_version */
    if ((shm = create_master_shm(spec_version)) == NULL)
      return 1;
    if ((shm_status = ll_addnew_array(spec_version, NULL, 1, NULL,
	     shm->head.head.shmid, 1, shm))   == NULL) {
      c_shmdt((void *)shm);
      return 1;
    }
    private_shm = add_private_shm(shm, spec_version, NULL, 1);
    shm_status->handle = private_shm;


  } else {
    if (shm_status->shm == NULL) {
      if ((shm = (SHM *) shmat(shm_status->id, NULL, 0)) == (SHM *) -1) {
	return 1;
      }
      shm_status->shm = shm;
    } else
      shm = shm_status->shm;
  }

  /* There is already an array with this name - delete it */
  if ((shm_array = ll_find_array(spec_version, arrayname, 0)) != NULL) {
    if (shm_array->shm != NULL)
      shmdt((void *)shm_array->shm);
    delete_id_from_status(shm_array->status_shm->shm, shm_array->id);
    delete_shm(shm_array->id);
    ll_delete_array(shm_array);
  }

  /* Create the new array */
  ashm = create_shm(spec_version, arrayname, rows, cols, type, flags);
  if (ashm == NULL)
    return 1;
  if ((shm_array = ll_addnew_array(spec_version, arrayname, 0, shm_status,
		 ashm->head.head.shmid, 1, ashm)) == NULL) {
    shmdt((void*)ashm);
    return 1;
  }

  /* Add reference to this array to the STATUS array */
  st = (struct shm_status *) &(shm->data);

  for (i = 0; i < SHM_MAX_IDS; i++)
    if (st->ids[i] == -1)
      break;
  st->ids[i] = ashm->head.head.shmid;
  st->utime++;

  private_shm = add_private_shm(ashm, spec_version, arrayname, 1);
  shm_array->handle = private_shm;


  return 0;
}

static int delete_handle(SPS_ARRAY handle) {
  if (handle == NULL)
    return 1;
  if (handle->buffer_len && handle->private_data_copy)
    free (handle->private_data_copy);
  if (handle->private_info_copy)
    free (handle->private_info_copy);
  if (handle->private_meta_copy)
    free (handle->private_meta_copy);
  if (handle->spec)
    free(handle->spec);
  if (handle->array)
    free(handle->array);
  free(handle);
  return 0;
}

/* Deletes everything which there is */
/* Should be called before you quit the program */

void SPS_CleanUpAll(void)
{
  struct shm_created *created, *created_next;
  for (created = SHM_CREATED_HEAD; created;) {
    if (created->handle && created->handle->attached && created->handle->shm)
      shmdt ((void *)created->handle->shm);

    if (created->my_creation) {
      delete_shm(created->id);
    }

    if (created->handle)
      delete_handle (created->handle);

    if (created->array_name)
      free(created->array_name);

    if (created->spec_version)
      free(created->spec_version);

    created_next = created->next;
    free(created);

    created = created_next;
  }

  SHM_CREATED_HEAD = NULL;
  id_no = 0;

  delete_SpecIDTab();
}


#if SPS_DEBUG

#include <sys/time.h>

static void bench_mark();

static void PrintIDDir()
{
  int i, j;
  char *spec_version, *array;
  int rows,cols,type,flags;
  int state;
  double *double_buf=NULL;
  int k_row, k_col;

  for (i=0; (spec_version = SPS_GetNextSpec(i)); i++) {
    state = SPS_GetSpecState(spec_version);
    printf("%s state = 0x%x\n",spec_version, state);
    for (j=0; (array = SPS_GetNextArray(spec_version, j)); j++) {
      SPS_GetArrayInfo(spec_version, array, &rows, &cols, &type, &flags);
      printf("    %s: %dx%d type:%d flags:0x%x \n",
	      array, rows, cols, type, flags);
      /* Reading data */
      double_buf = SPS_GetDataCopy(spec_version, array, SHM_DOUBLE,
				    &rows, &cols);

      for (k_row = 0; k_row < rows && k_row < 10; k_row++) {
	for (k_col = 0; k_col < cols && k_col < 7; k_col++)
	  printf("%10.10g ",*(double_buf + k_row * cols + k_col));
	printf("\n");
      }
      SPS_FreeDataCopy(spec_version,array);
    }
  }
}

int
main()
{
  char buf[256];
  char buf1[256],buf2[256], buf3[256], buf4[256];
  char *str;
  int rows,cols,type, flag, flags;
  int i, j;
  s32_t *ptr;
  double *double_buf=NULL;
  int k_row, k_col;
  char *spec_version, *array;
  int state;

  printf("(C)reate,(*)(d)ir,(p)oll,(r)st,(e)nv,(w)rite env,(b)ench,(q)uit\n");

  while (scanf("%s",buf) == 1) {
    switch (buf[0]) {
    case 'c':
      printf("spec_version array_name rows cols type flags\n");
      scanf("%s %s %d %d %d %d",buf1,buf2,&rows,&cols, &type, &flag);
      SPS_CreateArray(buf1,buf2, rows, cols, type, flag);
      SPS_GetArrayInfo(buf1, buf2, &rows, &cols, &type, &flag);
      ptr = SPS_GetDataPointer(buf1, buf2, 1);
      if (type == SHM_LONG) {
	for (i = 0; i < rows; i++)
	  for (j = 0; j < cols; j++)
	    *(ptr + i * cols + j) = i * 100 + j;
      }
      break;
    case 'd':
      PrintIDDir();
      break;
    case '*':
      for (j=0; (array = SPS_GetNextArray(NULL, j)); j++) {
	SPS_GetArrayInfo(NULL, array, &rows, &cols, &type, &flags);
	printf("    %s: %dx%d type:%d flags:0x%x \n",
		array, rows, cols, type, flags);
	/* Reading data */
	double_buf = SPS_GetDataCopy(NULL, array, SHM_DOUBLE,
				      &rows, &cols);

	for (k_row = 0; k_row < rows && k_row < 10; k_row++) {
	  for (k_col = 0; k_col < cols && k_col < 7; k_col++)
	    printf ("%10.10g ",*(double_buf + k_row * cols + k_col));
	  printf("\n");
	}
	SPS_FreeDataCopy(NULL, array);
      }
      break;
    case 'q':
      exit(0);
    case 'b':
      bench_mark();
      break;
    case 'r':
      SPS_CleanUpAll();
      break;
    case 'e':
      printf("spec_version array_name identifier\n");
      scanf("%s %s %s",buf1,buf2,buf3);
      if (buf1[0] == '*')
	spec_version = NULL;
      else
	spec_version = buf1;
      str = SPS_GetEnvStr(spec_version, buf2, buf3);
      printf("%s=<%s>\n",buf3,str ? str : "not found");
      break;
    case 'w':
      printf("spec_version array_name identifier value\n");
      scanf("%s %s %s %s",buf1,buf2,buf3,buf4);
      if (buf1[0] == '*')
	spec_version = NULL;
      else
	spec_version = buf1;
      printf("Result %d\n",SPS_PutEnvStr(spec_version, buf2, buf3, buf4));
      break;
    case 'p':
      printf("Waiting for specversion hop array test\n");
      printf("spec_version array_name\n");
      scanf("%s %s",buf1,buf2);
      if (buf1[0] == '*')
	spec_version = NULL;
      else
	spec_version = buf1;
      SPS_AttachToArray(spec_version, buf2, 0); /* What I want to say here is
					    that I would like to stay
					    attached */
      printf("Waiting for %s:%s\n",(spec_version == NULL)?"NULL":spec_version,
	     buf2);
      for (;;) {
	if (SPS_IsUpdated(spec_version,buf2) == 1) {
	  printf("Changed:\n");
	  double_buf = SPS_GetDataCopy(spec_version, buf2, SHM_DOUBLE,
					&rows, &cols);
	  for (k_row = 0; k_row < rows && k_row < 10; k_row++) {
	    for (k_col = 0; k_col < cols && k_col < 7; k_col++)
	      printf("%10.10g ",*(double_buf + k_row * cols + k_col));
	    printf("\n");
	  }
	}
	sleep(1);
      }
      break;
    }

  }
}

/* Benchmark our routines */
static void bench(char *str) {
static struct timeval start, stop;
  struct timezone tzp;

  if (str == (char *)NULL) {
    gettimeofday(&start, &tzp);
  }
  else {
    gettimeofday(&stop, &tzp);
    printf("Time in %s : %10.3f\n", str,
	   (double)(stop.tv_sec-start.tv_sec) +
	   (double)(stop.tv_usec-start.tv_usec) * (double)0.000001);
    start.tv_sec = stop.tv_sec;
    start.tv_usec = stop.tv_usec;
  }
}


static void bench_mark()
{
  int i;

  printf("\n\
     Version spec must run with\n\
     shared array doubletest[1024][1024]\n\
     shared long array longtest[1024][1024]\n\
     shared string array text[200][200]\n\
     array_op(\"fill\",doubletest,10000)\n\
     array_op(\"fill\",longtest,10000)\n\
     text[0][]=\"id=myvalue\"\n\
  \n");

  getchar();

  bench(NULL);
  SPS_IsUpdated("not","found");
  bench("SPS_IsUpdated non existing specversion not");
  SPS_IsUpdated("not","found");
  bench("SPS_IsUpdated non existing specversion not");
  SPS_IsUpdated("spec","not");
  bench("SPS_IsUpdated version spec, array not");

  printf("\n");
  SPS_GetDataCopy("spec", "doubletest", SHM_DOUBLE, NULL, NULL);
  bench("SPS_GetDataCopy version spec, array doubletest as double");
  SPS_GetDataCopy("spec", "longtest", SHM_DOUBLE, NULL, NULL);
  bench("SPS_GetDataCopy version spec, array longtest as double");

  printf("\n");
  for (i=0;i<1000;i++)
    SPS_IsUpdated("spec","doubletest");
  bench("SPS_IsUpdated version spec, array doubletest, 1000 times");
  for (i=0;i<1000;i++)
    SPS_GetSpecState("spec");
  bench("SPS_GetSpecState version spec, 1000 times");
  for (i=0;i<1000;i++)
    SPS_GetEnvStr("spec","text","id");
  bench("SPS_GetEnvStr version spec, array text, id, 1000 times");

  printf("\n");
  SPS_AttachToArray("spec","doubletest",0);
  SPS_AttachToArray("spec","longtest",0);
  SPS_AttachToArray("spec","text",0);
  bench("Attached and Staying attached now");

  for (i=0;i<1000;i++)
    SPS_IsUpdated("spec","doubletest");
  bench("SPS_IsUpdated version spec, array doubletest, 1000 times");
  for (i=0;i<1000;i++)
    SPS_GetSpecState("spec");
  bench("SPS_GetSpecState version spec, 1000 times");
  for (i=0;i<1000;i++)
    SPS_GetEnvStr("spec","text","id");
  bench("SPS_GetEnvStr version spec, array text, id, 1000 times");

  printf("\n");
  SPS_DetachFromArray("spec","doubletest");
  SPS_DetachFromArray("spec","longtest");
  SPS_DetachFromArray("spec","text");
  bench("Detach and Staying detached now");

  for (i=0;i<1000;i++)
    SPS_IsUpdated("spec","doubletest");
  bench("SPS_IsUpdated version spec, array doubletest, 1000 times");
  for (i=0;i<1000;i++)
    SPS_GetSpecState("spec");
  bench("SPS_GetSpecState version spec, 1000 times");
  for (i=0;i<1000;i++)
    SPS_GetEnvStr("spec","text","id");
  bench("SPS_GetEnvStr version spec, array text, id, 1000 times");

  printf("\nNow testing the NULL version tests\n");
  for (i=0;i<1000;i++)
    SPS_GetEnvStr(NULL, "text","id");
  bench("SPS_GetEnvStr NULL, array text, id, 1000 times");
  for (i=0;i<1000;i++)
    SPS_IsUpdated(NULL, "doubletest");
  bench("SPS_IsUpdated NULL, array doubletest, 1000 times");

  printf("\nNow mix it\n");
  for (i=0;i<1000;i++) {
    SPS_IsUpdated(NULL, "doubletest");
    SPS_IsUpdated("spec", "doubletest");
  }
  bench("SPS_IsUpdated NULL mixed, array doubletest, 1000 times");

  printf("\n");
  SPS_CreateArray("version","array",100,100,0,0);
  bench("SPS_CreateArray");
  SPS_CleanUpAll();
  bench("SPS_CleanUpAll");
}
#endif
