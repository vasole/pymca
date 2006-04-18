#/*##########################################################################
# Copyright (C) 2004-2006 European Synchrotron Radiation Facility
#
# This file is part of the PyMCA X-ray Fluorescence Toolkit developed at
# the ESRF by the Beamline Instrumentation Software Support (BLISS) group.
#
# This toolkit is free software; you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option) 
# any later version.
#
# PyMCA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# PyMCA; if not, write to the Free Software Foundation, Inc., 59 Temple Place,
# Suite 330, Boston, MA 02111-1307, USA.
#
# PyMCA follows the dual licensing model of Trolltech's Qt and Riverbank's PyQt
# and cannot be used as a free plugin for a non-free program. 
#
# Please contact the ESRF industrial unit (industry@esrf.fr) if this license 
# is a problem to you.
#############################################################################*/
static char RcsId[] = "$Header: /segfs/bliss/source/python/specfile/specfile-3.1/src/RCS/specex.c,v 1.1 2003/09/12 13:23:19 rey Exp $";
/***********************************************************************
 * 
 *    File:          extract.c
 *
 *    Description:   program to extract scan data from a SPEC file
 *
 *    Author:        Vicente Rey Bakaikoa
 *
 ************************************************************************/
/*
 *   $Log: specex.c,v $
 *   Revision 1.1  2003/09/12 13:23:19  rey
 *   Initial revision
 *
 * Revision 2.0  2000/04/13  13:28:39  13:28:39  rey (Vicente Rey-Bakaikoa)
 * New version of the library. Complete rewrite
 * Adds support for MCA
 * 
 * Revision 1.5  2000/02/16  13:58:02  13:58:02  rey (Vicente Rey-Bakaikoa)
 * Version before major changes for MCA support
 * 
 * Revision 1.1  98/04/28  17:57:44  17:57:44  rey (Vicente Rey-Bakaikoa)
 * Initial revision
 * 
 * Revision 1.2  97/06/05  17:29:46  17:29:46  rey (Vicente Rey-Bakaikoa)
 * Small correction
 * 
 * Revision 1.1  97/06/05  15:51:04  15:51:04  rey (Vicente Rey-Bakaikoa)
 * Initial revision
 * 
 */

/*
 * Include files
 */
#include <SpecFile.h>

/*
 * Define 
 */ 
#define MAX_LEN_NAME   50

/*
 * Function declarations
 */
static short  getargs(int argc,char **argv);
static long   getNumFromList(char *list,long **nos,long max);
static long   getIxNumFromList(char *list,long **nos,long max);
static void   printUsage(char *name);
static void   promptOptions(char *name);
static char  *compList(long *list,long size);


/*
 * Externals
 */
extern char *optarg;
extern int   optind, optopt,opterr;

/*
 * Globals
 */
static char  *collist;
static char  *scanlist;
static char  *outfmt;
static char  *Lpar;
static char  *prefix;
static char  *filename;

SpecFile *sf;
FILE     *fd;
char     *outfile;

/*
 * Option flags.
 */
static short cflag=0,
             dflag=0,
             fflag=0,
             hflag=0,
             Hflag=0,
             iflag=0,
             lflag=0,
             ldflag=0,
             Lflag=0,
             oflag=0,
             Oflag=0,
             sflag=0,
             Sflag=0,
             tflag=0;


/*
 *  MAIN
 */
main(int argc, char *argv[])
{
      int      error=0;
      int      i,j;
      int      ret;

      long     scanct;
      long    *scanno;
      long     colct;
      long    *colno;

      long     index;
      long     order;

      double **data;
      long    *info;
      char   **hdrlines;
      char   **labels;
      char    *cmd;

      long    *list;
      long     totlab;
      long     cl;
      long     col;
      long     hdr;
      long     nohdr;
      long     nohdrbefore;
      long     line;
      long     totline;
      long     max_length;
 
     /*
      * Decode command line
      */
      if (argc > 1) {
          if (getargs(argc,argv) != 0) {
               printUsage(argv[0]);
               exit(-1);
          }     
      } else { iflag++; }

     /*
      * Help with usage ( -h )
      */
      if (hflag) {
          printUsage(argv[0]);
          exit(0);
      }

     /* 
      * Interactively ask for options: -i or no parameters
      */
      if (iflag) {
          promptOptions(argv[0]);
      }

     /*
      * Try to open input file
      */
      if ((sf = SfOpen(filename,&error)) == (SpecFile *)NULL) {
            printf("Error opening spec file (%s)\n",SfError(error));
            exit(-1);
      } 

      max_length = strlen(filename) + 4;

      outfile = (char *) malloc( sizeof(char) * (max_length + 7));

      if (!oflag) {
         prefix = (char *) malloc(sizeof(char) * max_length);
         sprintf(prefix,"%s_out",filename);
      }
 
     /*
      * Give a scan list and exits if L option.
      */
      scanct = SfScanNo(sf);

      if (Lflag && scanct > 0) {
            list = SfList(sf,&error); 
            if (Lpar[0] == 'l') {
                for (i=0;i<scanct;i++) {
                    cmd = SfCommand(sf,i+1,&error); 
                    printf("%3d  %s\n",list[i],cmd);
                    free(cmd);
                }
            } else { 
                if (list != (long *) NULL) {
                       printf("%s\n",compList(list,scanct));
                }
            } 
            SfClose(sf);
            exit(0);
      }

     /*
      * Prepare arrays with scan and column list.
      */
      if (sflag) { 
         scanct = getIxNumFromList(scanlist,&scanno,scanct);
         if (scanct) {
            colct  = SfNoColumns(sf,scanno[0],&error);
         } else {
            colct = 0;
         }
      } else {
         colct  = SfNoColumns(sf,1,&error);
      }

      if (cflag && colct) 
         colct  = getNumFromList(collist,&colno,colct);

      if (Sflag) {
           if (Oflag) {
               fd = stdout;
           } else {
               strcpy(outfile,prefix);
               if ((fd= fopen(outfile,"w")) == NULL) {
                   printf("Cannot open file (%s), sorry\n",outfile);
                   exit(-1);
               }
           }
      }

     /*
      * Main loop. Go through scanlist
      */
      for (i=0;i<scanct && colct>0 ;i++) {
            if (sflag) index = scanno[i];
            else       index = i+1;

            if (!cflag) colct = SfNoColumns(sf,index,&error);
            /*
             * Open output file.
             */
            if (!Sflag) {
                 sprintf(outfile,"%s%.4d",prefix,SfNumber(sf,index) );
                 if ((order = SfOrder(sf,index))> 1) {
                    sprintf(outfile,"%s.%d",outfile,order);
                 }
                 if ((fd= fopen(outfile,"w")) == NULL) {
                    printf("Cannot open file (%s), sorry\n",outfile);
                    exit(-1);
                 }
            }
            printf("Scan %d\n",index);
            /*
             * find requested data
             * (options -c ) 
             */
            if (!dflag) {
              if ((ret = SfData(sf,index,&data,&info,&error)) == -1) {
                  fprintf(stderr,"Error extracting data for scan index (%d)\n"
                         ,index);
                  if (!Sflag) fclose(fd);
                  continue;
              }
              totline = info[0];
            }

            /*
             * write to file
             */

            /* HEADERS */
            if (Hflag) {
              if ((nohdr = SfHeader(sf,index,(char *)NULL,&hdrlines,&error)) == -1) {
                fprintf(stderr,"Error extracting headers for scan index (%d)\n"
                        ,index);
                if (!dflag) {
                   freeArrNZ((void ***)&data,totline);
                   freePtr(info);
                }
                if (!Sflag) fclose(fd);
                continue; 
              }
              nohdrbefore = SfNoHeaderBefore(sf,index,&error);

              for (hdr = 0; hdr < nohdrbefore - 1; hdr++) {
                 fprintf(fd,"%s\n",hdrlines[hdr]);
              }
            } 

            /* LABELS */
            if (lflag) {
              if ((totlab = SfAllLabels(sf,index,&labels,&error)) == -1) {
                 fprintf(stderr,"Error extracting labels for scan index (%d)\n"
                        ,index);
                 if (Hflag) freeArrNZ((void ***)&hdrlines,nohdr);
                 if (!dflag) {
                    freeArrNZ((void ***)&data,totline);
                    freePtr(info);
                 }
                 if (!Sflag) fclose(fd);
                 continue; 
              }

              if (ldflag) {
                 fprintf(fd,"#L ");
              }
              for (cl=0;cl<colct;cl++) {
                  if (cflag) col = colno[cl]-1;
                  else       col = cl;
                  fprintf(fd,"%s  ",labels[col]);
              }
              fprintf(fd,"\n");
            } else {
              if (Hflag) 
                  fprintf(fd,"%s\n",hdrlines[hdr]);
            }           

            /* DATA */
            if (!dflag) {
              for (line = 0; line < totline;line++) {
                 for (cl=0;cl<colct;cl++) {  
                     if (cflag) col = colno[cl]-1;
                     else       col = cl;

                     fprintf(fd,outfmt,data[line][col]);
                 }
                 fprintf(fd,"\n");
              }
            }
             
            /* HEADERS AFTER */
            if (Hflag) {
                 for (hdr+=1;hdr<nohdr;hdr++) {
                     fprintf(fd,"%s\n",hdrlines[hdr]);
                 }
            }
            fprintf(fd,"\n");

           /* 
            * Free arrays.
            */
            if (Hflag) freeArrNZ((void ***)&hdrlines,nohdr);
            if (lflag) freeArrNZ((void ***)&labels,totlab);
            if (!dflag) {
                freeArrNZ((void ***)&data,totline);
                freePtr(info);
            }

           /*
            * Close output file.
            */
           if (!Sflag) fclose(fd);

      }

      if (Sflag) fclose(fd);
 
      free(prefix);
      free(outfile);
      free(outfmt);
      if (sflag && scanct) {
         free(scanno);
         free(scanlist);
      }
      if (cflag && colct) {
         free(colno);
         free(collist);
      }


      SfClose(sf);
      exit(0);
}


/*****************************************************************
 *  
 *  Function:     getargs()
 *  
 *  Description:  interprets options from command line
 * 
 *****************************************************************/
static short
getargs(argc,argv)
int   argc;
char *argv[];
{
    int       i,c;
    SpecFile *sf;

    short  errflag = 0;

    opterr = 0;

    while((c = getopt(argc,argv,":c:dDf:ihHlL:o:Os:St")) != -1 )
    {
        switch(c) {
           case 'c':             /* c olumns: arg is a list of columns */
              cflag++;
              collist = (char *) strdup(optarg);
              break;

           case 'd':             /* d ata is not extracted */
              dflag++;
              break;

           case 'D':             /* set hash for column labels */
              ldflag++;
              break;

           case 'f':             /* ouput f ormat: for columns */
              fflag++;
              outfmt = (char *) strdup(optarg);
              break;

           case 'h':             /* h elp: show usage */
              hflag++;
              break;

           case 'H':             /* extract all headers */
              Hflag++;
              break;

           case 'i':             /* i nteractive */
              iflag++;
              break;

           case 'l':             /* extract column l abels */
              lflag++;
              break;

           case 'L':             /* shows a list of scans */
              Lflag++;
              Lpar  = (char *) strdup(optarg);
              break;

           case 'o':             /* prefix for o utput files */
              oflag++;
              prefix = (char *) strdup(optarg);
              break;

           case 'O':             /* use standard O utput */
              Sflag++;
              Oflag++;
              break;

           case 's':             /* s can: arg is a list of scans */
              sflag++;
              scanlist = (char *) strdup(optarg);
              break;

           case 'S':             /* S ingle file */
              Sflag++;
              break;

           case 't':             /* t ab:  separates values with tabs */
              tflag++;
              break;

           case ':':  
              errflag++;
              fprintf(stderr,"Option -%c requires an argument\n",optopt);
              break;

           case '?': /* Unrecognized option */
              errflag++;
              fprintf(stderr,"Unrecognized option\n",optopt);
              break;

        }
    }

    if (!fflag)  {
       outfmt = (char *)malloc(sizeof(char) * 10);
       if (tflag) {
         sprintf(outfmt,"%s\t","%g");
       } else {
         sprintf(outfmt,"%s ","%g");
       }
    }

    if (!iflag) {
       if (argc <= optind ) {   /* where is the file name */
             errflag++;
       } else {
             filename = (char *)strdup(argv[optind]);
       }
    }

    return(errflag);

}


/*****************************************************************
 *
 *   Function:   getIxNumFromList()
 *
 *   Description: returns in array numbers a list of scan indexes
 *                extracted from a string "list" that understands
 *                "," to separate tokens
 *                ":" inside a token to give a range
 *                "." for one scan number to give the occurrence of
 *                    that scan number inside the file
 *                example:  1,2,3,3.2,4:8
 *                means scan indexes for scan numbers 1,2,3,3(2nd),4,5,6,7,8
 *   Parameters:
 *        - list:    string to be interpreted
 *        - numbers: output long array with scan indexes
 *        - max:     scan maximum index
 *
 *******************************************************************/
static long  
getIxNumFromList(list,numbers,max)
char *list;
long **numbers;
long  max;
{
     long *array;
     long counter=0,i;

     char *ptr,*ptr2,*ptr3;
     long findex,lindex;
     long fscan,forder;
     long lscan,lorder;

     array = (long *) malloc( max * sizeof(long));
     ptr = strtok(list,",");

     while(ptr) {

           ptr2 = strchr(ptr,':');

           if (!ptr2) {
              fscan = atol(ptr);
              ptr3  = strchr(ptr,'.');
              if (!ptr3) {
                 forder = 1;
              } else {
                 forder = atol(ptr3+1);
              }
              findex = SfIndex(sf,fscan,forder);
              lindex = findex;
           } else {
              fscan = atol(ptr);
              ptr3  = strchr(ptr,'.');
              if (!ptr3) {
                 forder = 1;
              } else {
                 forder = atol(ptr3+1);
              }
              lscan = atol(ptr2+1);
              ptr3  = strchr(ptr2+1,'.');
              if (!ptr3) {
                 lorder = 1;
              } else {
                 lorder = atol(ptr3+1);
              }
              findex = SfIndex(sf,fscan,forder);
              lindex = SfIndex(sf,lscan,lorder);
           }
 
          /*
           *  Check.
           */ 
          if (findex > lindex || lindex > max || findex < 1) {
             fprintf(stderr,"Wrong scan selection\n");
             free(array);
             return(0);
          }
 
         /*
          * Fill.
          */
          for (i=0;i<lindex - findex +1;i++) {
               array[counter] = findex + i;
               counter++;
          }

          ptr = strtok((char *)NULL,",");
     }

     if (counter) {
       *numbers = (long *)malloc(sizeof(long) * counter);
       memcpy(*numbers,array,sizeof(long)*counter);
     } 

     free(array);
     return(counter);
}


/*****************************************************************
 *
 *   Function:   getNumFromList()
 *
 *   Description: returns in an array of longs
 *                extracted from a string "list" that understands
 *                "," to separate tokens
 *                ":" inside a token to give a range
 *                example:  1,2,3,4:8
 *                means numbers 1,2,3,4,5,6,7,8
 *   Parameters:
 *        - list:    string to be interpreted
 *        - numbers: output long array with scan indexes
 *        - max:     scan maximum index
 *
 *******************************************************************/
static long  
getNumFromList(list,numbers,max)
char *list;
long **numbers;
long   max;
{
     long  *array;
     long   counter=0,i;

     char  *ptr; 
     char  *ptr2;

     long   fcol,lcol;
     long   error=0;

     array = (long *) malloc(max * sizeof(long));
     ptr = strtok(list,",");

     while(ptr) {

           ptr2 = strchr(ptr,':');

           if (!ptr2) {
              fcol = atol(ptr);
              lcol = fcol;
           } else {
              fcol = atol(ptr);
              lcol = atol(ptr2+1);
           }
 
           if (fcol < 0) fcol = max +fcol+1;
           if (lcol < 0) lcol = max +lcol+1;

          /*
           * Check
           */
           if (fcol > lcol || lcol > max || lcol == 0 || fcol == 0) {
               fprintf(stderr,"Wrong column selection\n");
               error++;
               free(array);
               return(0);
           }

          /*
           * Fill.
           */
           for (i=0;i<lcol - fcol +1;i++) {
                array[counter] = fcol + i;
                counter++;
           }

           if (error) { free(array); return(-1); };

           ptr = strtok((char *)NULL,",");
     }

     *numbers = (long *)malloc(sizeof(long) * counter);
    
     memcpy(*numbers,array,sizeof(long)*counter);

     free(array);
     return(counter);
}


/************************************************************
 * 
 *  Function:   printUsage()
 * 
 *  Description: shows usage for the program 
 * 
 *************************************************************/
static void
printUsage(name)
char *name;
{
      printf("Usage: %s [-options...] filename\n",name);      
      printf("\nwhere options include:\n");      

      printf("-c column-list      list of columns to be extracted (def: all)\n");
      printf("-d                  do not output data (useful for extracting header only)\n");
      printf("-f format           output format for columns\n");
      printf("-h                  show this help\n");
      printf("-H                  keep headers as in original file \n");
      printf("-i                  use interactive mode\n");
      printf("-l                  extract column label too\n");
      printf("-D                  add hash before labels\n");
      printf("-L {s/l}            shows a list of scans in file and exits\n");
      printf("                    s=short format, l=long format\n");
      printf("-o prefix           prefix for output files\n");
      printf("-O                  use standard output\n");
      printf("-s scan-list        list of scans to be extracted (def: all\n");
      printf("-S                  output to one Single file (use prefix as name)\n");
      printf("-t                  use tabs instead of spaces\n");
}


/*******************************************************************
 * 
 *  Function:   promptOptions() 
 * 
 *  Description:  interactively asks for all options.
 * 
 ********************************************************************/
static void   
promptOptions(name)
char *name;
{
    char strin[100];

    printf("Hello! This is \"%s\" program\n\n",name);

    printf("Name of the file to extract data from (\"?\" for list) -> ");
    gets(strin);
    if (strin[0] == '?') {
        system("ls");
    printf("               Name of the file to extract data from -> ");
        gets(strin);
    }
    filename = (char *) strdup(strin);
    printf("      Scan list (use \',\' and \':\' to specify a list) -> ");
    gets(strin);
    if (strin[0] != '\0') { sflag++; scanlist = (char *) strdup(strin);}
    printf("    Column list (use \',\' and \':\' to specify a list) -> ");
    gets(strin);
    if (strin[0] != '\0') { cflag++; collist = (char *) strdup(strin);}
    printf("                Extract labels for selected columns -> ");
    gets(strin);
    if (strin[0] == 'y' || strin[0] == 'Y') lflag++;
    printf("                                   Keep all headers -> ");
    gets(strin);
    if (strin[0] == 'y' || strin[0] == 'Y') Hflag++;
    printf(" Suppress data output (only headers will be output) -> ");
    gets(strin);
    if (strin[0] == 'y' || strin[0] == 'Y') dflag++;
    printf("                         Use tabs instead of spaces -> ");
    gets(strin);
    if (strin[0] == 'y' || strin[0] == 'Y') tflag++;
    printf("                          Format for output columns -> ");
    gets(strin);
    if (strin[0] != '\0') { fflag++; outfmt = (char *) strdup(strin);}
    printf("                Use a single output file for output -> ");
    gets(strin);
    if (strin[0] == 'y' || strin[0] == 'Y') Sflag++;
    printf("                            Prefix for output files -> ");
    gets(strin);
    if (strin[0] != '\0') { oflag++; prefix = (char *) strdup(strin);}
    printf("                                Use standard output -> ");
    gets(strin);
    if (strin[0] == 'y' || strin[0] == 'Y') Oflag++;

}



/*********************************************************************
 * 
 *  Function:    compList()
 *
 *  Description: returns a string with a compresed list of an array of
 *               longs
 *
 *********************************************************************/
char *
compList( long *list, long size )
{
     long    first,this,last,      
                i;

     char    buf[30],
               *str;


     if (size < 1) { return((char *)NULL);}

     first    = list[0];
     last     = first;
     
     str      = (char *)malloc( sizeof(char) * 5000 );

     sprintf( buf, "%d",first);
     strcpy ( str, buf ); 
     
     for( i=1 ; i < size ; i++ ) {
          this = list[i];

          if ( this != last + 1 ) {
               if (last != first) {
                   sprintf( buf,":%d",last);
                   strcat( str, buf ); 
               } 
               sprintf( buf,",%d",this);
               strcat( str, buf ); 
               first = this;
          }
          last = this;
     }

     if (last != first) {
        sprintf( buf,":%d",last);
        strcat( str, buf ); 
     } 

     return( str );
}
