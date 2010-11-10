#include <locale_management.h>

#ifdef PYMCA_POSIX
#else
#ifdef SPECFILE_POSIX
#include <locale.h>
#endif
#endif

#include <stdlib.h>
#include <string.h>


double PyMcaAtof(const char * inputString)
{
#ifdef PYMCA_POSIX
	return atof(inputString);
#else 
#ifdef SPECFILE_POSIX
	char *currentLocaleBuffer;
	char *restoredLocaleBuffer;
	char localeBuffer[21];
	double result;
	currentLocaleBuffer = setlocale(LC_NUMERIC, NULL);
	strcpy(localeBuffer, currentLocaleBuffer);
	setlocale(LC_NUMERIC, "C\0");
	result = atof(inputString);
	restoredLocaleBuffer = setlocale(LC_NUMERIC, localeBuffer);
	return(result);
#else
	return atof(inputString);
#endif
#endif
}
