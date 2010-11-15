#include <locale_management.h>
#include <stdlib.h>

#ifdef _GNU_SOURCE
#include <xlocale.h>
#include <locale.h>
#else
#ifdef PYMCA_POSIX
#else
#ifdef SPECFILE_POSIX
#include <locale.h>
#endif
#endif
#endif

#include <string.h>

double PyMcaAtof(const char * inputString)
{
#ifdef _GNU_SOURCE
	double result;
	locale_t newLocale;
	newLocale = newlocale(LC_NUMERIC_MASK, "C", NULL); 
	result = strtod_l(inputString, NULL, newLocale);
	freelocale(newLocale);
	return result;
#else
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
#endif
}
