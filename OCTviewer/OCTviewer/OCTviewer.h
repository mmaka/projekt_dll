#pragma once
#include<Windows.h>
//#include"Tools.h"


#ifdef OCTVIEWER_EXPORTS
#define OCTVIEWER_API __declspec(dllexport)
#else
#define OCTVIEWER_API __declspec(dllimport)
#endif


using size = unsigned int;

enum class WIZUALIZACJA { TYP_2D, TYP_3D };

struct visualizationParams {

	WIZUALIZACJA typ;
	size liczbaBskanow;
	size liczbaPrzekrojowPoprzecznych;
	size liczbaPrzekrojowPoziomych;
	float xSizeScale;
	float ySizeScale;
	float zSizeScale;
	size bscanSize_px;
	size ascanSize_px;
	size depth_px;
	float x_mm;
	float y_mm;
	float z_mm;
	int jasnosc;
	int kontrast;

};

extern "C" OCTVIEWER_API void setParams(WIZUALIZACJA type,size liczbaBskanow,size liczbaPrzekrojowPoprzecznych,size liczbaPrzekrojowPoziomych,float xSizeScale, float ySizeScale, float zSizeScale, size bscanSize,size ascanSize,size depth_px, float x_size_mm, float y_size_mm, float z_size_mm,int jasnosc,int kontrast);
extern "C" OCTVIEWER_API void init();
extern "C" OCTVIEWER_API void loadData(const char* plik);
extern "C" OCTVIEWER_API void loadColorDef(const char* plik);
extern "C" OCTVIEWER_API void display();
extern "C" OCTVIEWER_API void clear();

extern HINSTANCE hInstance;


