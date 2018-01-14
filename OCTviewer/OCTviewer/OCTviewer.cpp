#pragma once
#include "OCTviewer.h"
#include"Okno.h"
#include<thread>
#include"CudaTekstury.cuh"
#include<future>


Wizualizator& wizualizator() {

	static Wizualizator wiz;
	return wiz;

}

LRESULT CALLBACK  WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {

	return wizualizator().getOkno()->WndProc(hWnd, message, wParam, lParam);
}


OCTVIEWER_API void setParams(WIZUALIZACJA type, size ileBskanow,size ilePrzekrojowPoprzecznych,size ilePrzekrojowPoziomych, float xSizeScale, float ySizeScale, float zSizeScale, size bscanSize, size ascanSize, size depth, float x_size_mm, float y_size_mm, float z_size_mm,int jasn,int kontr,char *plik)
{
	wizualizator().setParams(type, ileBskanow, ilePrzekrojowPoprzecznych, ilePrzekrojowPoziomych, xSizeScale,ySizeScale,zSizeScale,bscanSize, ascanSize, depth, x_size_mm, y_size_mm,z_size_mm,jasn,kontr,plik);
};


OCTVIEWER_API void init()
{
	wizualizator().initCuda();
}

OCTVIEWER_API void loadData(const char* plik)
{

}

OCTVIEWER_API void updateData()
{

}

OCTVIEWER_API void display()
{
	wizualizator().wyswietlOkno();
}

OCTVIEWER_API void clear() 
{

}