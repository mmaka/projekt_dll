#include "OCTviewer.h"
#include"Okno.h"
#include<thread>
#include"CudaTekstury.cuh"

visualizationParams params;
OknoGL *okno;
CudaTekstury *cudaTekstury;
std::thread *t;
//CudaTekstury *cudaTekstury = nullptr;
int StworzOkno();

LRESULT CALLBACK  WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {

	return okno->WndProc(hWnd, message, wParam, lParam);
}


OCTVIEWER_API void setParams(visualizationType type, size ileBskanow,size ilePrzekrojowPoprzecznyc,size ilePrzekrojowPoziomych, float xSizeScale, float ySizeScale, float zSizeScale, size bscanSize, size ascanSize, size depth, float x_size_mm, float y_size_mm, float z_size_mm,int jasn,int kontr)
{
	params.type = type;
	params.liczbaBskanow = ileBskanow;
	params.liczbaPrzekrojowPoprzecznych = ilePrzekrojowPoprzecznyc;
	params.liczbaPrzekrojowPoziomych = ilePrzekrojowPoziomych;
	params.xSizeScale = xSizeScale;
	params.ySizeScale = ySizeScale;
	params.zSizeScale = zSizeScale;
	params.bscanSize_px = bscanSize;
	params.ascanSize_px = ascanSize;
	params.depth_px = depth;
	params.x_mm = x_size_mm;
	params.y_mm = y_size_mm;
	params.z_mm = z_size_mm;
	params.jasnosc = jasn;
	params.kontrast = kontr;
};


OCTVIEWER_API void init()
{
	MessageBox(NULL, "init", "", MB_OK);
	cudaTekstury = new CudaTekstury(params);
	//cudaTekstury.getInstance(params);
}

OCTVIEWER_API void loadData(const char* plik)
{
	//tutaj musi byæ przygotowany bufor, tak by u¿ytkownik móg³ za³adowaæ dane w dowolnym momencie, ale przekazanie ich do GPU nast¹pi³o wtedy gdy stworzenie okna przebieg³o pomyœlnie
	

	MessageBox(NULL, "load1", "", MB_OK);
//	cudaTekstury->wczytajBMP("lab512x256.bmp");
//	cudaTekstury->wczytajDane(plik);
//	cudaTekstury->wczytajDaneBinarne("kostka.bin");
//	cTekstury->wprowadzTestoweDane();
	MessageBox(NULL, "load2", "", MB_OK);
//	cudaTekstury->pobierzDaneCPU();
	MessageBox(NULL, "load3", "", MB_OK);
//	cudaTekstury->tworzPrzekroje();
	MessageBox(NULL, "load4", "", MB_OK);
}

OCTVIEWER_API void updateData()
{


}

OCTVIEWER_API void display()
{
//	std::thread t2(StworzOkno);
	t = new std::thread(StworzOkno);
//	t2.detach();
	t->detach();
}

OCTVIEWER_API void clear() {

	//delete t;
	//delete okno;
//	if(cudaTekstury!=nullptr)
//		delete cudaTekstury;

}

int StworzOkno() {

	
	okno = new OknoGL(params,cudaTekstury);
	POINT polozenieOkna = { 100,100 };
	POINT rozmiarOkna = { 800,600 };
	if (!okno->Inicjuj(hInstance, polozenieOkna, rozmiarOkna)) {

		MessageBox(NULL, "Inicjacja okna nie powiodla sie", "Aplikacja OpenGL", MB_OK | MB_ICONERROR);
		return EXIT_FAILURE;
	}
	else return okno->Uruchom();
}