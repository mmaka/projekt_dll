#pragma once
//#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include<cuda_runtime.h>
#include "device_launch_parameters.h"
#include "glew.h"
#include<stdlib.h>
#include<stdio.h>
#include<cuda_gl_interop.h>//ten nag³ówek dodawany wy¿ej generuje b³¹d
#include"OCTviewer.h"
#include<fstream>
#include"PlikBMP.h"
#include<sstream>
#include<vector>
#include<future>
#include<thread>



static void HandlerError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandlerError(err,__FILE__,__LINE__))

//using oct_t = unsigned long;
using oct_t = char;

enum class EDYCJA_MAPY_KOLOROW {ZMNIEJSZ_KONTRAST,ZWIEKSZ_KONTRAST,ZMNIEJSZ_JASNOSC,ZWIEKSZ_JASNOSC};

class CudaTekstury {

//	cudaArray_t *tabliceCuda2;
//	cudaSurfaceObject_t *bskany;
//	cudaSurfaceObject_t *przekrojePoprzeczne;
//	cudaSurfaceObject_t *przekrojePoziome;
	cudaStream_t *streams;

	std::vector<cudaArray_t> tabliceCuda;
	std::vector<cudaSurfaceObject_t> bskany;
	std::vector<cudaSurfaceObject_t> przekrojePoprzeczne;
	std::vector<cudaSurfaceObject_t> przekrojePoziome;
//	std::vector<cudaStream_t> streams;

	
//	unsigned int *daneGPU;
	oct_t *daneGPU;
	uchar4 *daneGPU_tab;
	uchar4 *daneGPU_ppop;
	uchar4 *daneGPU_ppoz;


	size liczbaBskanow;
	size liczbaPrzekrojowPoprzecznych;
	size liczbaPrzekrojowPoziomych;
	size rozmiarAskanu;
	size szerokoscBskanu;
	size glebokoscPomiaru;

	size krok_bskan;
	size krok_przekrojePoprzeczne;
	size krok_przekrojePoziome;


	const size liczbaStrumieni = 3;
	oct_t *daneCPU;//tmp
	//mapa kolorów
	int jasnosc, kontrast;
	unsigned char mapaSzarosci[256], defKol[256][3];
	uchar3 mapaKolorow[256];
	uchar4 mapaKolorySzarosc[256];
	unsigned char *d_mapaSzarosci;
	uchar3 *d_mapaKolorow;
	uchar4 *d_mapaKolory_Szarosc;
	bool inicjalizacja = false;
	bool zmianaKoloru;

public:
	 explicit CudaTekstury(visualizationParams params,char* dane): 
		liczbaBskanow(params.liczbaBskanow), 
		liczbaPrzekrojowPoprzecznych(params.liczbaPrzekrojowPoprzecznych), 
		liczbaPrzekrojowPoziomych(params.liczbaPrzekrojowPoziomych),
		jasnosc(params.jasnosc),
		kontrast(params.kontrast), 
		rozmiarAskanu(params.ascanSize_px),
		szerokoscBskanu(params.bscanSize_px),
		glebokoscPomiaru(params.depth_px),daneCPU(dane),zmianaKoloru(true){

		krok_bskan = (float)glebokoscPomiaru / (liczbaBskanow-1);
		krok_przekrojePoprzeczne = (float)rozmiarAskanu / (liczbaPrzekrojowPoprzecznych-1);
		krok_przekrojePoziome = (float)szerokoscBskanu / (liczbaPrzekrojowPoziomych-1);
		

	}
	
	CudaTekstury(const CudaTekstury&) = delete;
	void operator=(const CudaTekstury&) = delete;
	void init();
//	cudaArray_t* cudaArray() { return tabliceCuda2; }
	std::vector<cudaArray_t>& cudaArray() { return tabliceCuda; }
	void pobierzDaneCPU();
	inline size liczbaPrzekrojow() const { return liczbaBskanow; }// +liczbaPrzekrojowPoziomych + liczbaPrzekrojowPoprzecznych;} //constexpr
	inline size_t calkowityRozmiarDanych() const { return rozmiarAskanu*szerokoscBskanu*glebokoscPomiaru; }//constexpr
	void wczytajDane(const char *nazwaPliku);
	void wczytajBMP(char* plik);
	void wczytajDaneBinarne(char *nazwaPliku);
	void tworzPrzekroje();
	void launch_bskany(size_t i);
	void launch_bskany2();
	void launch_przekrojePoprzeczne(size_t i);
	void launch_przekrojePoziome(size_t i);
	void launch_przekrojePoprzeczne2();
	void launch_przekrojePoziome2();
	void tworzenie_tekstur(cudaArray_t *tab, cudaSurfaceObject_t *surf);
	void przygotowanieTekstur();
	void wprowadzTestoweDane();
	void pobierzDefinicjeKolorow();
	void ustawMapeKolorow();
	void edycjaMapyKolorow(EDYCJA_MAPY_KOLOROW tryb, int value);
	void odswiezTekstury();
	void odswiezTekstury2();
	void odswiez_bskany();
	void odswiez_przekrojePoprzeczne();
	void odswiez_przekrojePoziome();
	void pobierzDaneCPU2();
	void sprzatanie();
	void tworzDane();
	void launch_przygotowanie_bskanow(size_t i);
	void przepisanie();

	void ustawMapeKolorow2();
	void kolorowanieB();
	void kopiowanieB();
	void launch_bskany_bezkoloru(size_t i);
	void kopiowanieP();

	~CudaTekstury() {

//		if (inicjalizacja) {

			//for (size_t i = 0; i < liczbaStrumieni; ++i) HANDLE_ERROR(cudaStreamDestroy(streams[i]));

			//HANDLE_ERROR(cudaFree(d_mapaKolorow));
//			delete[] streams;
//		}
		

		
		
	//	delete[] tabliceCuda;
	//	delete[] bskany;
	//	delete[] przekrojePoprzeczne;
	//	delete[] przekrojePoziome;
	//	delete[] streams;

//		HANDLE_ERROR(cudaFree(daneGPU));
		//if (daneGPU != nullptr) delete[] daneGPU;
	//	if (daneCPU != nullptr) delete[] daneCPU;
	}
};

/*
static void ustanowienieWspolpracyCudaOpenGL(GLuint* indeksy, cudaArray_t* tablice_cuda, size_t ileTekstur) {

	cudaGraphicsResource_t* resources = new cudaGraphicsResource_t[ileTekstur];
	cudaStream_t strumien;
	cudaStreamCreateWithFlags(&strumien, cudaStreamDefault);
	GLenum target = GL_TEXTURE_2D;
	//unsigned int  flags = cudaGraphicsRegisterFlagsNone;
	unsigned int flags = cudaGraphicsRegisterFlagsSurfaceLoadStore;
	unsigned int arrayIndex = 0;
	unsigned int mipLevel = 0;

	for (size_t i = 0; i != ileTekstur; ++i) {

		HANDLE_ERROR(cudaGraphicsGLRegisterImage(&resources[i], indeksy[i], target, flags));
		HANDLE_ERROR(cudaGraphicsMapResources(1, &resources[i], strumien));
		HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&tablice_cuda[i], resources[i], arrayIndex, mipLevel));
	}

	HANDLE_ERROR(cudaGraphicsUnmapResources(ileTekstur, resources, strumien));

	delete[] resources;
}
*/

static void ustanowienieWspolpracyCudaOpenGL(GLuint* indeksy, std::vector<cudaArray_t>& tablice_cuda, int ileTekstur) {

	cudaGraphicsResource_t* resources = new cudaGraphicsResource_t[ileTekstur];
	cudaStream_t strumien;
	cudaStreamCreate(&strumien);
	GLenum target = GL_TEXTURE_2D;
	//unsigned int  flags = cudaGraphicsRegisterFlagsNone;
	unsigned int flags = cudaGraphicsRegisterFlagsSurfaceLoadStore;
	unsigned int arrayIndex = 0;
	unsigned int mipLevel = 0;
	tablice_cuda.resize(ileTekstur);

	for (size_t i = 0; i != ileTekstur; ++i) {

		HANDLE_ERROR(cudaGraphicsGLRegisterImage(&resources[i], indeksy[i], target, flags));
		HANDLE_ERROR(cudaGraphicsMapResources(1, &resources[i], strumien));
		HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&tablice_cuda[i], resources[i], arrayIndex, mipLevel));
	}

	HANDLE_ERROR(cudaGraphicsUnmapResources(ileTekstur, resources, strumien));

	delete[] resources;
	HANDLE_ERROR(cudaStreamDestroy(strumien));
}
