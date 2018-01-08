#pragma once
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




static void HandlerError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandlerError(err,__FILE__,__LINE__))

//using oct_t = unsigned long;
using oct_t = unsigned long;

class CudaTekstury {

	cudaArray_t *tabliceCuda;
	cudaSurfaceObject_t *bskany;
	cudaSurfaceObject_t *przekrojePoprzeczne;
	cudaSurfaceObject_t *przekrojePoziome;
	cudaStream_t *streams;
//	unsigned int *daneGPU;
	oct_t *daneGPU;
	
	size liczbaBskanow;
	size liczbaPrzekrojowPoprzecznych;
	size liczbaPrzekrojowPoziomych;
	size rozmiarAskanu;
	size szerokoscBskanu;
	size glebokoscPomiaru;

	size krok_bskan;
	size krok_przekrojePoprzeczne;
	size krok_przekrojePoziome;


	const size liczbaStrumieni = 1024;
	oct_t *daneCPU;//tmp
	//char *daneCPU;
	//mapa kolorów
	int jasnosc, kontrast;
	unsigned char mapaSzarosci[256], defKol[256][3];
	uchar3 mapaKolorow[256];
	unsigned char *d_mapaSzarosci;
	uchar3 *d_mapaKolorow;

public:
	 explicit CudaTekstury(visualizationParams params): 
		liczbaBskanow(params.liczbaBskanow), 
		liczbaPrzekrojowPoprzecznych(params.liczbaPrzekrojowPoprzecznych), 
		liczbaPrzekrojowPoziomych(params.liczbaPrzekrojowPoziomych),
		jasnosc(params.jasnosc),
		kontrast(params.kontrast), 
		rozmiarAskanu(params.ascanSize_px),
		szerokoscBskanu(params.bscanSize_px),
		glebokoscPomiaru(params.depth_px){

		krok_bskan = (float)glebokoscPomiaru / (liczbaBskanow-1);
		krok_przekrojePoprzeczne = (float)rozmiarAskanu / (liczbaPrzekrojowPoprzecznych-1);
		krok_przekrojePoziome = (float)szerokoscBskanu / (liczbaPrzekrojowPoziomych-1);
		

	}

//	static CudaTekstury& getInstance(visualizationParams params) {

//		static CudaTekstury instance(params);
//		return instance;
//	}
	CudaTekstury(const CudaTekstury&) = delete;
	void operator=(const CudaTekstury&) = delete;
	void init();
	cudaArray_t* cudaArray() { return tabliceCuda; }
	void pobierzDaneCPU();
	inline size liczbaPrzekrojow() const { return liczbaPrzekrojowPoziomych+ liczbaPrzekrojowPoprzecznych+ liczbaBskanow;}
	inline size_t calkowityRozmiarDanych() const { return rozmiarAskanu*szerokoscBskanu*glebokoscPomiaru; }
	void wczytajDane(const char *nazwaPliku);
	void wczytajBMP(char* plik);
	void wczytajDaneBinarne(char *nazwaPliku);
	void tworzPrzekroje();
	void launch_bskany(size_t i);
	void launch_przekrojePoprzeczne(size_t i);
	void launch_przekrojePoziome(size_t i);
	void tworzenie_tekstur(cudaArray_t *tab, cudaSurfaceObject_t *surf);
	void przygotowanieTekstur();
	void wprowadzTestoweDane();
	void pobierzDefinicjeKolorow();
	void ustawMapeKolorow();
//	void zwiekszKontrast();

	~CudaTekstury() {

		for (size_t i = 0; i < liczbaStrumieni; ++i) HANDLE_ERROR(cudaStreamDestroy(streams[i]));

		//HANDLE_ERROR(cudaFree(d_mapaSzarosci));
		HANDLE_ERROR(cudaFree(d_mapaKolorow));
		delete[] tabliceCuda;
		delete[] bskany;
		delete[] przekrojePoprzeczne;
		delete[] przekrojePoziome;
		delete[] streams;
		if (daneGPU != nullptr) delete[] daneGPU;
		if (daneCPU != nullptr) delete[] daneCPU;
	}
};


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
