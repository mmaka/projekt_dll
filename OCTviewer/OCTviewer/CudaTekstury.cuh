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

	std::vector<cudaArray_t> tabliceCuda;
	std::vector<cudaStream_t> streams;

//	unsigned int *daneGPU;
	oct_t *daneGPU;

	oct_t *daneGPU_bskan_oct;
	oct_t *daneGPU_ppop_oct;
	oct_t *daneGPU_ppoz_oct;

	uchar4 *daneGPU_bskan_kolor;
	uchar4 *daneGPU_ppop_kolor;
	uchar4 *daneGPU_ppoz_kolor;
	
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


	size liczbaStrumieni = 3;
	oct_t *daneCPU;//tmp
	//mapa kolorów
	int jasnosc, kontrast;
	unsigned char defKol[256][3];
	uchar4 mapaKolorySzarosc[256];
	uchar4 *d_mapaKolory_Szarosc;
	bool inicjalizacja = false;
	bool kolor;
	std::atomic_flag zmianaTrybuRGBnaGS = ATOMIC_FLAG_INIT;
	std::atomic_flag przetwarzanieMapyKolorow = ATOMIC_FLAG_INIT;
	WIZUALIZACJA trybWyswietlania;

public:
	 explicit CudaTekstury(visualizationParams params,char* dane,unsigned char kolory[256][3]): 
	
		 jasnosc(params.jasnosc),
		 kontrast(params.kontrast), 
		 rozmiarAskanu(params.ascanSize_px),
		 szerokoscBskanu(params.bscanSize_px),
		 glebokoscPomiaru(params.depth_px),
		 daneCPU(dane),
		 kolor(true),
		 trybWyswietlania(params.typ){

		 //jakis swap trzeba tu zrobic
		 for (int i = 0; i < 256; ++i) {
			 defKol[i][0] = kolory[i][0];
			 defKol[i][1] = kolory[i][1];
			 defKol[i][2] = kolory[i][2];
		}
		
		 liczbaBskanow = params.liczbaBskanow;
		 krok_bskan = (float)glebokoscPomiaru / (liczbaBskanow-1);

		 if (params.typ == WIZUALIZACJA::TYP_3D) {

			 liczbaStrumieni = 3;
			 liczbaPrzekrojowPoprzecznych=params.liczbaPrzekrojowPoprzecznych;
			 liczbaPrzekrojowPoziomych=params.liczbaPrzekrojowPoziomych;
			 krok_przekrojePoprzeczne = (float)rozmiarAskanu / (liczbaPrzekrojowPoprzecznych - 1);
			 krok_przekrojePoziome = (float)szerokoscBskanu / (liczbaPrzekrojowPoziomych - 1);
		 }
	}
	
	CudaTekstury(const CudaTekstury&) = delete;
	void operator=(const CudaTekstury&) = delete;
	void init();
	std::vector<cudaArray_t>& cudaArray() { return tabliceCuda; }
	void pobierzDaneCPU();
	inline size_t calkowityRozmiarDanych() const { return rozmiarAskanu*szerokoscBskanu*glebokoscPomiaru; }
	void ustawMapeKolorow();
	void trybWyswietlaniaRGBczyGS();
	void edycjaMapyKolorow(EDYCJA_MAPY_KOLOROW tryb, int value);
	void odswiez_bskany();
	void odswiez_przekrojePoprzeczne();
	void odswiez_przekrojePoziome();
	void sprzatanie();
	void przepisanie_oct_t_ppop();
	void przepisanie_oct_t_ppoz();
	void przepisanie_oct_t_bskan();
	void przepisanie_oct_t();
	void kolorowaniePpoz();
	void kolorowaniePpop();
	void kolorowanieBskan();
	void kolorowanie_oct_t();
	void kopiowaniePrzekrojow();
	void pokolorujTeksturyIprzeslijDoTablicCuda();
	void ppoz_przepisanie_i_kolorowanie();
	void ppop_przepisanie_i_kolorowanie();
	void bskan_przepisanie_i_kolorowanie();

	~CudaTekstury(){

		
//		HANDLE_ERROR(cudaFree(daneGPU_tab));
//		HANDLE_ERROR(cudaFree(daneGPU_ppop));
//		HANDLE_ERROR(cudaFree(daneGPU_ppoz));

	}
};

static void ustanowienieWspolpracyCudaOpenGL(const std::vector<GLuint>& indeksy, std::vector<cudaArray_t>& tablice_cuda) {

	size_t ileTekstur = indeksy.size();
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
