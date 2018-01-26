#include"CudaTekstury.cuh"

__constant__ size rozmiaryDanych[6];

void CudaTekstury::init() {

	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	HANDLE_ERROR(cudaGLSetGLDevice(dev));
	
	auto f = std::async(std::launch::async, [&] {

		ustawMapeKolorow();
		HANDLE_ERROR(cudaMalloc(&d_mapaKolory_Szarosc, 256 * sizeof(uchar4)));
		HANDLE_ERROR(cudaMemcpy(d_mapaKolory_Szarosc, mapaKolorySzarosc, 256 * sizeof(uchar4), cudaMemcpyHostToDevice));

	});

	streams.resize(liczbaStrumieni);
	for (size_t i = 0; i < liczbaStrumieni; ++i)
		HANDLE_ERROR(cudaStreamCreate(&streams[i]));

	inicjalizacja = true;
	
}

void CudaTekstury::pobierzDaneCPU() {
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	HANDLE_ERROR(cudaMalloc(&daneGPU, calkowityRozmiarDanych() *sizeof(oct_t)));
	HANDLE_ERROR(cudaMemcpy(daneGPU, daneCPU, calkowityRozmiarDanych() * sizeof(oct_t), cudaMemcpyHostToDevice));
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	printf("czas kopiowania: %f\n", j);
	delete[] daneCPU;
	daneCPU = nullptr;
}

void CudaTekstury::ustawMapeKolorow() {

	int progCzerni = jasnosc - (kontrast / 2);//do progu czerni kolor czarny
	

	for (int i = 0, end = (progCzerni<256) ? progCzerni : 256; i != end; ++i) {
		
		mapaKolorySzarosc[i].x = 0;
		mapaKolorySzarosc[i].y = 0;
		mapaKolorySzarosc[i].z = 0;
		mapaKolorySzarosc[i].w = 0;
		
	}
	//miedzy czarnym i bia³ym przedzia³ kolorow
	int progSzarosciKolorow = jasnosc + (kontrast / 2);
	//tutaj jest b³¹d: progCzerni moze byc wieksz niz 256 -> spojrz na warunek konca powy¿szej pêtli
	for (int i = progCzerni, przedzial = progSzarosciKolorow - progCzerni, end = (progSzarosciKolorow<256) ? progSzarosciKolorow : 256; i != end; ++i) {

		unsigned char val = (unsigned char)((255 * ((float)i - progCzerni)) / przedzial);
		mapaKolorySzarosc[i].x = defKol[val][0];
		mapaKolorySzarosc[i].y = defKol[val][1];
		mapaKolorySzarosc[i].z = defKol[val][2];
		mapaKolorySzarosc[i].w = val;

	}
	//powyzej progu szarosci kolor bialy
	for (int i = progSzarosciKolorow; i != 256; ++i) {

		mapaKolorySzarosc[i].x = 255;
		mapaKolorySzarosc[i].y = 255;
		mapaKolorySzarosc[i].z = 255;
		mapaKolorySzarosc[i].w = 255;

	}
}


void CudaTekstury::edycjaMapyKolorow(EDYCJA_MAPY_KOLOROW tryb, int value) {

	//gwarantujemy, ¿e kolejne przetwarzanie mapy zacznie siê po zakoñczeniu trwaj¹cego przetwarzania
	if (!przetwarzanieMapyKolorow.test_and_set()) {
		
		//przed przetworzeniem mapy kolorów i uruchomieniem funkcji j¹dra sprawdzamy czy wartoœci kontrastu i jasnoœci nale¿¹ do odpowiedniego zakresu
		//np. gdy kontrast osi¹gnie wartoœæ 0, zmniejszanie wartoœci nie powinno mieæ miejsca
		bool czyOdswiezamyKoloryPrzekrojow = false;

		switch (tryb)
		{
		case EDYCJA_MAPY_KOLOROW::ZWIEKSZ_KONTRAST:
		{
			int tmp = kontrast - value;

			if (tmp >= 0) {

				kontrast = tmp;
				czyOdswiezamyKoloryPrzekrojow = true;

			}
			else {

				kontrast = 0;

			}
		}
		break;
		case EDYCJA_MAPY_KOLOROW::ZMNIEJSZ_KONTRAST:
		{
			int tmp = kontrast + value;

			if (tmp <= 256) {

				kontrast = tmp;
				czyOdswiezamyKoloryPrzekrojow = true;

			}
			else {

				kontrast = 256;
			}
		}
		break;
		case EDYCJA_MAPY_KOLOROW::ZWIEKSZ_JASNOSC:
		{
			int tmp = jasnosc - value;

			if (tmp >= 0) {

				jasnosc = tmp;
				czyOdswiezamyKoloryPrzekrojow = true;

			}
			else {

				jasnosc = 0;
			}
		}
		break;
		case EDYCJA_MAPY_KOLOROW::ZMNIEJSZ_JASNOSC:
		{
			int tmp = jasnosc + value;

			if (jasnosc <= 256) {

				jasnosc = tmp;
				czyOdswiezamyKoloryPrzekrojow = true;

			}
			else {

				jasnosc = 256;
			}
		}
		break;
		default:
			break;
		}

		if (czyOdswiezamyKoloryPrzekrojow) {

			ustawMapeKolorow();
			LARGE_INTEGER countPerSec, tim1, tim2;
			QueryPerformanceFrequency(&countPerSec);
			QueryPerformanceCounter(&tim1);
			HANDLE_ERROR(cudaMemcpy(d_mapaKolory_Szarosc, mapaKolorySzarosc, 256 * sizeof(uchar4), cudaMemcpyHostToDevice));
			QueryPerformanceCounter(&tim2);
			double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
			//	printf("czas kopiowania mapy: %f\n", j);
			pokolorujTeksturyIprzeslijDoTablicCuda();

		}

		przetwarzanieMapyKolorow.clear();
	}
	
}
void CudaTekstury::trybWyswietlaniaRGBczyGS(){

	if (!zmianaTrybuRGBnaGS.test_and_set()) {

		kolor = !kolor;
		pokolorujTeksturyIprzeslijDoTablicCuda();
		zmianaTrybuRGBnaGS.clear();

	}
}


void CudaTekstury::sprzatanie() {

	if (inicjalizacja) {

		HANDLE_ERROR(cudaFree(d_mapaKolory_Szarosc));
	//	HANDLE_ERROR(cudaFree(daneGPU));
	//	HANDLE_ERROR(cudaFree(daneGPU_bskan_oct));
	//	HANDLE_ERROR(cudaFree(daneGPU_ppop_oct));
	//	HANDLE_ERROR(cudaFree(daneGPU_ppoz_oct));

		HANDLE_ERROR(cudaFree(daneGPU_bskan_kolor));
		if (trybWyswietlania == WIZUALIZACJA::TYP_3D) {

			HANDLE_ERROR(cudaFree(daneGPU_ppop_kolor));
			HANDLE_ERROR(cudaFree(daneGPU_ppoz_kolor));
		}
		
		for (size_t i = 0; i < liczbaStrumieni; ++i) HANDLE_ERROR(cudaStreamDestroy(streams[i]));

	}
	
	//if (daneGPU != nullptr) delete[] daneGPU;
	if (daneCPU != nullptr) delete[] daneCPU;

}

__global__ void przepisanieObuPrzekrojow(const uchar4 *dstGPU, uchar4 * dstGPU_ppop, uchar4 * dstGPU_ppoz) {

	if (threadIdx.x < blockDim.x) {

		dstGPU_ppop[blockIdx.x*blockDim.x*gridDim.y + blockIdx.y*blockDim.x + threadIdx.x] = dstGPU[blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]; 
		dstGPU_ppoz[threadIdx.x*gridDim.y*gridDim.x + blockIdx.x*gridDim.y + blockIdx.y] = dstGPU[blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
																								
	}
}

__global__ void kolorowanie_bskan(uchar4 *dstGPU, const oct_t *source, const uchar4 *kolory ) {

	if (threadIdx.x < blockDim.x) {

		dstGPU[blockIdx.x*blockDim.x + threadIdx.x] = kolory[(unsigned char)source[blockIdx.x*blockDim.x + threadIdx.x]];
		dstGPU[blockIdx.x*blockDim.x + threadIdx.x].w = (unsigned char)source[blockIdx.x*blockDim.x + threadIdx.x];	
	}
}

__global__ void kolorowanie_ppop(uchar4 *dstGPU, const oct_t *source, const uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {

		dstGPU[blockIdx.x*blockDim.x + threadIdx.x] = kolory[(unsigned char)source[blockIdx.x*blockDim.x + threadIdx.x]];
		dstGPU[blockIdx.x*blockDim.x + threadIdx.x].w = (unsigned char)source[blockIdx.x*blockDim.x + threadIdx.x];
	}
}

__global__ void kolorowanie_ppoz(uchar4 *dstGPU, const oct_t *source, const uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {

		dstGPU[blockIdx.x*blockDim.x + threadIdx.x] = kolory[(unsigned char)source[blockIdx.x*blockDim.x + threadIdx.x]];
		dstGPU[blockIdx.x*blockDim.x + threadIdx.x].w = (unsigned char)source[blockIdx.x*blockDim.x + threadIdx.x];
	}
}

__global__ void wybraniePpoz(const oct_t *dstGPU, oct_t * dstGPU_ppoz,size krok_ppoz,size szerB) {

	if (threadIdx.x < blockDim.x) {

	//	register int ktory_ppoz = blockIdx.y*krok_ppoz;//chyba bez sensu
		dstGPU_ppoz[blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x] = dstGPU[threadIdx.x*gridDim.x*szerB + blockIdx.x*szerB + blockIdx.y*krok_ppoz];
	//	dstGPU_ppoz[blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x] = dstGPU[threadIdx.x*gridDim.x*rozmiaryDanych[0] + blockIdx.x*rozmiaryDanych[0] + blockIdx.y*rozmiaryDanych[5]];
	}
}


__global__ void wybraniePpop(const oct_t *dstGPU, oct_t * dstGPU_ppop, size krok_ppop, size rozA) {

	if (threadIdx.x < blockDim.x) {

	//	register int ktory_ppop = blockIdx.x * krok_ppop;
		dstGPU_ppop[blockIdx.x*blockDim.x*gridDim.y + blockIdx.y*blockDim.x + threadIdx.x] = dstGPU[blockIdx.y*rozA * blockDim.x + blockIdx.x * krok_ppop * blockDim.x + threadIdx.x];
		//dstGPU_ppop[blockIdx.x*blockDim.x*gridDim.y + blockIdx.y*blockDim.x + threadIdx.x] = dstGPU[blockIdx.y*rozmiaryDanych[1] *blockDim.x + blockIdx.x * rozmiaryDanych[4] *blockDim.x + threadIdx.x];

	}
}


__global__ void wybranieBskanow(const oct_t *dstGPU, oct_t * dstGPU_bskan,size krok_bskany) {

	if (threadIdx.x < blockDim.x) {

		//register int ktory_bskan = blockIdx.y*krok_bskany;
		//dstGPU_bskan[blockIdx.y*blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x] = dstGPU[blockIdx.y*rozmiaryDanych[3]*blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		dstGPU_bskan[blockIdx.y*blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x] = dstGPU[blockIdx.y*krok_bskany * blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x];

	}
}

__global__ void wyborIKolorowaniePpoz(const oct_t *dstGPU, uchar4 * dstGPU_ppoz, size krok_ppoz, size szerB, const uchar4* kolory) {

	if (threadIdx.x < blockDim.x) {

		register int ktory_ppoz = blockIdx.y*krok_ppoz;//chyba bez sensu
		dstGPU_ppoz[blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x] = kolory[dstGPU[threadIdx.x*gridDim.x*szerB + blockIdx.x*szerB + ktory_ppoz]];
	}
}


__global__ void  wyborIKolorowaniePpop(const oct_t *dstGPU, uchar4 * dstGPU_ppop, size krok_ppop, size rozA, const uchar4* kolory) {

	if (threadIdx.x < blockDim.x) {

		register int ktory_ppop = blockIdx.x * krok_ppop;
		dstGPU_ppop[blockIdx.x*blockDim.x*gridDim.y + blockIdx.y*blockDim.x + threadIdx.x] = kolory[dstGPU[blockIdx.y*rozA*blockDim.x + ktory_ppop*blockDim.x + threadIdx.x]]; //dstGPU[nrB + nrW + nrKol];

	}
}


__global__ void  wyborIKolorowanieBskan(const oct_t *dstGPU, uchar4 * dstGPU_bskan, size krok_bskany,const uchar4* kolory) {

	if (threadIdx.x < blockDim.x) {

		register int ktory_bskan = blockIdx.y*krok_bskany;
		dstGPU_bskan[blockIdx.y*blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x] = kolory[dstGPU[ktory_bskan*blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x]];
	}
}


void CudaTekstury::kolorowanieBskan() {

	HANDLE_ERROR(cudaMalloc(&daneGPU_bskan_kolor, szerokoscBskanu*rozmiarAskanu*liczbaBskanow * sizeof(uchar4)));
	dim3 block(szerokoscBskanu);
	dim3 grid(rozmiarAskanu*liczbaBskanow);

	cudaFuncSetCacheConfig(kolorowanie_bskan, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	kolorowanie_bskan << <grid, block, 0, streams[0] >> >(daneGPU_bskan_kolor, daneGPU_bskan_oct, d_mapaKolory_Szarosc);
	//kolorowanie_bskan << <grid, block>> >(daneGPU_bskan_kolor, daneGPU_bskan_oct, d_mapaKolory_Szarosc);
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas kolorowania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernelll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}

void CudaTekstury::kolorowaniePpop() {

	HANDLE_ERROR(cudaMalloc(&daneGPU_ppop_kolor, liczbaPrzekrojowPoprzecznych*szerokoscBskanu*glebokoscPomiaru * sizeof(uchar4)));

	dim3 block(szerokoscBskanu);
	dim3 grid(liczbaPrzekrojowPoprzecznych*glebokoscPomiaru);
	cudaFuncSetCacheConfig(kolorowanie_ppop, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	kolorowanie_ppop << <grid, block, 0, streams[1] >> >(daneGPU_ppop_kolor, daneGPU_ppop_oct, d_mapaKolory_Szarosc);
	//kolorowanie_ppop << <grid, block>> >(daneGPU_ppop_kolor, daneGPU_ppop_oct, d_mapaKolory_Szarosc);
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas kolorowania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernelll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}

void CudaTekstury::kolorowaniePpoz() {

	HANDLE_ERROR(cudaMalloc(&daneGPU_ppoz_kolor, rozmiarAskanu*glebokoscPomiaru*liczbaPrzekrojowPoziomych * sizeof(uchar4)));

	dim3 block(glebokoscPomiaru);
	dim3 grid(rozmiarAskanu*liczbaPrzekrojowPoziomych);
	cudaFuncSetCacheConfig(kolorowanie_ppoz, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	kolorowanie_ppoz << <grid, block, 0, streams[2] >> >(daneGPU_ppoz_kolor, daneGPU_ppoz_oct, d_mapaKolory_Szarosc);	
	//kolorowanie_ppoz << <grid, block>> >(daneGPU_ppoz_kolor, daneGPU_ppoz_oct, d_mapaKolory_Szarosc);
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas kolorowania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernelll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}

void CudaTekstury::ppoz_przepisanie_i_kolorowanie() {


	HANDLE_ERROR(cudaMalloc(&daneGPU_ppoz_kolor, rozmiarAskanu*glebokoscPomiaru*liczbaPrzekrojowPoziomych * sizeof(uchar4)));

	dim3 block(glebokoscPomiaru);
	dim3 grid(rozmiarAskanu, liczbaPrzekrojowPoziomych);
	cudaFuncSetCacheConfig(wyborIKolorowaniePpoz, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	//wyborIKolorowaniePpoz << <grid, block>> > (daneGPU, daneGPU_ppoz_kolor, krok_przekrojePoziome, szerokoscBskanu, mapaKolorySzarosc);
	wyborIKolorowaniePpoz << <grid, block, 0, streams[2] >> > (daneGPU, daneGPU_ppoz_kolor, krok_przekrojePoziome, szerokoscBskanu, mapaKolorySzarosc);
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas przepisania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernellll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}


void CudaTekstury::ppop_przepisanie_i_kolorowanie() {


	HANDLE_ERROR(cudaMalloc(&daneGPU_ppop_kolor, szerokoscBskanu*glebokoscPomiaru*liczbaPrzekrojowPoprzecznych * sizeof(uchar4)));

	dim3 block(szerokoscBskanu);
	dim3 grid(liczbaPrzekrojowPoprzecznych, glebokoscPomiaru);
	cudaFuncSetCacheConfig(wyborIKolorowaniePpop, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	wyborIKolorowaniePpop << <grid, block, 0, streams[1] >> > (daneGPU, daneGPU_ppop_kolor, krok_przekrojePoprzeczne, rozmiarAskanu, mapaKolorySzarosc);
	//wyborIKolorowaniePpop << <grid, block>> > (daneGPU, daneGPU_ppop_kolor, krok_przekrojePoprzeczne, rozmiarAskanu, mapaKolorySzarosc);
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas przepisania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernellll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}


void CudaTekstury::bskan_przepisanie_i_kolorowanie() {


	HANDLE_ERROR(cudaMalloc(&daneGPU_bskan_kolor, szerokoscBskanu*liczbaBskanow*rozmiarAskanu * sizeof(uchar4)));

	dim3 block(szerokoscBskanu);
	dim3 grid(rozmiarAskanu, liczbaBskanow);

	cudaFuncSetCacheConfig(wyborIKolorowanieBskan, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	wyborIKolorowanieBskan << <grid, block, 0, streams[0] >> > (daneGPU, daneGPU_bskan_kolor, krok_bskan,mapaKolorySzarosc);
	//wyborIKolorowanieBskan << <grid, block>> > (daneGPU, daneGPU_bskan_kolor, krok_bskan, mapaKolorySzarosc);
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas przepisania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernellll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}


void CudaTekstury::przepisanie_oct_t_ppoz() {

	HANDLE_ERROR(cudaMalloc(&daneGPU_ppoz_oct, rozmiarAskanu*glebokoscPomiaru*liczbaPrzekrojowPoziomych * sizeof(oct_t)));

	dim3 block(glebokoscPomiaru);
	dim3 grid(rozmiarAskanu, liczbaPrzekrojowPoziomych);
	cudaFuncSetCacheConfig(wybraniePpoz, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	wybraniePpoz << <grid, block, 0, streams[2] >> > (daneGPU,daneGPU_ppoz_oct,krok_przekrojePoziome,szerokoscBskanu);
	//wybraniePpoz << <grid, block>> > (daneGPU, daneGPU_ppoz_oct, krok_przekrojePoziome, szerokoscBskanu);
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas przepisania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernellll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}


void CudaTekstury::przepisanie_oct_t_ppop() {


	HANDLE_ERROR(cudaMalloc(&daneGPU_ppop_oct, szerokoscBskanu*glebokoscPomiaru*liczbaPrzekrojowPoprzecznych * sizeof(oct_t)));
	
	dim3 block(szerokoscBskanu);
	dim3 grid(liczbaPrzekrojowPoprzecznych, glebokoscPomiaru);
	cudaFuncSetCacheConfig(wybraniePpop, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	wybraniePpop << <grid, block, 0, streams[1] >> > (daneGPU, daneGPU_ppop_oct,krok_przekrojePoprzeczne,rozmiarAskanu);
	//wybraniePpop << <grid, block>> > (daneGPU, daneGPU_ppop_oct, krok_przekrojePoprzeczne, rozmiarAskanu);
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas przepisania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernellll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}


void CudaTekstury::przepisanie_oct_t_bskan() {


	HANDLE_ERROR(cudaMalloc(&daneGPU_bskan_oct, szerokoscBskanu*liczbaBskanow*rozmiarAskanu * sizeof(oct_t)));

	dim3 block(szerokoscBskanu);
	dim3 grid(rozmiarAskanu, liczbaBskanow);
	
	cudaFuncSetCacheConfig(wybranieBskanow, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	wybranieBskanow << <grid, block,0,streams[0] >> > (daneGPU, daneGPU_bskan_oct,krok_bskan);
	//wybranieBskanow << <grid, block>> > (daneGPU, daneGPU_bskan_oct, krok_bskan);
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas przepisania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernellll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}

}

void CudaTekstury::przepisanie_oct_t() {

	bskan_przepisanie_i_kolorowanie();
	ppoz_przepisanie_i_kolorowanie();
	ppop_przepisanie_i_kolorowanie();
}

void CudaTekstury::kolorowanie_oct_t() {
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
//	HANDLE_ERROR(cudaFree(daneGPU));

//std::future<void> f1;
//std::future<void> f2;
//std::future<void> f3;

std::thread t1;
std::thread t2;
std::thread t3;

	switch (trybWyswietlania)
	{
	case WIZUALIZACJA::TYP_3D:
	
//		f1 = std::async(std::launch::async, [&] {przepisanie_oct_t_bskan();kolorowanieBskan(); });
//		f2 = std::async(std::launch::async, [&] {przepisanie_oct_t_ppop();	kolorowaniePpop(); });
//		f3 = std::async(std::launch::async, [&] {przepisanie_oct_t_ppoz();	kolorowaniePpoz(); });
		
		t1 = std::thread([&] {przepisanie_oct_t_bskan(); kolorowanieBskan(); });
		t2 = std::thread([&] {przepisanie_oct_t_ppop();	kolorowaniePpop(); });
		t3 = std::thread([&] {przepisanie_oct_t_ppoz();	kolorowaniePpoz(); });

		break;
	case WIZUALIZACJA::TYP_2D:
	
//		f1 = std::async(std::launch::async, [&] {przepisanie_oct_t_bskan(); kolorowanieBskan(); });
		t1 = std::thread([&] {przepisanie_oct_t_bskan(); kolorowanieBskan(); });
		break;
	default:
		break;
	}

	
	t1.join();
	if (trybWyswietlania == WIZUALIZACJA::TYP_3D) {
		t2.join();
		t3.join();
	}
	

	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	printf("czas kolorowanie_oct: %f\n", j);
}



void CudaTekstury::kopiowaniePrzekrojow() {

	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	cudaDeviceSynchronize();

//	std::future<void> f1;
//	std::future<void> f2;
//	std::future<void> f3;

	std::thread t1;
	std::thread t2;
	std::thread t3;

	switch (trybWyswietlania) {

	case WIZUALIZACJA::TYP_3D:
/*	
		f1 = std::async(std::launch::async, [&] {
			for (int i = 0; i < liczbaBskanow; ++i)
				HANDLE_ERROR(cudaMemcpy2DToArrayAsync(tabliceCuda[i], 0, 0, daneGPU_bskan_kolor + i*rozmiarAskanu*szerokoscBskanu, szerokoscBskanu * sizeof(uchar4), szerokoscBskanu * sizeof(uchar4), rozmiarAskanu, cudaMemcpyDeviceToDevice, streams[0]));

		});

		f2 = std::async(std::launch::async, [&] {

			for (int i = 0; i < liczbaPrzekrojowPoprzecznych; ++i)
				HANDLE_ERROR(cudaMemcpy2DToArrayAsync(tabliceCuda[i + liczbaBskanow], 0, 0, daneGPU_ppop_kolor + i*glebokoscPomiaru*szerokoscBskanu, szerokoscBskanu * sizeof(uchar4), szerokoscBskanu * sizeof(uchar4), glebokoscPomiaru, cudaMemcpyDeviceToDevice, streams[1]));

		});

		f3 = std::async(std::launch::async, [&] {

			for (int i = 0; i < liczbaPrzekrojowPoziomych; ++i)
				HANDLE_ERROR(cudaMemcpy2DToArrayAsync(tabliceCuda[i + liczbaBskanow + liczbaPrzekrojowPoprzecznych], 0, 0, daneGPU_ppoz_kolor + i*glebokoscPomiaru*rozmiarAskanu, glebokoscPomiaru * sizeof(uchar4), glebokoscPomiaru * sizeof(uchar4), rozmiarAskanu, cudaMemcpyDeviceToDevice, streams[2]));

		});
*/
	t1 = std::thread([&] {
		for (int i = 0; i < liczbaBskanow; ++i)
			HANDLE_ERROR(cudaMemcpy2DToArrayAsync(tabliceCuda[i], 0, 0, daneGPU_bskan_kolor + i*rozmiarAskanu*szerokoscBskanu, szerokoscBskanu * sizeof(uchar4), szerokoscBskanu * sizeof(uchar4), rozmiarAskanu, cudaMemcpyDeviceToDevice, streams[0]));

	});

	t2 = std::thread([&] {

		for (int i = 0; i < liczbaPrzekrojowPoprzecznych; ++i)
			HANDLE_ERROR(cudaMemcpy2DToArrayAsync(tabliceCuda[i + liczbaBskanow], 0, 0, daneGPU_ppop_kolor + i*glebokoscPomiaru*szerokoscBskanu, szerokoscBskanu * sizeof(uchar4), szerokoscBskanu * sizeof(uchar4), glebokoscPomiaru, cudaMemcpyDeviceToDevice, streams[1]));

	});

	t3 = std::thread([&] {
		for (int i = 0; i < liczbaPrzekrojowPoziomych; ++i)
			HANDLE_ERROR(cudaMemcpy2DToArrayAsync(tabliceCuda[i + liczbaBskanow + liczbaPrzekrojowPoprzecznych], 0, 0, daneGPU_ppoz_kolor + i*glebokoscPomiaru*rozmiarAskanu, glebokoscPomiaru * sizeof(uchar4), glebokoscPomiaru * sizeof(uchar4), rozmiarAskanu, cudaMemcpyDeviceToDevice, streams[2]));

	});
		break;

	case WIZUALIZACJA::TYP_2D:
//		f1 = std::async(std::launch::async, [&] {
//			for (int i = 0; i < liczbaBskanow; ++i)
//				HANDLE_ERROR(cudaMemcpy2DToArrayAsync(tabliceCuda[i], 0, 0, daneGPU_bskan_kolor + i*rozmiarAskanu*szerokoscBskanu, szerokoscBskanu * sizeof(uchar4), szerokoscBskanu * sizeof(uchar4), rozmiarAskanu, cudaMemcpyDeviceToDevice, streams[0]));

//		});


		t1 = std::thread([&] {
			for (int i = 0; i < liczbaBskanow; ++i)
				HANDLE_ERROR(cudaMemcpy2DToArrayAsync(tabliceCuda[i], 0, 0, daneGPU_bskan_kolor + i*rozmiarAskanu*szerokoscBskanu, szerokoscBskanu * sizeof(uchar4), szerokoscBskanu * sizeof(uchar4), rozmiarAskanu, cudaMemcpyDeviceToDevice, streams[0]));

		});

		break;
	}
	
	t1.join();
	if (trybWyswietlania == WIZUALIZACJA::TYP_3D) {
		t2.join();
		t3.join();
	}
	;
	
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	printf("czas kopiowania cudaMemcpy2DtoArray: %f\n", j);
}


__global__ void odswiezaniePrzekroju(uchar4 *dane, const uchar4 *kolory,bool czyRGB) {

	if (threadIdx.x < blockDim.x) {

		unsigned char src = dane[blockIdx.x*blockDim.x + threadIdx.x].w;
		uchar4 tmp;
		if (czyRGB) {

			tmp = kolory[src];
			
		} else {

			tmp.x = kolory[src].w;
			tmp.y = kolory[src].w;
			tmp.z = kolory[src].w;

		}
		
		tmp.w = src;
		dane[blockIdx.x*blockDim.x + threadIdx.x] = tmp;
	}
}

void CudaTekstury::odswiez_bskany() {
	cudaFuncSetCacheConfig(odswiezaniePrzekroju, cudaFuncCachePreferL1);
	dim3 block(szerokoscBskanu);
	dim3 grid(rozmiarAskanu*liczbaBskanow);
	odswiezaniePrzekroju << <grid, block, 0, streams[0] >> > (daneGPU_bskan_kolor, d_mapaKolory_Szarosc,kolor);
	//odswiezaniePrzekroju << <grid, block>> > (daneGPU_bskan_kolor, d_mapaKolory_Szarosc,kolor);
}

void CudaTekstury::odswiez_przekrojePoprzeczne() {
	cudaFuncSetCacheConfig(odswiezaniePrzekroju, cudaFuncCachePreferL1);
	dim3 block(szerokoscBskanu);
	dim3 grid(glebokoscPomiaru*liczbaPrzekrojowPoprzecznych);
	odswiezaniePrzekroju << <grid, block, 0, streams[1] >> > (daneGPU_ppop_kolor, d_mapaKolory_Szarosc,kolor);
	//odswiezaniePrzekroju << <grid, block>> > (daneGPU_ppop_kolor, d_mapaKolory_Szarosc,kolor);
}

void CudaTekstury::odswiez_przekrojePoziome() {
	cudaFuncSetCacheConfig(odswiezaniePrzekroju, cudaFuncCachePreferL1);
	dim3 block(glebokoscPomiaru);
	dim3 grid(rozmiarAskanu*liczbaPrzekrojowPoziomych);
	odswiezaniePrzekroju << < grid, block, 0, streams[2] >> > (daneGPU_ppoz_kolor, d_mapaKolory_Szarosc,kolor);
	//odswiezaniePrzekroju << < grid, block>> > (daneGPU_ppoz_kolor, d_mapaKolory_Szarosc,kolor);
}


void CudaTekstury::pokolorujTeksturyIprzeslijDoTablicCuda() {

	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);

//	std::future<void> f1;
//	std::future<void> f2;
//	std::future<void> f3;

	std::thread t1;
	std::thread t2;
	std::thread t3;

	switch (trybWyswietlania) {

	case WIZUALIZACJA::TYP_3D:
	
//		f1 = std::async(std::launch::async, [&] {odswiez_bskany(); });
//		f2 = std::async(std::launch::async, [&] {odswiez_przekrojePoprzeczne(); });
//		f3 = std::async(std::launch::async, [&] {odswiez_przekrojePoziome(); });
	
		t1 = std::thread([&] {odswiez_bskany(); });
		t2 = std::thread([&] {odswiez_przekrojePoprzeczne(); });
		t3 = std::thread([&] {odswiez_przekrojePoziome(); });
		break;
	case WIZUALIZACJA::TYP_2D:
//		f1 = std::async(std::launch::async, [&] {odswiez_bskany(); });
		t1 = std::thread([&] {odswiez_bskany(); });
		break;
	}

	t1.join();
	if (trybWyswietlania == WIZUALIZACJA::TYP_3D) {
		t2.join();
		t3.join();
	}


	kopiowaniePrzekrojow();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	//	printf("czas odswiezania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}