#include"CudaTekstury.cuh"




void CudaTekstury::init() {

	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	HANDLE_ERROR(cudaGLSetGLDevice(dev));
	

	tabliceCuda = new cudaArray_t[liczbaPrzekrojow()];
	streams = new cudaStream_t[liczbaStrumieni];
	for (size_t i = 0; i < liczbaStrumieni; ++i)
		HANDLE_ERROR(cudaStreamCreateWithFlags(&streams[i], cudaStreamDefault));

	bskany = new cudaSurfaceObject_t[liczbaBskanow];
	przekrojePoprzeczne = new cudaSurfaceObject_t[liczbaPrzekrojowPoprzecznych];
	przekrojePoziome = new cudaSurfaceObject_t[liczbaPrzekrojowPoziomych];
	
	pobierzDefinicjeKolorow();
	ustawMapeKolorow();
//	HANDLE_ERROR(cudaMalloc((void**)&d_mapaSzarosci, 256 * sizeof(unsigned char)));
//	HANDLE_ERROR(cudaMemcpy((void**)d_mapaSzarosci, mapaSzarosci, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc(&d_mapaKolorow, 256 * sizeof(uchar3)));
	HANDLE_ERROR(cudaMemcpy(d_mapaKolorow, mapaKolorow, 256 * sizeof(uchar3), cudaMemcpyHostToDevice));
}

void CudaTekstury::pobierzDaneCPU() {

	MessageBox(NULL, "przed", "", MB_OK);
	HANDLE_ERROR(cudaMalloc(&daneGPU, calkowityRozmiarDanych() *sizeof(oct_t)));
	MessageBox(NULL, "miedzy", "", MB_OK);
	HANDLE_ERROR(cudaMemcpy(daneGPU, daneCPU, calkowityRozmiarDanych() * sizeof(oct_t), cudaMemcpyHostToDevice));
	MessageBox(NULL, "po", "", MB_OK);
//	delete[] daneCPU;
//	daneCPU = nullptr;
}

void CudaTekstury::wczytajBMP(char* plik) {
	
	int szer, wys;
	unsigned long *tmp = WczytajObrazZPlikuBitmap(NULL,plik, szer, wys, false, 255);
	daneCPU = new oct_t[wys*szer];
	memcpy(daneCPU, tmp, wys*szer * sizeof(oct_t));
}

void __global__ tworzenieBskanu(cudaSurfaceObject_t surf, const oct_t* source, size_t indeks, uchar3 *kolory) {

	if (threadIdx.x < blockDim.x) {
		
		//do wyswietlania napisu LAB
		indeks = 0;
		unsigned int value = (unsigned int)source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		

		/*
		//do1
		uchar4 data;
		if (blockIdx.x < gridDim.x / 3) {

			if (threadIdx.x < blockDim.x / 2) {

				data.x = 150;
				data.y = 0;
				data.z = 0;

			}
			else {

				data.x = 0;
				data.y = 150;
				data.z = 0;

			}
		}
		else {

			if (threadIdx.x < blockDim.x / 2) {

				data.x = 0;
				data.y = 0;
				data.z = 150;

			}
			else {

				data.x = 255;
				data.y = 255;
				data.z = 255;


			}

		}

		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		*/
	
		/*
		//do2 - najlepiej na kostce, oct_t = char
		uchar4 data;
		data.x = threadIdx.x / 2;
		data.y = 0;
		data.z = 0;
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		
		*/
		/*
		//do3
		unsigned char value = source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		uchar4 data;//x=R,y=G,z=B,w=A	
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		*/


	//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);

	}
}

void __global__ tworzeniePrzekrojuPoprzecznego(cudaSurfaceObject_t surf, const oct_t* source, size_t indeks, size_t rozmiarAskanu, uchar3 *kolory) {

	if (threadIdx.x < blockDim.x) {
		
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[indeks*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned int), blockIdx.x);
		
	
		/*
		//do1
		uchar4 data;
		if (blockIdx.x < gridDim.x / 3) {

			if (threadIdx.x < blockDim.x / 2) {

				data.x = 150;
				data.y = 0;
				data.z = 0;

			} else {

				data.x = 0;
				data.y = 150;
				data.z = 0;

			}
		}	else {

			if (threadIdx.x < blockDim.x / 2) {

				data.x = 0;
				data.y = 0;
				data.z = 150;

			} else {

				data.x = 255;
				data.y = 255;
				data.z = 255;
				
			}
		}
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		*/

		/*
		//do2
		uchar4 data;
		data.x = 0;
		data.y = threadIdx.x / 2;
		data.z = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		*/


		//unsigned char value = source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x];
		
		
		/*
		//do3
		unsigned char value = source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x];
		uchar4 data;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
	*/

	//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
	
	
	
		//surf2Dwrite(source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}

void __global__ tworzeniePrzekrojuPoziomego(cudaSurfaceObject_t surf, const oct_t* source, size_t indeks, size_t szerokoscBskanu, uchar3 *kolory) {

	if (threadIdx.x < blockDim.x) {
		
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[szerokoscBskanu * blockIdx.x + indeks];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		

		/*
		//do1
		uchar4 data;
		if (blockIdx.x < gridDim.x / 3) {

			if (threadIdx.x <= blockDim.x / 2) {

				data.x = 150;
				data.y = 0;
				data.z = 0;

			}	else {

				data.x = 0;
				data.y = 150;
				data.z = 0;

			}
		}	else {

			if (threadIdx.x <= blockDim.x / 2) {

				data.x = 0;
				data.y = 0;
				data.z = 150;

			}	else {

				data.x = 255;
				data.y = 255;
				data.z = 255;

			}
		}

		data.w = 0;

		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		*/

		/*
		//do2
		uchar4 data;
		data.x = 0;
		data.y = 0;
		data.z = threadIdx.x / 2;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		*/


		
	/*
		//do3
		unsigned char value = source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks];
		uchar4 data;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
	*/	


		//unsigned char value = source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks];
//		unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		//surf2Dwrite(source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}



void CudaTekstury::launch_bskany(size_t i) {

//	MessageBox(NULL, "bskany", "", MB_OK);
	dim3 grid(rozmiarAskanu);
	dim3 block(szerokoscBskanu);
	tworzenieBskanu << <grid, block, 0, streams[i] >> > (bskany[i], daneGPU, floor(i*krok_bskan),d_mapaKolorow);

}

void CudaTekstury::launch_przekrojePoprzeczne(size_t i) {

	dim3 grid(glebokoscPomiaru);
	dim3 block(szerokoscBskanu);
	tworzeniePrzekrojuPoprzecznego << <grid, block, 0, streams[i] >> > (przekrojePoprzeczne[i], daneGPU, floor(i*krok_przekrojePoprzeczne), rozmiarAskanu, d_mapaKolorow);
}

void CudaTekstury::launch_przekrojePoziome(size_t i) {

	dim3 grid(rozmiarAskanu);
	dim3 block(glebokoscPomiaru);
	tworzeniePrzekrojuPoziomego << < grid, block, 0, streams[i] >> > (przekrojePoziome[i], daneGPU, floor(i*krok_przekrojePoziome), szerokoscBskanu, d_mapaKolorow);
}

void CudaTekstury::tworzPrzekroje() {

	MessageBox(NULL, "tworzPrzekroje1", "", MB_OK);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);

	for (size_t i = 0; i < liczbaBskanow; ++i)
		launch_bskany(i);

	MessageBox(NULL, "tworzPrzekroje2", "", MB_OK);
	for (size_t i = 0; i < liczbaPrzekrojowPoprzecznych; ++i)
		launch_przekrojePoprzeczne(i);
	
	MessageBox(NULL, "tworzPrzekroje3", "", MB_OK);
	for (size_t i = 0; i < liczbaPrzekrojowPoziomych; ++i)
		launch_przekrojePoziome(i);

	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	printf("czas: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
	MessageBox(NULL, "Koniec tworzenie przekrojow", "", MB_OK);
	//trzeba to przemyœleæ: mo¿e nie ma sensu zwalniaæ pamiêci bo przy aktualizacji danych ponownie potrzebna bêdzie alokacja
	//delete[] daneGPU;
	//daneGPU = nullptr;
}


void CudaTekstury::przygotowanieTekstur() {

	for (size_t i = 0; i != liczbaBskanow; ++i) {

		tworzenie_tekstur(&tabliceCuda[i], &bskany[i]);
	}

	for (size_t i = 0; i != liczbaPrzekrojowPoprzecznych; ++i) {

		tworzenie_tekstur(&tabliceCuda[i+liczbaBskanow], &przekrojePoprzeczne[i]);

	}
	
	for (size_t i = 0; i != liczbaPrzekrojowPoziomych; ++i) {

		tworzenie_tekstur(&tabliceCuda[i+liczbaBskanow+liczbaPrzekrojowPoprzecznych], &przekrojePoziome[i]);
		//tworzenie_tekstur(&tabliceCuda[i], &przekrojePoziome[i]);
	}
	
}

void CudaTekstury::tworzenie_tekstur(cudaArray_t *tab, cudaSurfaceObject_t *surf) {

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = *tab;
	HANDLE_ERROR(cudaCreateSurfaceObject(surf, &resDesc));
}

void CudaTekstury::wczytajDane(const char* nazwaPliku) {

	std::ifstream source(nazwaPliku, std::ios::in | std::ios::binary);

	if (source.is_open()) {

		daneCPU = new oct_t[calkowityRozmiarDanych()];
		source.read((char*)daneCPU, calkowityRozmiarDanych() * sizeof(oct_t));

	}

	source.close();
}




void CudaTekstury::wprowadzTestoweDane() {

//	daneCPU = new oct_t[calkowityRozmiarDanych()];

	for (oct_t i = 0; i < calkowityRozmiarDanych(); ++i) {

		daneCPU[i] = i;

	}
}

void CudaTekstury::pobierzDefinicjeKolorow() {
	
	FILE *plik;
	errno_t err = fopen_s(&plik, "def.bin", "rb");
		if (err == 0) {
		for (int j = 0; j < 256; j++)
		fread(&defKol[j], 1, 3, plik);

			fclose(plik);
		}

		
/*	
	std::ofstream dane("def.bin", std::ios::in | std::ios::binary);

	if (dane.is_open()) {

		std::stringstream s;
		for (int j = 0; j < 256; j++) {
			std::copy_n(std::istreambuf_iterator<char>(dane.rdbuf()), 3, std::ostreambuf_iterator<char>(s));
			s.read((char*)defKol[j], 3);
		}
	}
	*/
}

void CudaTekstury::wczytajDaneBinarne(char *nazwaPliku) {

	std::ifstream dane(nazwaPliku, std::ios::in | std::ios::binary);

	if (dane.is_open()) {

		std::stringstream s;
		int tmp;
		dane.read((char*)&tmp, sizeof(tmp));
		dane.read((char*)&tmp, sizeof(tmp));
		dane.read((char*)&tmp, sizeof(tmp));
		daneCPU = new oct_t[calkowityRozmiarDanych()];
		dane.read((char*)daneCPU, calkowityRozmiarDanych() * sizeof(oct_t));

	}
	else {

		MessageBox(NULL, "Blad otwarcia pliku!", "", NULL);

	}

	dane.close();
}

void CudaTekstury::ustawMapeKolorow() {

	int progCzerni = jasnosc - (kontrast / 2);//do progu czerni kolor czarny
	
	for (int i = 0, end = (progCzerni<256)?progCzerni:256; i != end; ++i) {

		mapaSzarosci[i] = 0;
		mapaKolorow[i].x = 0;
		mapaKolorow[i].y = 0;
		mapaKolorow[i].z = 0;

	}
	//miedzy czarnym i bia³ym przedzia³ kolorow
	int progSzarosciKolorow = jasnosc + (kontrast / 2);
	//tutaj jest b³¹d: progCzerni moze byc wieksz niz 256 -> spojrz na warunek konca powy¿szej pêtli
	for (int i = progCzerni, przedzial = progSzarosciKolorow - progCzerni, end = (progSzarosciKolorow<256)?progSzarosciKolorow:256; i != end; ++i) {

		mapaSzarosci[i] = (unsigned char)((255 * ((float)i - progCzerni)) / przedzial);
		mapaKolorow[i].x = defKol[mapaSzarosci[i]][0];
		mapaKolorow[i].y = defKol[mapaSzarosci[i]][1];
		mapaKolorow[i].z = defKol[mapaSzarosci[i]][2];

	}
	//powyzej progu szarosci kolor bialy
	for (int i = progSzarosciKolorow; i != 256; ++i) {

		mapaSzarosci[i] = 255;
		mapaKolorow[i].x = 255;
		mapaKolorow[i].y = 255;
		mapaKolorow[i].z = 255;
	}
}