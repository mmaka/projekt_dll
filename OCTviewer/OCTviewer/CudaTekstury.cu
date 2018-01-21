#include"CudaTekstury.cuh"




void CudaTekstury::init() {

	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	HANDLE_ERROR(cudaGLSetGLDevice(dev));
	

//	tabliceCuda2 = new cudaArray_t[liczbaPrzekrojow()];
	streams = new cudaStream_t[liczbaStrumieni];
	for (size_t i = 0; i < liczbaStrumieni; ++i)
		HANDLE_ERROR(cudaStreamCreate(&streams[i]));
		//HANDLE_ERROR(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
		
//	streams.resize(liczbaStrumieni);
//	for (size_t i = 0; i < liczbaStrumieni; ++i)
//		HANDLE_ERROR(cudaStreamCreateWithFlags(&streams[i], cudaStreamDefault));

	//bskany = new cudaSurfaceObject_t[liczbaBskanow];
	//przekrojePoprzeczne = new cudaSurfaceObject_t[liczbaPrzekrojowPoprzecznych];
	//przekrojePoziome = new cudaSurfaceObject_t[liczbaPrzekrojowPoziomych];
	
	bskany.reserve(liczbaBskanow);
	przekrojePoprzeczne.reserve(liczbaPrzekrojowPoprzecznych);
	przekrojePoziome.reserve(liczbaPrzekrojowPoziomych);

	

	pobierzDefinicjeKolorow();
	ustawMapeKolorow2();
//	HANDLE_ERROR(cudaMalloc((void**)&d_mapaSzarosci, 256 * sizeof(unsigned char)));
//	HANDLE_ERROR(cudaMemcpy((void**)d_mapaSzarosci, mapaSzarosci, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice));

//	HANDLE_ERROR(cudaMalloc(&d_mapaKolorow, 256 * sizeof(uchar3)));
//	HANDLE_ERROR(cudaMemcpy(d_mapaKolorow, mapaKolorow, 256 * sizeof(uchar3), cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc(&d_mapaKolory_Szarosc, 256 * sizeof(uchar4)));
	HANDLE_ERROR(cudaMemcpy(d_mapaKolory_Szarosc, mapaKolorySzarosc, 256 * sizeof(uchar4), cudaMemcpyHostToDevice));

	inicjalizacja = true;
}

void CudaTekstury::pobierzDaneCPU() {
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
//	MessageBox(NULL, "przed", "", MB_OK);
	HANDLE_ERROR(cudaMalloc(&daneGPU, calkowityRozmiarDanych() *sizeof(oct_t)));

//	MessageBox(NULL, "miedzy", "", MB_OK);
	HANDLE_ERROR(cudaMemcpy(daneGPU, daneCPU, calkowityRozmiarDanych() * sizeof(oct_t), cudaMemcpyHostToDevice));
//	MessageBox(NULL, "po", "", MB_OK);
//	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
//	printf("czas kopiowania: %f\n", j);
	delete[] daneCPU;
	daneCPU = nullptr;
}



void CudaTekstury::wczytajBMP(char* plik) {
	
	int szer, wys;
	unsigned long *tmp = WczytajObrazZPlikuBitmap(NULL,plik, szer, wys, false, 255);
	daneCPU = new oct_t[wys*szer];
	memcpy(daneCPU, tmp, wys*szer * sizeof(oct_t));
}

__global__ void tworzenieBskanow_uchar4(uchar4 *dstGPU, const oct_t *source,size_t indeks,const uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {

		register int idx = indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
		unsigned char src_val = source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		uchar4 value = kolory[src_val];
		dstGPU[idx].x = value.x;
		dstGPU[idx].y = value.y;
		dstGPU[idx].z = value.z;
		dstGPU[idx].w = src_val;
	}
	

}

void CudaTekstury::launch_przygotowanie_bskanow(size_t i) {

	//	MessageBox(NULL, "bskany", "", MB_OK);

	//	dim3 grid(rozmiarAskanu);
	//	dim3 block(szerokoscBskanu);

	dim3 grid(szerokoscBskanu);
	dim3 block(rozmiarAskanu);

	tworzenieBskanow_uchar4 << <grid, block, 0, streams[0] >> > (daneGPU_tab, daneGPU, floor(i*krok_bskan), d_mapaKolory_Szarosc);
	//tworzenieBskanu << <grid, block >> > (bskany[i], daneGPU, floor(i*krok_bskan), d_mapaKolorow);

}


void __global__ kolorowanieBskanu2(cudaSurfaceObject_t surf,const uchar4 *kolory,int rozmiarAskanu,int szerokoscBskanu) {


	int wsp_x = blockIdx.x*blockDim.x + threadIdx.x;
	int wsp_y = blockIdx.y*blockDim.y + threadIdx.y;

	if (wsp_x <szerokoscBskanu && wsp_y<rozmiarAskanu) {
		/*
		//do wyswietlania napisu LAB
		indeks = 0;
		unsigned int value = (unsigned int)source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		*/

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

		//do3
		
		uchar4 data;//x=R,y=G,z=B,w=A	


		surf2Dread(&data, surf, wsp_x * sizeof(uchar4), wsp_y);
		//char val = data.w;
		unsigned char value = data.w;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		//data.w = value;
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, wsp_x * sizeof(uchar4), wsp_y);


		//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);

	}
}


void __global__ kolorowanieBskanu(cudaSurfaceObject_t surf, const uchar4 *kolory, int rozmiarAskanu, int szerokoscBskanu) {


	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		indeks = 0;
		unsigned int value = (unsigned int)source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		*/

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

		//do3

		uchar4 data;//x=R,y=G,z=B,w=A	


		surf2Dread(&data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		//char val = data.w;
		unsigned char value = data.w;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		//data.w = value;
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);


		//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);

	}
}



void __global__ kolorowaniePrzekrojuPoprzecznego2(cudaSurfaceObject_t surf, uchar4 *kolory,int szerokoscBskanu,int glebokoscPomiaru) {

	int wsp_x = blockIdx.x*blockDim.x + threadIdx.x;
	int wsp_y = blockIdx.y*blockDim.y + threadIdx.y;

	if (wsp_x < szerokoscBskanu && wsp_y < glebokoscPomiaru) {
		/*
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[indeks*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned int), blockIdx.x);
		*/

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



		//do3
		uchar4 data;//x=R,y=G,z=B,w=A	


		surf2Dread(&data, surf, wsp_x * sizeof(uchar4), wsp_y);
		//char val = data.w;
		unsigned char value = data.w;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		//data.w = value;
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, wsp_x * sizeof(uchar4), wsp_y);

		//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);



		//surf2Dwrite(source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}



void __global__ kolorowaniePrzekrojuPoprzecznego(cudaSurfaceObject_t surf, uchar4 *kolory, int szerokoscBskanu, int glebokoscPomiaru) {

	
	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[indeks*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned int), blockIdx.x);
		*/

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



		//do3
		uchar4 data;//x=R,y=G,z=B,w=A	


		surf2Dread(&data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		//char val = data.w;
		unsigned char value = data.w;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		//data.w = value;
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);

		//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);



		//surf2Dwrite(source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}



void __global__ kolorowaniePrzekrojuPoziomego2(cudaSurfaceObject_t surf, uchar4 *kolory, int glebokoscPomiaru,int rozmiarAskanu) {

	int wsp_x = blockIdx.x*blockDim.x + threadIdx.x;
	int wsp_y = blockIdx.y*blockDim.y + threadIdx.y;

	if (wsp_x < glebokoscPomiaru && wsp_y < rozmiarAskanu) {
		/*
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[szerokoscBskanu * blockIdx.x + indeks];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		*/

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




		//do3
		uchar4 data;//x=R,y=G,z=B,w=A	


		surf2Dread(&data, surf, wsp_x * sizeof(uchar4), wsp_y);
		//char val = data.w;
		unsigned char value = data.w;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		//data.w = value;
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, wsp_x * sizeof(uchar4), wsp_y);

		//unsigned char value = source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks];
		//		unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		//surf2Dwrite(source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}



void __global__ kolorowaniePrzekrojuPoziomego(cudaSurfaceObject_t surf, uchar4 *kolory, int glebokoscPomiaru, int rozmiarAskanu) {

	
	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[szerokoscBskanu * blockIdx.x + indeks];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		*/

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




		//do3
		uchar4 data;//x=R,y=G,z=B,w=A	


		surf2Dread(&data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		//char val = data.w;
		unsigned char value = data.w;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		//data.w = value;
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);

		//unsigned char value = source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks];
		//		unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		//surf2Dwrite(source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}




void __global__ tworzenieBskanu_bezkoloru(cudaSurfaceObject_t surf, const uchar4* source, size_t indeks) {

	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		indeks = 0;
		unsigned int value = (unsigned int)source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		*/

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
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		*/

		/*
		//do2 - najlepiej na kostce, oct_t = char
		uchar4 data;
		data.x = threadIdx.x / 2;
		data.y = 0;
		data.z = 0;
		data.w = 0;
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		*/


		//do3
		uchar4 value = source[blockIdx.x*blockDim.x + threadIdx.x];
		
		surf2Dwrite(value, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		//surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);


		//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);

	}
}



void __global__ tworzenieBskanu(cudaSurfaceObject_t surf, const oct_t* source, size_t indeks, const uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		indeks = 0;
		unsigned int value = (unsigned int)source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		*/

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
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		*/
	
		/*
		//do2 - najlepiej na kostce, oct_t = char
		uchar4 data;
		data.x = threadIdx.x / 2;
		data.y = 0;
		data.z = 0;
		data.w = 0;
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		*/
		
		
		//do3
		unsigned char value = source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		uchar4 data;//x=R,y=G,z=B,w=A	
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		data.w = value;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		//surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		

	//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);

	}
}

void __global__ tworzeniePrzekrojuPoprzecznego(cudaSurfaceObject_t surf, const oct_t* source, size_t indeks, size_t rozmiarAskanu, const uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[indeks*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned int), blockIdx.x);
		*/
	
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
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		*/

		/*
		//do2
		uchar4 data;
		data.x = 0;
		data.y = threadIdx.x / 2;
		data.z = 0;
	//	surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		*/

		//unsigned char value = source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x];
		
		
		
		//do3
		unsigned char value = source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x];
		
		uchar4 data;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		//surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
			

	//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
	
	
	
		//surf2Dwrite(source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}

void __global__ tworzeniePrzekrojuPoziomego(cudaSurfaceObject_t surf, const oct_t* source, size_t indeks, size_t szerokoscBskanu, const uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[szerokoscBskanu * blockIdx.x + indeks];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		*/

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

		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		*/
		/*
		//do2
		uchar4 data;
		data.x = 0;
		data.y = 0;
		data.z = threadIdx.x / 2;
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		*/


		
	
		//do3
		unsigned char value = source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks];
		
		uchar4 data;
		data.x = kolory[value].x;
		data.y = kolory[value].y;
		data.z = kolory[value].z;
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		//surf2Dwrite(data, surf, blockIdx.x * sizeof(uchar4), threadIdx.x);
		


		//unsigned char value = source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks];
//		unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		//surf2Dwrite(source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}


void __global__ tworzenieBskanu_skalaSzarosci(cudaSurfaceObject_t surf, const oct_t* source, size_t indeks, uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		indeks = 0;
		unsigned int value = (unsigned int)source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		*/

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

		//do3
		unsigned char value = source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		uchar4 data;//x=R,y=G,z=B,w=A	
		data.x = kolory[value].x;
		data.y = kolory[value].x;
		data.z = kolory[value].x;
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);



		//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);

	}
}

void __global__ tworzeniePrzekrojuPoprzecznego_skalaSzarosci(cudaSurfaceObject_t surf, const oct_t* source, size_t indeks, size_t rozmiarAskanu, uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[indeks*blockDim.x + threadIdx.x];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned int), blockIdx.x);
		*/

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



		//do3
		unsigned char value = source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x];

		uchar4 data;
		data.x = kolory[value].x;
		data.y = kolory[value].x;
		data.z = kolory[value].x;
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);


		//	unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);



		//surf2Dwrite(source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}

void __global__ tworzeniePrzekrojuPoziomego_skalaSzarosci(cudaSurfaceObject_t surf, const oct_t* source, size_t indeks, size_t szerokoscBskanu, uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {
		/*
		//do wyswietlania napisu LAB
		unsigned int value = (unsigned int)source[szerokoscBskanu * blockIdx.x + indeks];
		surf2Dwrite(value, surf, threadIdx.x * sizeof(unsigned long), blockIdx.x);
		*/

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




		//do3
		unsigned char value = source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks];

		uchar4 data;
		data.x = kolory[value].x;
		data.y = kolory[value].x;
		data.z = kolory[value].x;
		data.w = 0;
		surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);



		//unsigned char value = source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks];
		//		unsigned int tmp = (data.w << 24) + (data.z << 16) + (data.y << 8) + (data.x);
		//surf2Dwrite(data, surf, threadIdx.x * sizeof(uchar4), blockIdx.x);
		//surf2Dwrite(source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
		//surf2Dwrite(mapaSzarosci[source[threadIdx.x*szerokoscBskanu*gridDim.x + szerokoscBskanu * blockIdx.x + indeks]], surf, threadIdx.x * sizeof(oct_t), blockIdx.x);
	}
}




void __global__ tworzenieBskanu_tab(oct_t* tab, const oct_t* source, size_t indeks, uchar3 *kolory) {

	register int idx = (blockIdx.x*blockDim.x + threadIdx.x)*4;
	uchar3 value = kolory[source[indeks*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]];
	tab[idx] = 120;// value.x;
	tab[idx+1] = 120;// value.y;
	tab[idx+2] = 120;// value.z;
	tab[idx+3] = 120;

}

void __global__ tworzeniePrzekrojuPoprzecznego_tab(oct_t* tab, const oct_t*  source, size_t indeks, size_t rozmiarAskanu, uchar3 *kolory) {

	tab[blockIdx.x*blockDim.x + threadIdx.x] = source[blockIdx.x*rozmiarAskanu*blockDim.x + indeks*blockDim.x + threadIdx.x];

}

void __global__ tworzeniePrzekrojuPoziomego_tab(oct_t*  tab, const oct_t* source, size_t indeks, size_t szerokoscBskanu, uchar3 *kolory) {

	tab[blockIdx.x + gridDim.x * threadIdx.x] = source[blockIdx.x*szerokoscBskanu*blockDim.x + szerokoscBskanu * threadIdx.x + indeks];

}



void CudaTekstury::launch_bskany2() {

	//	MessageBox(NULL, "bskany", "", MB_OK);

	//	dim3 grid(rozmiarAskanu);
	//	dim3 block(szerokoscBskanu);

	dim3 grid(szerokoscBskanu);
	dim3 block(rozmiarAskanu);

//	dim3 block(32, 32);
//	dim3 grid((-1 + szerokoscBskanu) / 32 + 1,(-1 + rozmiarAskanu) / 32 + 1 );

	for(size_t i=0;i<liczbaBskanow;++i)
		kolorowanieBskanu << <grid, block, 0, streams[0] >> > (bskany[i], d_mapaKolory_Szarosc,rozmiarAskanu,szerokoscBskanu);
	//tworzenieBskanu << <grid, block >> > (bskany[i], daneGPU, floor(i*krok_bskan), d_mapaKolorow);

}


void CudaTekstury::launch_przekrojePoprzeczne2() {

	//	dim3 grid(glebokoscPomiaru);
	//	dim3 block(szerokoscBskanu);

	dim3 grid(szerokoscBskanu);
	dim3 block(glebokoscPomiaru);

//	dim3 block(32, 32);
//	dim3 grid((-1 + szerokoscBskanu) / 32 + 1, (-1 + glebokoscPomiaru) / 32 + 1);

	for(size_t i = 0;i<liczbaPrzekrojowPoprzecznych;++i)
		kolorowaniePrzekrojuPoprzecznego << <grid, block, 0, streams[1] >> > (przekrojePoprzeczne[i], d_mapaKolory_Szarosc,szerokoscBskanu,glebokoscPomiaru);
	//tworzeniePrzekrojuPoprzecznego << <grid, block>> > (przekrojePoprzeczne[i], daneGPU, floor(i*krok_przekrojePoprzeczne), rozmiarAskanu, d_mapaKolorow);
}

void CudaTekstury::launch_przekrojePoziome2() {

	//	dim3 grid(rozmiarAskanu);
	//	dim3 block(glebokoscPomiaru);

	dim3 grid(glebokoscPomiaru);
	dim3 block(rozmiarAskanu);

//	dim3 block(32, 32);
//	dim3 grid((-1 + glebokoscPomiaru) / 32 + 1, (-1 + rozmiarAskanu) / 32 + 1);

	for(size_t i=0;i<liczbaPrzekrojowPoziomych;++i)
		kolorowaniePrzekrojuPoziomego << < grid, block, 0, streams[2] >> > (przekrojePoziome[i], d_mapaKolory_Szarosc,glebokoscPomiaru,rozmiarAskanu);
	//tworzeniePrzekrojuPoziomego << < grid, block>> > (przekrojePoziome[i], daneGPU, floor(i*krok_przekrojePoziome), szerokoscBskanu, d_mapaKolorow);
}


void CudaTekstury::launch_bskany_bezkoloru(size_t i) {

	//	MessageBox(NULL, "bskany", "", MB_OK);

	dim3 grid(rozmiarAskanu);
	dim3 block(szerokoscBskanu);

	//	dim3 grid(szerokoscBskanu);
	//	dim3 block(rozmiarAskanu);

	tworzenieBskanu_bezkoloru << <grid, block, 0, streams[0] >> > (bskany[i], daneGPU_tab+i*rozmiarAskanu*szerokoscBskanu, i);
	//tworzenieBskanu << <grid, block >> > (bskany[i], daneGPU, floor(i*krok_bskan), d_mapaKolorow);

}


void CudaTekstury::launch_bskany(size_t i) {

//	MessageBox(NULL, "bskany", "", MB_OK);
	
	dim3 grid(rozmiarAskanu);
	dim3 block(szerokoscBskanu);

//	dim3 grid(szerokoscBskanu);
//	dim3 block(rozmiarAskanu);

	tworzenieBskanu << <grid, block, 0, streams[0] >> > (bskany[i], daneGPU, floor(i*krok_bskan),d_mapaKolory_Szarosc);
	//tworzenieBskanu << <grid, block >> > (bskany[i], daneGPU, floor(i*krok_bskan), d_mapaKolorow);

}

void CudaTekstury::launch_przekrojePoprzeczne(size_t i) {

	dim3 grid(glebokoscPomiaru);
	dim3 block(szerokoscBskanu);

//	dim3 grid(szerokoscBskanu);
//	dim3 block(glebokoscPomiaru);

	tworzeniePrzekrojuPoprzecznego << <grid, block, 0, streams[1] >> > (przekrojePoprzeczne[i], daneGPU, floor(i*krok_przekrojePoprzeczne), rozmiarAskanu, d_mapaKolory_Szarosc);
	//tworzeniePrzekrojuPoprzecznego << <grid, block>> > (przekrojePoprzeczne[i], daneGPU, floor(i*krok_przekrojePoprzeczne), rozmiarAskanu, d_mapaKolorow);
}

void CudaTekstury::launch_przekrojePoziome(size_t i) {

	dim3 grid(rozmiarAskanu);
	dim3 block(glebokoscPomiaru);

//	dim3 grid(glebokoscPomiaru);
//	dim3 block(rozmiarAskanu);

	tworzeniePrzekrojuPoziomego << < grid, block, 0, streams[2] >> > (przekrojePoziome[i], daneGPU, floor(i*krok_przekrojePoziome), szerokoscBskanu, d_mapaKolory_Szarosc);
	//tworzeniePrzekrojuPoziomego << < grid, block>> > (przekrojePoziome[i], daneGPU, floor(i*krok_przekrojePoziome), szerokoscBskanu, d_mapaKolorow);
}

void CudaTekstury::tworzDane() {

	for (size_t i = 0; i < liczbaBskanow; ++i)
		launch_przygotowanie_bskanow(i);
		
		
	
		//launch_bskany_tab(i);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernell launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}



void CudaTekstury::tworzPrzekroje() {

	//	MessageBox(NULL, "tworzPrzekroje1", "", MB_OK);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	
//	auto f1 = std::async(std::launch::async, [&] {
//	for (size_t i = 0; i < liczbaBskanow; ++i)
//		launch_bskany(i);
//	});

//	auto f2 = std::async(std::launch::async, [&] {
//	for (size_t i = 0; i < liczbaPrzekrojowPoprzecznych; ++i)
//		launch_przekrojePoprzeczne(i);
//	});

//	auto f3 = std::async(std::launch::async, [&] {
//	for (size_t i = 0; i < liczbaPrzekrojowPoziomych; ++i)
//		launch_przekrojePoziome(i);
//		});
		

	for (size_t i = 0; i < liczbaBskanow; ++i)
			launch_bskany_bezkoloru(i);

//	for (size_t i = 0; i < liczbaPrzekrojowPoprzecznych; ++i)
//			launch_przekrojePoprzeczne(i);

//	for (size_t i = 0; i < liczbaPrzekrojowPoziomych; ++i)
//			launch_przekrojePoziome(i);

	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	printf("czas: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernell launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
//	HANDLE_ERROR(cudaFree(daneGPU));
//	MessageBox(NULL, "Koniec tworzenie przekrojow", "", MB_OK);
	//trzeba to przemyœleæ: mo¿e nie ma sensu zwalniaæ pamiêci bo przy aktualizacji danych ponownie potrzebna bêdzie alokacja
	//delete[] daneGPU;
	//daneGPU = nullptr;
}

void CudaTekstury::tworzenie_tekstur(cudaArray_t *tab, cudaSurfaceObject_t *surf) {

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	//	HANDLE_ERROR(cudaMemcpyToArray(*tab, 0, 0, daneGPU_tab+i*szerokoscBskanu*rozmiarAskanu, 4*sizeof(oct_t)*szerokoscBskanu*rozmiarAskanu, cudaMemcpyDeviceToDevice));
	resDesc.res.array.array = *tab;
	HANDLE_ERROR(cudaCreateSurfaceObject(surf, &resDesc));
}

void CudaTekstury::przygotowanieTekstur() {

	bskany.resize(liczbaBskanow);
	for (size_t i = 0; i != liczbaBskanow; ++i) {

		tworzenie_tekstur(&tabliceCuda[i], &bskany[i]);
	}

	przekrojePoprzeczne.resize(liczbaPrzekrojowPoprzecznych);
	for (size_t i = 0; i != liczbaPrzekrojowPoprzecznych; ++i) {

		tworzenie_tekstur(&tabliceCuda[i+liczbaBskanow], &przekrojePoprzeczne[i]);

	}
	
	przekrojePoziome.resize(liczbaPrzekrojowPoziomych);
	for (size_t i = 0; i != liczbaPrzekrojowPoziomych; ++i) {

		tworzenie_tekstur(&tabliceCuda[i+liczbaBskanow+liczbaPrzekrojowPoprzecznych], &przekrojePoziome[i]);

	}
	
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


void CudaTekstury::ustawMapeKolorow2() {

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

void CudaTekstury::odswiezTekstury() {

	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);

//	std::async(std::launch::async, [&] {launch_bskany2(); });
//	std::async(std::launch::async, [&] {launch_przekrojePoprzeczne2(); });
//	std::async(std::launch::async, [&] {launch_przekrojePoziome2(); });

	
	std::thread t1([&] {launch_bskany2(); });
	std::thread t2([&] {launch_przekrojePoprzeczne2(); });
	std::thread t3([&] {launch_przekrojePoziome2(); });

	t1.join();
	t2.join();
	t3.join();
	
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	printf("czas: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}

void CudaTekstury::edycjaMapyKolorow(EDYCJA_MAPY_KOLOROW tryb, int value) {

	static bool poprawneDane = false;

	switch (tryb)
	{
	case EDYCJA_MAPY_KOLOROW::ZWIEKSZ_KONTRAST:
	{
		int tmp = kontrast - value;

		if (tmp >= 0) {

			kontrast = tmp;
			poprawneDane = true;

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
			poprawneDane = true;

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
			poprawneDane = true;

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
			poprawneDane = true;

		}
		else {

			jasnosc = 256;
		}
	}
		break;
	default:
		break;
	}

	if (poprawneDane) {

		ustawMapeKolorow2();
		LARGE_INTEGER countPerSec, tim1, tim2;
		QueryPerformanceFrequency(&countPerSec);
		QueryPerformanceCounter(&tim1);
		HANDLE_ERROR(cudaMemcpy(d_mapaKolory_Szarosc, mapaKolorySzarosc, 256 * sizeof(uchar4), cudaMemcpyHostToDevice));
		QueryPerformanceCounter(&tim2);
		double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
		printf("czas kopiowania mapy: %f\n", j);
		odswiezTekstury();
		poprawneDane = false;
	}
	
}


void CudaTekstury::sprzatanie() {

	if (inicjalizacja) {


		for (size_t i = 0; i < liczbaStrumieni; ++i) HANDLE_ERROR(cudaStreamDestroy(streams[i]));

	//	HANDLE_ERROR(cudaFree(daneGPU));
		HANDLE_ERROR(cudaFree(d_mapaKolory_Szarosc));
		
		delete[] streams;
	}

	
	//if (daneGPU != nullptr) delete[] daneGPU;
	if (daneCPU != nullptr) delete[] daneCPU;


}



void CudaTekstury::pobierzDaneCPU2() {

	HANDLE_ERROR(cudaMalloc(&daneGPU, liczbaBskanow*rozmiarAskanu*szerokoscBskanu * sizeof(oct_t)));
	HANDLE_ERROR(cudaMalloc(&daneGPU_tab, liczbaBskanow*rozmiarAskanu*szerokoscBskanu * sizeof(uchar4)));
	HANDLE_ERROR(cudaMalloc(&daneGPU_ppop, szerokoscBskanu*liczbaBskanow*liczbaPrzekrojowPoprzecznych * sizeof(uchar4)));
	HANDLE_ERROR(cudaMalloc(&daneGPU_ppoz, rozmiarAskanu*liczbaBskanow*liczbaPrzekrojowPoziomych * sizeof(uchar4)));
	for (int i = 0; i < liczbaBskanow; ++i) {
		int idx = ((int)floor(i*krok_bskan))*rozmiarAskanu*szerokoscBskanu;
		HANDLE_ERROR(cudaMemcpy(daneGPU + i*rozmiarAskanu*szerokoscBskanu, daneCPU +idx , rozmiarAskanu*szerokoscBskanu * sizeof(oct_t), cudaMemcpyHostToDevice));
	}
		


}

__global__ void kolorowanie(uchar4 *dstGPU, const oct_t *source, const uchar4 *kolory) {

	

//	__shared__ uchar4 kolor[256];
	
//	uchar4 data;
//	data.x = 255;
//	if (threadIdx.x < 256)
//		kolor[threadIdx.x] = kolory[threadIdx.x];

//	__syncthreads();

	if (threadIdx.x < blockDim.x) {

	
	//	register int idx = blockIdx.x*blockDim.x + threadIdx.x;
		

		//register int wsp_x = blockIdx.x*blockDim.x + threadIdx.x;
		//register int wsp_y = blockIdx.y*blockDim.y + threadIdx.y;
		//register int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//	unsigned char src_val = source[blockIdx.x*blockDim.x + threadIdx.x];
		dstGPU[blockIdx.x*blockDim.x + threadIdx.x] = kolory[(unsigned char)source[blockIdx.x*blockDim.x + threadIdx.x]];
		dstGPU[blockIdx.x*blockDim.x + threadIdx.x].w = (unsigned char)source[blockIdx.x*blockDim.x + threadIdx.x];
		//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].x = value.x;
		//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].y = value.y;
		//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].z = value.z;
		//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].w = 0;
	}
	
	/*
	if (threadIdx.x < blockDim.x && threadIdx.y < blockDim.y) {

		register int wsp_x = blockIdx.x*blockDim.x + threadIdx.x;
		register int wsp_y = blockIdx.y*blockDim.y + threadIdx.y;
		unsigned char src_val = source[wsp_y*blockDim.x+wsp_x];
		uchar4 value = kolory[src_val];
		dstGPU[wsp_y*blockDim.x + wsp_x].x =  value.x;
		dstGPU[wsp_y*blockDim.x + wsp_x].y =  value.y;
		dstGPU[wsp_y*blockDim.x + wsp_x].z = value.z;
		dstGPU[wsp_y*blockDim.x + wsp_x].w = 0;
	}
	*/
}

__global__ void przepisanieDoPrzekrojow(const uchar4 *dstGPU, uchar4 *dstGPU_ppop, uchar4 *dstGPU_ppoz,size krokPop,size krokPoz) {

	if (threadIdx.x < blockDim.x) {

		//register int wsp_x = blockIdx.x*blockDim.x + threadIdx.x;
		//register int wsp_y = blockIdx.y*blockDim.y + threadIdx.y;
		register int nrB = blockIdx.y*gridDim.x*blockDim.x;
		register int nrKol = threadIdx.x;
		register int nrW = blockIdx.x*blockDim.x;
		uchar4 dane;
		dane.x = 255;
		dane.y = 0;
		dane.z = 0;
		dane.w = 0;
		register int nrPpop = blockIdx.x*krokPop;
		register int nrPpoz = nrKol*krokPoz;
		dstGPU_ppop[blockIdx.x*blockDim.x*gridDim.y + blockIdx.y*blockDim.x + threadIdx.x] = dstGPU[nrB + nrW + nrKol];
		if (threadIdx.x == blockDim.x-2)
			dstGPU_ppoz[threadIdx.x*gridDim.y*gridDim.x+blockIdx.x*gridDim.y+blockIdx.y] = dane ;//dstGPU[nrB + nrW + nrKol];
		//dstGPU_ppoz[threadIdx.x*gridDim.y*gridDim.x + blockIdx.x*gridDim.y + blockIdx.y] = dstGPU[nrB + nrW + nrKol];
		
		
		
		//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].x = value.x;
		//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].y = value.y;
		//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].z = value.z;
		//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].w = 0;
	}

	/*
	if (threadIdx.x < blockDim.x && threadIdx.y < blockDim.y) {

	register int wsp_x = blockIdx.x*blockDim.x + threadIdx.x;
	register int wsp_y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned char src_val = source[wsp_y*blockDim.x+wsp_x];
	uchar4 value = kolory[src_val];
	dstGPU[wsp_y*blockDim.x + wsp_x].x =  value.x;
	dstGPU[wsp_y*blockDim.x + wsp_x].y =  value.y;
	dstGPU[wsp_y*blockDim.x + wsp_x].z = value.z;
	dstGPU[wsp_y*blockDim.x + wsp_x].w = 0;
	}
	*/
}


__global__ void przepisanieDoPrzekrojow2(const uchar4 *dstGPU, uchar4 * dstGPU_ppop, uchar4 * dstGPU_ppoz) {

	if (threadIdx.x < blockDim.x) {

		//register int wsp_x = blockIdx.x*blockDim.x + threadIdx.x;
		//register int wsp_y = blockIdx.y*blockDim.y + threadIdx.y;
	//	register int nrB = blockIdx.y*gridDim.x*blockDim.x;
	//	register int nrKol = threadIdx.x;
	//	register int nrW = blockIdx.x*blockDim.x;
		//uchar4 dane = dstGPU[nrB + nrW + nrKol];
		//register int nrPpop = blockIdx.x*krokPop;
		//register int nrPpoz = nrKol*krokPoz;
		//register uchar4 data = dstGPU[blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x];
		dstGPU_ppop[blockIdx.x*blockDim.x*gridDim.y + blockIdx.y*blockDim.x + threadIdx.x] = dstGPU[blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]; //dstGPU[nrB + nrW + nrKol];
		dstGPU_ppoz[threadIdx.x*gridDim.y*gridDim.x + blockIdx.x*gridDim.y + blockIdx.y] = dstGPU[blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x]; //dstGPU[nrB + nrW + nrKol];
																									//dstGPU_ppoz[threadIdx.x*gridDim.y*gridDim.x + blockIdx.x*gridDim.y + blockIdx.y] = dstGPU[nrB + nrW + nrKol];



																									//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].x = value.x;
																									//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].y = value.y;
																									//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].z = value.z;
																									//dstGPU[blockIdx.x*blockDim.x + threadIdx.x].w = 0;
	}

	/*
	if (threadIdx.x < blockDim.x && threadIdx.y < blockDim.y) {

	register int wsp_x = blockIdx.x*blockDim.x + threadIdx.x;
	register int wsp_y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned char src_val = source[wsp_y*blockDim.x+wsp_x];
	uchar4 value = kolory[src_val];
	dstGPU[wsp_y*blockDim.x + wsp_x].x =  value.x;
	dstGPU[wsp_y*blockDim.x + wsp_x].y =  value.y;
	dstGPU[wsp_y*blockDim.x + wsp_x].z = value.z;
	dstGPU[wsp_y*blockDim.x + wsp_x].w = 0;
	}
	*/
}




void CudaTekstury::kolorowanieB() {

//	dim3 block(32, 32);
//	dim3 grid((-1 + szerokoscBskanu*liczbaBskanow) / 32 + 1, (-1 + rozmiarAskanu*liczbaBskanow) / 32 + 1);


	HANDLE_ERROR(cudaMalloc(&daneGPU_tab, calkowityRozmiarDanych() * sizeof(uchar4)));
	

	dim3 block(szerokoscBskanu);
	dim3 grid(rozmiarAskanu*glebokoscPomiaru);
	cudaFuncSetCacheConfig(kolorowanie,cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	kolorowanie<<<grid,block>>>(daneGPU_tab,daneGPU, d_mapaKolory_Szarosc);
//	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
//	printf("czas kolorowania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernelll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
	HANDLE_ERROR(cudaFree(daneGPU));
}

void CudaTekstury::przepisanie() {


	HANDLE_ERROR(cudaMalloc(&daneGPU_ppop, szerokoscBskanu*glebokoscPomiaru*rozmiarAskanu * sizeof(uchar4)));
	HANDLE_ERROR(cudaMalloc(&daneGPU_ppoz, rozmiarAskanu*glebokoscPomiaru*szerokoscBskanu * sizeof(uchar4)));

	dim3 block(szerokoscBskanu);
	dim3 grid(rozmiarAskanu, glebokoscPomiaru);
	cudaFuncSetCacheConfig(przepisanieDoPrzekrojow2, cudaFuncCachePreferL1);
	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	przepisanieDoPrzekrojow2 << <grid, block >> > (daneGPU_tab, daneGPU_ppop, daneGPU_ppoz);
//	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
//	printf("czas przepisania: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernellll launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}

}



void CudaTekstury::kopiowanieB() {

	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	
	std::thread t1([&]{
		for (int i = 0; i < liczbaBskanow; ++i) {

			int idx = floor(i*krok_bskan);
			//HANDLE_ERROR(cudaMemcpyToArray(tabliceCuda2[i],0,0,daneGPU_tab+i*rozmiarAskanu*szerokoscBskanu,rozmiarAskanu*szerokoscBskanu*sizeof(uchar4),cudaMemcpyDeviceToDevice));
			//HANDLE_ERROR(cudaMemcpyToArray(tabA[i], 0, 0, daneGPU_tab + i*rozmiarAskanu*szerokoscBskanu, rozmiarAskanu*szerokoscBskanu * sizeof(uchar4), cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemcpy2DToArray(tabliceCuda[i], 0, 0, daneGPU_tab + idx*rozmiarAskanu*szerokoscBskanu, szerokoscBskanu * sizeof(char) * 4, szerokoscBskanu * sizeof(char) * 4, rozmiarAskanu, cudaMemcpyDeviceToDevice));

		}
	});

	std::thread t2([&]{

		for (int i = 0; i < liczbaPrzekrojowPoprzecznych; ++i) {

			int idx = floor(i*krok_przekrojePoprzeczne);
			//HANDLE_ERROR(cudaMemcpyToArray(tabliceCuda2[i],0,0,daneGPU_tab+i*rozmiarAskanu*szerokoscBskanu,rozmiarAskanu*szerokoscBskanu*sizeof(uchar4),cudaMemcpyDeviceToDevice));
			//HANDLE_ERROR(cudaMemcpyToArray(tabA[i], 0, 0, daneGPU_tab + i*rozmiarAskanu*szerokoscBskanu, rozmiarAskanu*szerokoscBskanu * sizeof(uchar4), cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemcpy2DToArray(tabliceCuda[i + liczbaBskanow], 0, 0, daneGPU_ppop + idx*glebokoscPomiaru*szerokoscBskanu, szerokoscBskanu * sizeof(char) * 4, szerokoscBskanu * sizeof(char) * 4, glebokoscPomiaru, cudaMemcpyDeviceToDevice));

		}
	});

	std::thread t3([&]{

		for (int i = 0; i < liczbaPrzekrojowPoziomych; ++i) {

			int idx = floor(i*krok_przekrojePoziome);
			//HANDLE_ERROR(cudaMemcpyToArray(tabliceCuda2[i],0,0,daneGPU_tab+i*rozmiarAskanu*szerokoscBskanu,rozmiarAskanu*szerokoscBskanu*sizeof(uchar4),cudaMemcpyDeviceToDevice));
			//HANDLE_ERROR(cudaMemcpyToArray(tabA[i], 0, 0, daneGPU_tab + i*rozmiarAskanu*szerokoscBskanu, rozmiarAskanu*szerokoscBskanu * sizeof(uchar4), cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemcpy2DToArray(tabliceCuda[i + liczbaBskanow + liczbaPrzekrojowPoprzecznych], 0, 0, daneGPU_ppoz + idx*glebokoscPomiaru*rozmiarAskanu, glebokoscPomiaru * sizeof(char) * 4, glebokoscPomiaru * sizeof(char) * 4, rozmiarAskanu, cudaMemcpyDeviceToDevice));

		}
	});

	t1.join();
	t2.join();
	t3.join();
//	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
//	printf("czas kopiowania cudaMemcpy2DtoArray: %f\n", j);
}


void CudaTekstury::kopiowanieP() {

	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);
	for (int i = 0; i < liczbaPrzekrojowPoprzecznych; ++i) {

		int idx = floor(i*krok_przekrojePoprzeczne);
		//HANDLE_ERROR(cudaMemcpyToArray(tabliceCuda2[i],0,0,daneGPU_tab+i*rozmiarAskanu*szerokoscBskanu,rozmiarAskanu*szerokoscBskanu*sizeof(uchar4),cudaMemcpyDeviceToDevice));
		//HANDLE_ERROR(cudaMemcpyToArray(tabA[i], 0, 0, daneGPU_tab + i*rozmiarAskanu*szerokoscBskanu, rozmiarAskanu*szerokoscBskanu * sizeof(uchar4), cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMemcpy2DToArray(tabliceCuda[i+liczbaBskanow], 0, 0, daneGPU_ppop + idx*glebokoscPomiaru*szerokoscBskanu, szerokoscBskanu * sizeof(char) * 4, szerokoscBskanu * sizeof(char) * 4, glebokoscPomiaru, cudaMemcpyDeviceToDevice));


	}
	for (int i = 0; i < liczbaPrzekrojowPoziomych; ++i) {

		int idx = floor(i*krok_przekrojePoziome);
		//HANDLE_ERROR(cudaMemcpyToArray(tabliceCuda2[i],0,0,daneGPU_tab+i*rozmiarAskanu*szerokoscBskanu,rozmiarAskanu*szerokoscBskanu*sizeof(uchar4),cudaMemcpyDeviceToDevice));
		//HANDLE_ERROR(cudaMemcpyToArray(tabA[i], 0, 0, daneGPU_tab + i*rozmiarAskanu*szerokoscBskanu, rozmiarAskanu*szerokoscBskanu * sizeof(uchar4), cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMemcpy2DToArray(tabliceCuda[i + liczbaBskanow+liczbaPrzekrojowPoprzecznych], 0, 0, daneGPU_ppoz + idx*glebokoscPomiaru*rozmiarAskanu, glebokoscPomiaru * sizeof(char) * 4, glebokoscPomiaru * sizeof(char) * 4, rozmiarAskanu, cudaMemcpyDeviceToDevice));


	}
//	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
//	printf("czas kopiowania 2 cudaMemcpy2DtoArray: %f\n", j);
}


__global__ void odswiezaniePrzekroju(uchar4 *dane, const uchar4 *kolory) {

	if (threadIdx.x < blockDim.x) {

		unsigned char src = dane[blockIdx.x*blockDim.x + threadIdx.x].w;
		uchar4 nowy = kolory[src];
		nowy.w = src;
		dane[blockIdx.x*blockDim.x + threadIdx.x] = nowy;
	}
}

void CudaTekstury::odswiez_bskany() {

	//	MessageBox(NULL, "bskany", "", MB_OK);

	//	dim3 grid(rozmiarAskanu);
	//	dim3 block(szerokoscBskanu);

	dim3 grid(szerokoscBskanu);
	dim3 block(rozmiarAskanu*liczbaBskanow);

	//	dim3 block(32, 32);
	//	dim3 grid((-1 + szerokoscBskanu) / 32 + 1,(-1 + rozmiarAskanu) / 32 + 1 );

	for (size_t i = 0; i<liczbaBskanow; ++i)
		odswiezaniePrzekroju << <grid, block, 0, streams[0] >> > (daneGPU_tab, d_mapaKolory_Szarosc);
	//tworzenieBskanu << <grid, block >> > (bskany[i], daneGPU, floor(i*krok_bskan), d_mapaKolorow);

}


void CudaTekstury::odswiez_przekrojePoprzeczne() {

	//	dim3 grid(glebokoscPomiaru);
	//	dim3 block(szerokoscBskanu);

	dim3 grid(szerokoscBskanu);
	dim3 block(glebokoscPomiaru);

	//	dim3 block(32, 32);
	//	dim3 grid((-1 + szerokoscBskanu) / 32 + 1, (-1 + glebokoscPomiaru) / 32 + 1);

	for (size_t i = 0; i<liczbaPrzekrojowPoprzecznych; ++i)
		kolorowaniePrzekrojuPoprzecznego << <grid, block, 0, streams[1] >> > (przekrojePoprzeczne[i], d_mapaKolory_Szarosc, szerokoscBskanu, glebokoscPomiaru);
	//tworzeniePrzekrojuPoprzecznego << <grid, block>> > (przekrojePoprzeczne[i], daneGPU, floor(i*krok_przekrojePoprzeczne), rozmiarAskanu, d_mapaKolorow);
}

void CudaTekstury::odswiez_przekrojePoziome() {

	//	dim3 grid(rozmiarAskanu);
	//	dim3 block(glebokoscPomiaru);

	dim3 grid(glebokoscPomiaru);
	dim3 block(rozmiarAskanu);

	//	dim3 block(32, 32);
	//	dim3 grid((-1 + glebokoscPomiaru) / 32 + 1, (-1 + rozmiarAskanu) / 32 + 1);

	for (size_t i = 0; i<liczbaPrzekrojowPoziomych; ++i)
		kolorowaniePrzekrojuPoziomego << < grid, block, 0, streams[2] >> > (przekrojePoziome[i], d_mapaKolory_Szarosc, glebokoscPomiaru, rozmiarAskanu);
	//tworzeniePrzekrojuPoziomego << < grid, block>> > (przekrojePoziome[i], daneGPU, floor(i*krok_przekrojePoziome), szerokoscBskanu, d_mapaKolorow);
}


void CudaTekstury::odswiezTekstury2() {

	LARGE_INTEGER countPerSec, tim1, tim2;
	QueryPerformanceFrequency(&countPerSec);
	QueryPerformanceCounter(&tim1);

	//	std::async(std::launch::async, [&] {launch_bskany2(); });
	//	std::async(std::launch::async, [&] {launch_przekrojePoprzeczne2(); });
	//	std::async(std::launch::async, [&] {launch_przekrojePoziome2(); });


	std::thread t1([&] {launch_bskany2(); });
	std::thread t2([&] {launch_przekrojePoprzeczne2(); });
	std::thread t3([&] {launch_przekrojePoziome2(); });

	t1.join();
	t2.join();
	t3.join();

	cudaDeviceSynchronize();
	QueryPerformanceCounter(&tim2);
	double j = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	printf("czas: %f\n", j);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
}