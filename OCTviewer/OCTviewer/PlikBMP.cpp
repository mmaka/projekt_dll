#include "PlikBMP.h"

#include <stdio.h>

bool CzyPlikIstnieje(const char* filename) {

	FILE* plik_test;
	if (fopen_s(&plik_test, filename, "r") != 0)
		return false;
	else
		fclose(plik_test);

	return true;
}

#include <stdexcept>

unsigned long* WczytajObrazZPlikuBitmap(HWND uchwytOkna, char* nazwaPliku, int& szerokoscObrazu, int& wysokoscObrazu, bool czytajZZasobow, unsigned char alfa) {

	try {
		//czytanie bitmapy z pliku lub z zasobow
		char komunikat[1024] = "Brak pliku obrazu";
		strcat_s(komunikat, nazwaPliku);
		if (!czytajZZasobow && !CzyPlikIstnieje(nazwaPliku))
			throw std::invalid_argument(komunikat);

		HBITMAP uchwytObrazu = (HBITMAP)LoadImage(GetModuleHandle(NULL), nazwaPliku, IMAGE_BITMAP, 0, 0, ((czytajZZasobow) ? 0 : LR_LOADFROMFILE) | LR_CREATEDIBSECTION);

		//informacje o bitmapie
		BITMAP obraz;
		GetObject(uchwytObrazu, sizeof(BITMAP), &obraz);
		szerokoscObrazu = obraz.bmWidth;
		wysokoscObrazu = obraz.bmHeight;

		//informacja o ilosc bitow na piksel
		//char bufor[256];
		//ShowMessage(_gcvt(obraz.bmBitsPixel,10,bufor);

		unsigned long* piksele = new unsigned long[szerokoscObrazu*wysokoscObrazu];

		switch (obraz.bmBitsPixel) {

		case 24:
		{
			unsigned char* piksele24bppBRG = new unsigned char[obraz.bmWidthBytes*wysokoscObrazu];
			memcpy(piksele24bppBRG, obraz.bmBits, szerokoscObrazu*wysokoscObrazu * 3);

			//konwersja do RGBA

			for (int ih = 0; ih < wysokoscObrazu; ++ih) {
				for (int iw = 0; iw < szerokoscObrazu; ++iw) {

					int i = 3 * iw + (ih*obraz.bmWidthBytes);//uwzglednia uzupelnienie do WORD
					unsigned char A = alfa;
					unsigned char B = piksele24bppBRG[i];
					unsigned char G = piksele24bppBRG[i + 1];
					unsigned char R = piksele24bppBRG[i + 2];

					/*
					//jezeli konwersja na BW
					unsigned char jasnosc = (R + G + B) / 3;
					R = jasnosc;
					B = jasnosc;
					G = jasnosc;
					*/

					piksele[iw + (ih*szerokoscObrazu)] = (A << 24) + (B << 16) + (G << 8) + (R);
				}
			}
			delete[] piksele24bppBRG;
		}
		break;

		case 1: //monochromatyczne
		{
			unsigned char* piksele1bppMono = new unsigned char[obraz.bmWidthBytes*wysokoscObrazu];
			memcpy(piksele1bppMono, obraz.bmBits, obraz.bmWidthBytes*wysokoscObrazu);

			//konwersja do RGBA

			for (int ih = 0; ih < wysokoscObrazu; ++ih) {

				for (int iw = 0; iw < szerokoscObrazu; ++iw) {

					int i = iw / 8 + (ih*obraz.bmWidthBytes);
					int numerBitu = iw % 8;
					unsigned char A = alfa;
					bool bitZapalony = ((piksele1bppMono[i] << numerBitu) & 128) == 128;
					//ignorujemy palete i tworzymy obraz czarno-bialy

					unsigned char B = bitZapalony ? 255 : 0;
					unsigned char G = bitZapalony ? 255 : 0;
					unsigned char R = bitZapalony ? 255 : 0;

					piksele[iw + (ih*szerokoscObrazu)] = (A << 24) + (B << 16) + (G << 8) + (R);
				}
			}

			delete[] piksele1bppMono;
		}
		break;

		case 8: //256 kolorow, wymaga palety barw (tabeli kolorow)
		{
			unsigned char* piksel8bppPalette = new unsigned char[obraz.bmWidthBytes*wysokoscObrazu];

			//pobieranie tabeli kolorow (pomijamy czytanie BITMAPINFO)
			memcpy(piksel8bppPalette, obraz.bmBits, obraz.bmWidthBytes*wysokoscObrazu);
			HDC uchwyt = CreateCompatibleDC(GetDC(uchwytOkna));
			SelectObject(uchwyt, uchwytObrazu);
			RGBQUAD tabelaKolorow[256];
			GetDIBColorTable(uchwyt, 0, 256, tabelaKolorow);

			for (int ih = 0; ih < wysokoscObrazu; ++ih) {

				for (int iw = 0; iw < szerokoscObrazu; ++iw) {

					int i = iw + (ih*obraz.bmWidthBytes);
					unsigned char A = alfa;
					unsigned char R = tabelaKolorow[piksel8bppPalette[i]].rgbRed;
					unsigned char G = tabelaKolorow[piksel8bppPalette[i]].rgbGreen;
					unsigned char B = tabelaKolorow[piksel8bppPalette[i]].rgbBlue;

					piksele[iw + (ih*szerokoscObrazu)] = (A << 24) + (B << 16) + (G << 8) + (R);
				}
			}
			delete[] piksel8bppPalette;
		}
		break;

		case 4: //16 kolorow (paleta barw jak paint)
		{
			unsigned char* piksele4bppPalette = new unsigned char[obraz.bmWidthBytes*wysokoscObrazu];
			memcpy(piksele4bppPalette, obraz.bmBits, obraz.bmWidthBytes*wysokoscObrazu);

			//pobieranie tabeli kolorow
			HDC uchwyt = CreateCompatibleDC(GetDC(uchwytOkna));
			SelectObject(uchwyt, uchwytObrazu);
			RGBQUAD tabelaKolorow[16];
			GetDIBColorTable(uchwyt, 0, 16, tabelaKolorow);

			for (int ih = 0; ih < wysokoscObrazu; ++ih) {

				for (int iw = 0; iw < szerokoscObrazu; ++iw) {

					int i = iw / 2 + (ih*obraz.bmWidthBytes);
					bool pierwszaPolowaBajtu = !(iw % 2);
					unsigned char A = alfa;
					int numerKoloruZPalety = (pierwszaPolowaBajtu) ? ((piksele4bppPalette[i] & 0xF0) >> 4) : (piksele4bppPalette[i] & 0x0F);
					unsigned char R = tabelaKolorow[numerKoloruZPalety].rgbRed;
					unsigned char G = tabelaKolorow[numerKoloruZPalety].rgbGreen;
					unsigned char B = tabelaKolorow[numerKoloruZPalety].rgbBlue;

					piksele[iw + (ih*szerokoscObrazu)] = (A << 24) + (B << 16) + (G << 8) + (R);
				}
			}
			delete[] piksele4bppPalette;
		}
		break;

		default: throw std::exception("Nieobslugiwany format bitmapy");

		}

		DeleteObject(uchwytObrazu);
		return piksele;
	}
	catch (const std::exception& exc) {

		char komunikat[256] = "blad podczas pobierania tekstury:\n";
		strcat_s(komunikat, exc.what());
		MessageBox(NULL, komunikat, "Blad teksturowania", NULL);

		return NULL;
	}
}