#pragma once
#include<Windows.h>
#include "MacierzGL.h"
#include"Tomogram.h"
#include"CrossSection.h"
#include<sstream>
#include"OCTviewer.h"
#include"CudaTekstury.cuh"
#include<memory>

class Okno {

protected:
	HWND uchwytOkna;
	long szerokoscObszaruUzytkownika;
	long wysokoscObszaruUzytkownika;

public:
	Okno() : uchwytOkna(NULL) {};
	LRESULT WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	bool Inicjuj(HINSTANCE uchwytAplikacji, POINT polozenieOkna, POINT rozmiarOkna, bool trybPelnoekranowy = false, bool zmianaRzdzielczosci = true);
	WPARAM Uruchom();
	virtual ~Okno() {};
private:
	bool ZmianaRozdzieloczosci(long szerokosc, long wysokosc, long glebiaKolorow = 32) const;
};


enum TrybKontroliKamery { tkkFPP, tkkTPP, tkkArcBall, tkkModel };

class  OknoGL : public Okno {
private:
	HGLRC uchwytRC;
	HDC uchwytDC;
	bool UstalFomatPikseli(HDC uchwytDC) const;
	bool InicjujWGL(HWND uchwytOkna);
	void UsunWGL();
	void UmiescInformacjeNaPaskuTytulu(HWND uchwytOkna);

	unsigned int idProgramuShaderow;
	static unsigned int KompilujShader(const char* nazwaPliku, GLenum typ, bool trybDebugowania = false);
	static unsigned int PrzygotujShadery(const char* vsNazwaPliku, const char* fsNazwaPliku, bool trybDebugowania = false);


	Wektor3 PobierzPolozenieKamery(bool pominOborty = false) const;
	float OdlegloscKamery() const;
	TrybKontroliKamery trybKontroliKamery = tkkArcBall;
	void ModyfikujPolozenieKamery(Macierz4 macierzPrzeksztalcenia);
	float przezroczystosc;

protected:

	CrossSection** przekroje;
	unsigned int liczbaPrzekrojow;
	unsigned int przygotujPrzekroje();
	void RysujAktorow();
	void UsunAktorow();
	bool swobodneObrotyKameryAktywne;
	void SwobodneObrotyKamery(const bool inicjacja, const float poczatowe_dx = 0, const float poczatkowe_dy = 0, const float wspolczynnikWygaszania = 0);
	void UstawienieSceny(bool rzutowanieIzometryczne = false);
	void RysujScene();
	Macierz4 macierzSwiata, macierzWidoku, macierzRzutowania;
	Macierz4 MVP, VP;
	typedef void (OknoGL::*TypMetodyObslugujacejPrzesuniecieMyszy)(const POINT biezacaPozycjaKursoraMyszy, const POINT przesuniecieKursoraMyszy);
	void ObslugaKlawiszy(WPARAM wParam);
	void ObliczaniePrzesunieciaMyszy(const LPARAM lParam, const float prog, POINT& poprzedniaPozycjaKursoraMyszy, TypMetodyObslugujacejPrzesuniecieMyszy MetodaObslugujacaPrzesuniecieMyszy);
	void ObslugaMyszyZWcisnietymLewymPrzyciskiem(const POINT biezacaPozycjaKursoraMyszy, const POINT przesuniecieKursoraMyszy);
	void ObslugaMyszyZWcisnietymPrawymPrzyciskiem(const POINT biezacaPozycjaKursoraMyszy, const POINT przesuniecieKursoraMyszy);
	void ObslugaRolkiMyszy(WPARAM wParam);

	const bool teksturowanieWlaczone;
	GLuint *indeksyTekstur;
	unsigned int liczbaTekstur;

	bool kolor;
	void UsunTekstury();
	TomogramTekstury tekstury;
	std::unique_ptr<CudaTekstury> cudaTekstury;
	
	LARGE_INTEGER countPerSec, tim1, tim2;
public:

	bool zmianaKoloru;
	bool flaga;
	bool tryb2Dgotowy;

	visualizationParams parametryWyswietlania;
	inline void zwiekszPrzezroczystosc() { if(przezroczystosc<1.0f) przezroczystosc += 0.01f; }
	inline void zmniejszPrzezroczystosc() { if (przezroczystosc > 0.0f) przezroczystosc -= 0.01f; }

	OknoGL(visualizationParams& params, char* dane,unsigned char defKol[256][3])
		: Okno(),
		uchwytRC(NULL), uchwytDC(NULL),
		macierzSwiata(Macierz4::Jednostkowa), macierzWidoku(Macierz4::Jednostkowa), macierzRzutowania(Macierz4::Jednostkowa),
		MVP(Macierz4::Jednostkowa), VP(Macierz4::Jednostkowa),
		swobodneObrotyKameryAktywne(false), teksturowanieWlaczone(true), kolor(true), zmianaKoloru(false),przezroczystosc(0.02f), tryb2Dgotowy(false), flaga(true), parametryWyswietlania(params),cudaTekstury(std::make_unique<CudaTekstury>(params,dane,defKol))/*cudaTekstury(std::move(cTekstury))*/ {}
	LRESULT __stdcall WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
};


extern class Wizualizator {

	visualizationParams params;
	char *dane;
	unsigned char defKol[256][3];
	std::unique_ptr<OknoGL> okno;

public:
	std::unique_ptr<OknoGL>& getOkno() { return okno; }
	int StworzOkno() {

		okno = std::make_unique<OknoGL>(params,dane,defKol);
		POINT polozenieOkna = { 100,100 };
		POINT rozmiarOkna = { 800,600 };
		if (!okno->Inicjuj(hInstance, polozenieOkna, rozmiarOkna)) {

			MessageBox(NULL, "Inicjacja okna nie powiodla sie", "Aplikacja OpenGL", MB_OK | MB_ICONERROR);
			return EXIT_FAILURE;
		}
		else return okno->Uruchom();
	}
	
	void wyswietlOkno() {

		StworzOkno();
		
	}


	void setParams(WIZUALIZACJA type, size ileBskanow, size ilePrzekrojowPoprzecznych, size ilePrzekrojowPoziomych, float xSizeScale, float ySizeScale, float zSizeScale, size bscanSize, size ascanSize, size depth, float x_size_mm, float y_size_mm, float z_size_mm, int jasn, int kontr)
	{
		params.typ = type;
		params.liczbaBskanow = ileBskanow;
		params.liczbaPrzekrojowPoprzecznych = ilePrzekrojowPoprzecznych;
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
	
	}
	void wczytajDaneBinarne(const char *nazwaPliku) {

		std::ifstream plik(nazwaPliku, std::ios::in | std::ios::binary);

		if (plik.is_open()) {

			std::stringstream s;
			int tmp;
			plik.read((char*)&tmp, sizeof(tmp));
			plik.read((char*)&tmp, sizeof(tmp));
			plik.read((char*)&tmp, sizeof(tmp));
			dane = new oct_t[params.bscanSize_px*params.ascanSize_px*params.depth_px];
			plik.read((char*)dane, params.bscanSize_px*params.ascanSize_px*params.depth_px * sizeof(oct_t));

		}
		else {

			MessageBox(NULL, "Blad otwarcia pliku!", "", NULL);

		}

		plik.close();
	}

	void pobierzDefinicjeKolorow(const char *nazwaPliku) {

		FILE *plik;
		errno_t err = fopen_s(&plik, nazwaPliku, "rb");
		if (err == 0) {
			for (int j = 0; j < 256; j++)
				fread(&defKol[j], 1, 3, plik);

			fclose(plik);
		}
		else {

			MessageBox(NULL, "blad pobieranie kolorowo", "", MB_OK);

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
};


