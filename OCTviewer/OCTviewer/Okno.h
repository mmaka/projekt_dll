#pragma once
#include<Windows.h>
#include "MacierzGL.h"
#include"Tomogram.h"
#include"CrossSection.h"
#include<sstream>
#include"OCTviewer.h"
#include"CudaTekstury.cuh"

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
	TomogramTekstury *tekstury;
	CudaTekstury *cudaTekstury;
	LARGE_INTEGER countPerSec, tim1, tim2;
public:

	bool zmianaKoloru;
	bool flaga;
	std::stringstream vs;
	std::stringstream fs;
	visualizationParams parametryWyswietlania;
	
public:
	OknoGL(visualizationParams& params,CudaTekstury *cTekstury)
		: Okno(),
		uchwytRC(NULL), uchwytDC(NULL),
		macierzSwiata(Macierz4::Jednostkowa), macierzWidoku(Macierz4::Jednostkowa), macierzRzutowania(Macierz4::Jednostkowa),
		MVP(Macierz4::Jednostkowa), VP(Macierz4::Jednostkowa),
		swobodneObrotyKameryAktywne(false), teksturowanieWlaczone(true), kolor(true), zmianaKoloru(false), flaga(true), parametryWyswietlania(params),cudaTekstury(cTekstury) {};
	LRESULT __stdcall WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	~OknoGL() {
		cudaTekstury->~CudaTekstury();
	}
};
