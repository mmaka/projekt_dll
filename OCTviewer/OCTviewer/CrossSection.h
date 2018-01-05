#pragma once

#include "MacierzGL.h"
#include "Werteks.h"


class CrossSection {

	unsigned int vao, vbo;
	unsigned int liczbaWerteksow;
	float dlugoscKrawedziX, dlugoscKrawedziY;
	void InicjujBuforWerteksow();
	void UsunBuforWerteksow();
	unsigned int TworzTabliceWerteksow(Werteks*& werteksy);

public:
	
	CrossSection(GLuint atrybutPolozenie, GLuint atrybutWspolrzedneTeksturowania, GLuint atrybuKolor, float dlugoscKrawedziX = 2.0f, float dlugoscKrawedziY = 2.0f);
	Macierz4 MacierzSwiata;
	void Inicjuj(GLuint atrybutPolozenie, GLuint atrybutWspolrzedneTeksturowania, GLuint atrybutKolor);
	void Rysuj();
	GLuint IndeksTekstury;
	GLuint IndeksTekstury2;
	float przezroczystosc;
	~CrossSection();
};