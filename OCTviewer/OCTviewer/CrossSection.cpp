#include"CrossSection.h"


CrossSection::CrossSection(GLuint atrybutPolozenie, GLuint atrybutWspolrzedneTeksturowania, GLuint atrybutKolor, float dlugoscKrawedziX, float dlugoscKrawedziY)
	: liczbaWerteksow(-1), MacierzSwiata(Macierz4::Jednostkowa), przezroczystosc(1.0f),	IndeksTekstury(-1), dlugoscKrawedziX(dlugoscKrawedziX), dlugoscKrawedziY(dlugoscKrawedziY) {

	Inicjuj(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor);
}

void CrossSection::InicjujBuforWerteksow() {

	//Vertex Array Object (VAO)
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	//Vertex Buffer Object (VBO)
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	Werteks* werteksy = NULL;
	liczbaWerteksow = TworzTabliceWerteksow(werteksy);
	glBufferData(GL_ARRAY_BUFFER, liczbaWerteksow * sizeof(Werteks), werteksy, GL_STATIC_DRAW);
	delete[] werteksy;

}

void CrossSection::Inicjuj(GLuint atrybutPolozenie, GLuint atrybutWspolrzedneTeksturowania, GLuint atrybutKolor) {

	InicjujBuforWerteksow();

	glVertexAttribPointer(atrybutPolozenie, Werteks::liczbaWpolrzednychPolozenia, GL_FLOAT, GL_FALSE, Werteks::rozmiarWerteksu, 0);
	glEnableVertexAttribArray(atrybutPolozenie);
	glVertexAttribPointer(atrybutWspolrzedneTeksturowania, Werteks::liczbaWspolrzednychTeksturowania, GL_FLOAT, GL_FALSE, Werteks::rozmiarWerteksu, (const GLvoid*)(Werteks::rozmiarWektoraPolozenia));
	glEnableVertexAttribArray(atrybutWspolrzedneTeksturowania);
	glVertexAttribPointer(atrybutKolor, Werteks::liczbaSkladowychKoloru, GL_FLOAT, GL_FALSE, Werteks::rozmiarWerteksu, (const GLvoid*)(Werteks::rozmiarWektoraPolozenia + Werteks::rozmiarWspolrzednychTeksturowania));
	glEnableVertexAttribArray(atrybutKolor);

}

unsigned int CrossSection::TworzTabliceWerteksow(Werteks*& werteksy) {

	const float x0 = dlugoscKrawedziX / 2.0f;
	const float y0 = dlugoscKrawedziY / 2.0f;

	werteksy = new Werteks[4];

	werteksy[0] = Werteks(-x0, -y0, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f);
	werteksy[1] = Werteks(x0, -y0, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f);
	werteksy[2] = Werteks(-x0, y0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
	werteksy[3] = Werteks(x0, y0, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);


	return 4;
}


void CrossSection::Rysuj() {

	assert(liczbaWerteksow > 0);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, liczbaWerteksow);
	//glBindBuffer(GL_ARRAY_BUFFER,NULL);
	//glBindVertexArray(NULL);
}

void CrossSection::UsunBuforWerteksow() {

	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
}

CrossSection::~CrossSection() {

	UsunBuforWerteksow();
}