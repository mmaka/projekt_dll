#pragma once

struct Werteks {

	float x, y, z;
//	float nx, ny, nz;
	float s, t;
	float r, g, b, a;

	static const int liczbaWpolrzednychPolozenia = 3;
	static const int rozmiarWektoraPolozenia = liczbaWpolrzednychPolozenia * sizeof(float);
//	static const int liczbaWspolrzednychNormalnej = 3;
//	static const int rozmiarNormalnej = liczbaWspolrzednychNormalnej * sizeof(float);
	static const int liczbaWspolrzednychTeksturowania = 2;
	static const int rozmiarWspolrzednychTeksturowania = liczbaWspolrzednychTeksturowania * sizeof(float);
	static const int liczbaSkladowychKoloru = 4;
	static const int rozmiarWektoraKoloru = liczbaSkladowychKoloru * sizeof(float);
//	static const int rozmiarWerteksu = rozmiarWektoraPolozenia + rozmiarNormalnej+rozmiarWspolrzednychTeksturowania+ rozmiarWektoraKoloru;
	static const int rozmiarWerteksu = rozmiarWektoraPolozenia  + rozmiarWspolrzednychTeksturowania + rozmiarWektoraKoloru;

	Werteks()
		: x(0.0f), y(0.0f), z(0.0f),
//		nx(0.0f), ny(0.0f),nz(0.0f),
		s(0.0f),t(0.0f),
		r(0.0f), g(0.0f), b(0.0f), a(1.0f) {}

	Werteks(float x, float y, float z,
	//	float nx, float ny, float nz,
		float s,float t,
		float r=1.0f, float g=1.0f, float b=1.0f, float a = 1.0f)
		: x(x), y(y), z(z),
	//	nx(nx),ny(ny),nz(nz),
		s(s),t(t),
		r(r), g(g), b(b), a(a) {}
};