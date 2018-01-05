//Jacek Matulewski, e-mail: jacek@fizyka.umk.pl
//wersja 1.1

//pomoc w optymalizacji: Krzystof Chyzi�ski, Micha� Zieli�ski

#ifndef MACIERZGL_H
#define MACIERZGL_H

#include "Macierz.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <exception>

template<typename T>
T StopnieDoRadianow(T k�tWStopniach)
{
	return (T)(M_PI * k�tWStopniach / (T)180);
}

template<typename T>
T RadianyDoStopni(T k�tWStopniach)
{
	return (T)(180 * k�tWStopniach / M_PI);
}

#include "Wektor.h"

template<typename T>
struct TMacierzGrafika3D : public TMacierzKwadratowa<T, 4>
{
	/*
	wiersz,kolumna
	00 01 02 03
	10 11 12 13
	20 21 22 23
	30 31 32 33

	wiersz+kolumna*stopie� (column-majored order)
	0 4 8  12
	1 5 9  13
	2 6 10 14
	3 7 11 15
	*/

public:
	//dlaczego nie widzi konstruktor�w z szablonu bazowego - wola�bym tego nie mie�!!!!!!!!!!!!!!!
	TMacierzGrafika3D()
		: TMacierzKwadratowa<T, 4>()
	{
	}

	//dlaczego nie widzi konstruktor�w z szablonu bazowego - wola�bym tego nie mie�!!!!!!!!!!!!!!!!
	TMacierzGrafika3D(const T elementy[16])
		: TMacierzKwadratowa<T, 4>(elementy)
	{
	}

	TMacierzGrafika3D(const TMacierzKwadratowa<T, 4>& m)
		: TMacierzKwadratowa<T, 4>(m)
	{
	}

	Wektor4 KopiaKolumny(const int indeksKolumny) const
	{
		T elementyKolumny[4];
		TMacierzKwadratowa<T,4>::KopiaKolumny(indeksKolumny, elementyKolumny);
		return Wektor4(elementyKolumny);
	}

	Wektor4 KopiaWiersza(const int indeksWiersza) const
	{
		T elementyWiersza[4];
		KopiaWiersza(indeksWiersza, elementyWiersza);
		return Wektor4(elementyWiersza);
	}

	void UstawSkalowanie(T sx, T sy, T sz)
	{
		ZerujElementy();
		UstawElement(0, 0, sx);
		UstawElement(1, 1, sy);
		UstawElement(2, 2, sz);
		UstawElement(3, 3, 1);
	}

	static TMacierzGrafika3D Skalowanie(T sx, T sy, T sz)
	{
		TMacierzGrafika3D m;
		m.UstawSkalowanie(sx, sy, sz);
		return m;
	}

	void UstawPrzesuniecie(T tx, T ty, T tz)
	{
		UstawJednostkowa();
		UstawElement(0, 3, tx);
		UstawElement(1, 3, ty);
		UstawElement(2, 3, tz);
	}

	void UstawPrzesuniecie(Wektor3 t)
	{
		UstawPrzesuniecie(t[0], t[1], t[2]);
	}

	static TMacierzGrafika3D Przesuniecie(T tx, T ty, T tz)
	{
		TMacierzGrafika3D<T> m;
		m.UstawPrzesuniecie(tx, ty, tz);
		return m;
	}

	static TMacierzGrafika3D Przesuniecie(Wektor3 t)
	{
		return Przesuniecie(t[0], t[1], t[2]);
	}

	//w funkcjach obrot�w k�ty podawane s� w stopniach
	void UstawObrotX(T kat, bool katWRadianach = false)
	{
		if (!katWRadianach) kat = StopnieDoRadianow(kat);
		UstawJednostkowa();
		T s = sin(kat);
		T c = cos(kat);
		UstawElement(1, 1, c);
		UstawElement(1, 2, -s);
		UstawElement(2, 1, s);
		UstawElement(2, 2, c);
	}

	static TMacierzGrafika3D ObrotX(T k�t, bool k�tWRadianach = false)
	{
		TMacierzGrafika3D m;
		m.UstawObrotX(k�t, k�tWRadianach);
		return m;
	}

	void UstawObrotY(T k�t, bool katWRadianach = false)
	{
		if (!katWRadianach) k�t = StopnieDoRadianow(k�t);
		UstawJednostkowa();
		T s = sin(k�t);
		T c = cos(k�t);
		UstawElement(0, 0, c);
		UstawElement(0, 2, s);
		UstawElement(2, 0, -s);
		UstawElement(2, 2, c);
	}

	static TMacierzGrafika3D ObrotY(T k�t, bool k�tWRadianach = false)
	{
		TMacierzGrafika3D m;
		m.UstawObrotY(k�t, k�tWRadianach);
		return m;
	}

	void UstawObrotZ(T k�t, bool k�tWRadianach = false)
	{
		if (!k�tWRadianach) k�t = StopnieDoRadianow(k�t);
		UstawJednostkowa();
		T s = sin(k�t);
		T c = cos(k�t);
		UstawElement(0, 0, c);
		UstawElement(0, 1, -s);
		UstawElement(1, 0, s);
		UstawElement(1, 1, c);
	}

	static TMacierzGrafika3D ObrotZ(T k�t, bool k�tWRadianach = false)
	{
		TMacierzGrafika3D m;
		m.UstawObrotZ(k�t, k�tWRadianach);
		return m;
	}

	//yaw, pitch, roll
	void UstawObrotXYZ(T k�tX, T k�tY, T k�tZ, bool k�tWRadianach = false)
	{
		if (!k�tWRadianach)
		{
			k�tX = StopnieDoRadianow(k�tX);
			k�tY = StopnieDoRadianow(k�tY);
			k�tZ = StopnieDoRadianow(k�tZ);
		}
		T sx = sin(k�tX); T cx = cos(k�tX);
		T sy = sin(k�tY); T cy = cos(k�tY);
		T sz = sin(k�tZ); T cz = cos(k�tZ);
		ZerujElementy();
		UstawElement(0, 0, cy*cz);
		UstawElement(0, 1, -cy*sz);
		UstawElement(0, 2, sy);
		UstawElement(1, 0, sx*sy*cz + cx*sz);
		UstawElement(1, 1, -sx*sy*sz + cx*cz);
		UstawElement(1, 2, -sx*cy);
		UstawElement(2, 0, -cx*sy*cz + sx*sz);
		UstawElement(2, 1, cx*sy*sz + sx*cz);
		UstawElement(2, 2, cx*cy);
		UstawElement(3, 3, 1);
	}

	static TMacierzGrafika3D ObrotXYZ(T k�tX, T k�tY, T k�tZ, bool k�tWRadianach = false)
	{
		TMacierzGrafika3D m;
		m.UstawObrotXYZ(k�tX, k�tY, k�tZ, k�tWRadianach);
		return m;
	}

	//k�ty Eulera
	void UstawObrotZXZ(T k�tZ2, T k�tX, T k�tZ1, bool k�tWRadianach = false)
	{
		if (!k�tWRadianach)
		{
			k�tZ2 = StopnieDoRadianow(k�tZ2);
			k�tX = StopnieDoRadianow(k�tX);
			k�tZ1 = StopnieDoRadianow(k�tZ1);
		}
		T sz2 = sin(k�tZ2); T cz2 = cos(k�tZ2);
		T sx = sin(k�tX); T cx = cos(k�tX);
		T sz1 = sin(k�tZ1); T cz1 = cos(k�tZ1);
		ZerujElementy();
		UstawElement(0, 0, cz2*cz1 - sz2*cx*sz1);
		UstawElement(0, 1, -cz2*sz1 - sz2*cx*cz1);
		UstawElement(0, 2, sz2*sx);
		UstawElement(1, 0, sz2*cz1 + cz2*cx*sz1);
		UstawElement(1, 1, -sz2*sz1 + cz2*cx*cz1);
		UstawElement(1, 2, -cz2*sx);
		UstawElement(2, 0, sx*sz1);
		UstawElement(2, 1, sx*cz1);
		UstawElement(2, 2, cx);
		UstawElement(3, 3, 1);
	}

	static TMacierzGrafika3D ObrotZXZ(T k�tZ2, T k�tX, T k�tZ1, bool k�tWRadianach = false)
	{
		TMacierzGrafika3D m;
		m.UstawObrotZXZ(k�tZ2, k�tX, k�tZ1, k�tWRadianach);
		return m;
	}

	void UstawObrotWokolOsi(T k�t, T ux, T uy, T uz, bool k�tWRadianach = false)
	{
		if (!k�tWRadianach) k�t = StopnieDoRadianow(k�t);
		T s = sin(k�t); 
		T c = cos(k�t);
		ZerujElementy();
		UstawElement(0, 0, c + (1 - c)*ux*ux);
		UstawElement(0, 1, (1 - c)*ux*uy - s*uz);
		UstawElement(0, 2, (1 - c)*uz*ux + s*uy);
		UstawElement(1, 0, (1 - c)*ux*uy + s*uz);
		UstawElement(1, 1, c + (1 - c)*uy*uy);
		UstawElement(1, 2, (1 - c)*uy*uz - s*ux);
		UstawElement(2, 0, (1 - c)*uz*ux - s*uy);
		UstawElement(2, 1, (1 - c)*uy*uz + s*ux);
		UstawElement(2, 2, c + (1 - c)*uz*uz);
		UstawElement(3, 3, 1);
	}

	void UstawObrotWokolOsi(T k�t, Wektor3 u, bool k�tWRadianach = false)
	{
		UstawObrotWokolOsi(k�t, u[0], u[1], u[2], k�tWRadianach);
	}

	static TMacierzGrafika3D ObrotWokolOsi(T k�t, T ux, T uy, T uz, bool k�tWRadianach = false)
	{
		TMacierzGrafika3D m;
		m.UstawObrotWokolOsi(k�t, ux, uy, uz, k�tWRadianach);
		return m;
	}

	static TMacierzGrafika3D ObrotWokolOsi(T k�t, Wektor3 u, bool k�tWRadianach = false)
	{
		return ObrotWokolOsi(k�t, u[0], u[1], u[2], k�tWRadianach);
	}

	void UstawRzutNaP�aszczyzne(Wektor3 po�o�enie�r�d�a�wiat�a, Wektor3 normalnaDoP�aszczyznyRzutowania, float przesuni�cieP�aszczyznyRzutowania)
	{		
		Wektor4 L = Wektor4(po�o�enie�r�d�a�wiat�a[0], po�o�enie�r�d�a�wiat�a[1], po�o�enie�r�d�a�wiat�a[2], 1);
		Wektor4 N = Wektor4(normalnaDoP�aszczyznyRzutowania[0], normalnaDoP�aszczyznyRzutowania[1], normalnaDoP�aszczyznyRzutowania[2], przesuni�cieP�aszczyznyRzutowania);
		float alfa = Wektor4::IloczynSkalarny(N, L);
		//
		UstawElement(0, 0, alfa - N[0] * L[0]);
		UstawElement(0, 1, -N[1] * L[0]);
		UstawElement(0, 2, -N[2] * L[0]);
		UstawElement(0, 3, -N[3] * L[0]);
		//
		UstawElement(1, 0, -N[0] * L[1]);
		UstawElement(1, 1, alfa - N[1] * L[1]);		
		UstawElement(1, 2, -N[2] * L[1]);
		UstawElement(1, 3, -N[3] * L[1]);
		//
		UstawElement(2, 0, -N[0] * L[2]);
		UstawElement(2, 1, -N[1] * L[2]);
		UstawElement(2, 2, alfa - N[2] * L[2]);
		UstawElement(2, 3, -N[3] * L[2]);
		//
		UstawElement(3, 0, -N[0] * L[3]);
		UstawElement(3, 1, -N[1] * L[3]);
		UstawElement(3, 2, -N[2] * L[3]);
		UstawElement(3, 3, alfa - N[3] * L[3]);
	}

	static TMacierzGrafika3D RzutNaP�aszczyzne(Wektor3 po�o�enie�r�d�a�wiat�a, Wektor3 normalnaDoP�aszczyznyRzutowania, float przesuni�cieP�aszczyznyRzutowania)
	{
		TMacierzGrafika3D m;
		m.UstawRzutNaP�aszczyzne(po�o�enie�r�d�a�wiat�a, normalnaDoP�aszczyznyRzutowania, przesuni�cieP�aszczyznyRzutowania);
		return m;
	}

	void UstawRzutIzometryczny(T l, T r, T b, T t, T n, T f)
	{
		T w = r - l;
		T h = t - b;
		T d = f - n;
		UstawSkalowanie(2 / w, 2 / h, -2 / d);
		UstawElement(0, 3, -(r + l) / w);
		UstawElement(1, 3, -(t + b) / h);
		UstawElement(2, 3, -(f + n) / d);
	}

	static TMacierzGrafika3D RzutIzometryczny(T l, T r, T b, T t, T n, T f)
	{
		TMacierzGrafika3D m;
		m.UstawRzutIzometryczny(l, r, b, t, n, f);
		return m;
	}

	void UstawRzutPerspektywiczny(T l, T r, T b, T t, T n, T f)
	{
		T w = r - l;
		T h = t - b;
		T d = f - n;
		UstawSkalowanie(2 * n / w, 2 * n / h, -(f + n) / d);
		UstawElement(0, 2, (r + l) / w);
		UstawElement(1, 2, (t + b) / h);
		UstawElement(3, 2, -1);
		UstawElement(2, 3, -2 * n * f / d);
		UstawElement(3, 3, 0);
	}

	static TMacierzGrafika3D RzutPerspektywiczny(T l, T r, T b, T t, T n, T f)
	{
		TMacierzGrafika3D m;
		m.UstawRzutPerspektywiczny(l, r, b, t, n, f);
		return m;
	}

	static TMacierzGrafika3D UstawRzutPerspektywiczny2(T k�tPionowegoPolaWidzeniaWStopniach, T proporcjaEkranu, T n, T f)
	{
		T k�tPionowegoPolaWidzenia = StopnieDoRadianow(k�tPionowegoPolaWidzeniaWStopniach);
		T h = 2 * n * tan(k�tPionowegoPolaWidzenia / 2);
		T w = proporcjaEkranu * h;
		UstawRzutPerspektywiczny(-w / 2, w / 2, -h / 2, h / 2, n, f);
	}

	static TMacierzGrafika3D RzutPerspektywiczny2(T k�tPionowegoPolaWidzeniaWStopniach, T proporcjaEkranu, T n, T f)
	{
		TMacierzGrafika3D m;
		m.UstawRzutPerspektywiczny2(l, r, b, t, n, f);
		return m;
	}

	void UstawWidokPatrzNa(
		Wektor3 kamera,
		Wektor3 centrum,
		Wektor3 polaryzacja)
	{
		Wektor3 E = kamera;
		Wektor3 C = centrum;
		Wektor3 U = polaryzacja;

		Wektor3 F = C - E;
		Wektor3 Fp = F.Unormowany();

		Wektor3 R = Wektor3::IloczynWektorowy(Fp, U);
		R.Normuj();

		Wektor3 Up = Wektor3::IloczynWektorowy(R, Fp);

		ZerujElementy();
		for (int kolumna = 0; kolumna < 3; kolumna++)
		{
			UstawElement(0, kolumna, R[kolumna]);
			UstawElement(1, kolumna, Up[kolumna]);
			UstawElement(2, kolumna, -Fp[kolumna]);
		}
		UstawElement(0, 3, -Wektor3::IloczynSkalarny(R, E));
		UstawElement(1, 3, -Wektor3::IloczynSkalarny(Up, E));
		UstawElement(2, 3, Wektor3::IloczynSkalarny(Fp, E));
		UstawElement(3, 3, 1);
	}

	void UstawWidokPatrzNa(
		T kameraX, T kameraY, T kameraZ,
		T centrumX, T centrumY, T centrumZ,
		T polaryzacjaX, T polaryzacjaY, T polaryzacjaZ)
	{
		Wektor3 kamera(kameraX, kameraY, kameraZ);
		Wektor3 centrum(centrumX, centrumY, centrumZ);
		Wektor3 polaryzacja(polaryzacjaX, polaryzacjaY, polaryzacjaZ);
		UstawWidokPatrzNa(kamera, centrum, polaryzacja);
		
		/*
		T E[3] = { kameraX, kameraY, kameraZ };
		T C[3] = { centrumX, centrumY, centrumZ };
		T U[3] = { polaryzacjaX, polaryzacjaY, polaryzacjaZ };

		T F[3], Fp[3];
		R�nicaWektor�w<T, 3>(C, E, F);
		WektorUnormowany<T, 3>(F, Fp);

		T R[3];
		IloczynWektorowy<T>(Fp, U, R);
		NormujWektor<T, 3>(R);

		T Up[3];
		IloczynWektorowy<T>(R, Fp, Up);

		ZerujElementy();
		for (int kolumna = 0; kolumna < 3; kolumna++)
		{
			UstawElement(0, kolumna, R[kolumna]);
			UstawElement(1, kolumna, Up[kolumna]);
			UstawElement(2, kolumna, -Fp[kolumna]);
		}
		UstawElement(0, 3, -IloczynSkalarny<T, 3>(R, E));
		UstawElement(1, 3, -IloczynSkalarny<T, 3>(Up, E));
		UstawElement(2, 3, IloczynSkalarny<T, 3>(Fp, E));
		UstawElement(3, 3, 1);
		*/
	}

	static TMacierzGrafika3D WidokPatrzNa(
		Wektor3 kamera,
		Wektor3 centrum,
		Wektor3 polaryzacja)
	{
		TMacierzGrafika3D m;
		m.UstawWidokPatrzNa(kamera, centrum, polaryzacja);
		return m;
	}

	static TMacierzGrafika3D WidokPatrzNa(
		T kameraX, T kameraY, T kameraZ,
		T centrumX, T centrumY, T centrumZ,
		T polaryzacjaX, T polaryzacjaY, T polaryzacjaZ)
	{
		TMacierzGrafika3D m;
		m.UstawWidokPatrzNa(
			kameraX, kameraY, kameraZ,
			centrumX, centrumY, centrumZ,
			polaryzacjaX, polaryzacjaY, polaryzacjaZ);
		return m;
	}

	//PRZETESTOWAC!!!!!!!!!!!!!!!!!!!!!!
	static TMacierzGrafika3D OperatorGwiazdki(T x, T y, T z)
	{
		TMacierzGrafika3D m;
		m.UstawElement(0, 1, -z);
		m.UstawElement(0, 2, y);
		m.UstawElement(1, 0, z);
		m.UstawElement(1, 2, -x);
		m.UstawElement(2, 0, -y);
		m.UstawElement(2, 1, x);
		m.UstawElement(3, 3, 1);
		return m;
	}

	//http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
	TMacierzGrafika3D Odwrotna() const
	{
		TMacierzGrafika3D m = *this;
		T elementyMacierzyOdwrotnej[16];

		elementyMacierzyOdwrotnej[0] = 
			m[5] * m[10] * m[15] -
			m[5] * m[11] * m[14] -
			m[9] * m[6] * m[15] +
			m[9] * m[7] * m[14] +
			m[13] * m[6] * m[11] -
			m[13] * m[7] * m[10];

		elementyMacierzyOdwrotnej[4] = 
		    -m[4] * m[10] * m[15] +
			m[4] * m[11] * m[14] +
			m[8] * m[6] * m[15] -
			m[8] * m[7] * m[14] -
			m[12] * m[6] * m[11] +
			m[12] * m[7] * m[10];

		elementyMacierzyOdwrotnej[8] = 
			m[4] * m[9] * m[15] -
			m[4] * m[11] * m[13] -
			m[8] * m[5] * m[15] +
			m[8] * m[7] * m[13] +
			m[12] * m[5] * m[11] -
			m[12] * m[7] * m[9];

		elementyMacierzyOdwrotnej[12] = 
		    -m[4] * m[9] * m[14] +
			m[4] * m[10] * m[13] +
			m[8] * m[5] * m[14] -
			m[8] * m[6] * m[13] -
			m[12] * m[5] * m[10] +
			m[12] * m[6] * m[9];

		elementyMacierzyOdwrotnej[1] = 
			-m[1] * m[10] * m[15] +
			m[1] * m[11] * m[14] +
			m[9] * m[2] * m[15] -
			m[9] * m[3] * m[14] -
			m[13] * m[2] * m[11] +
			m[13] * m[3] * m[10];

		elementyMacierzyOdwrotnej[5] = 
			m[0] * m[10] * m[15] -
			m[0] * m[11] * m[14] -
			m[8] * m[2] * m[15] +
			m[8] * m[3] * m[14] +
			m[12] * m[2] * m[11] -
			m[12] * m[3] * m[10];

		elementyMacierzyOdwrotnej[9] = 
			-m[0] * m[9] * m[15] +
			m[0] * m[11] * m[13] +
			m[8] * m[1] * m[15] -
			m[8] * m[3] * m[13] -
			m[12] * m[1] * m[11] +
			m[12] * m[3] * m[9];

		elementyMacierzyOdwrotnej[13] = 
			m[0] * m[9] * m[14] -
			m[0] * m[10] * m[13] -
			m[8] * m[1] * m[14] +
			m[8] * m[2] * m[13] +
			m[12] * m[1] * m[10] -
			m[12] * m[2] * m[9];

		elementyMacierzyOdwrotnej[2] = 
			m[1] * m[6] * m[15] -
			m[1] * m[7] * m[14] -
			m[5] * m[2] * m[15] +
			m[5] * m[3] * m[14] +
			m[13] * m[2] * m[7] -
			m[13] * m[3] * m[6];

		elementyMacierzyOdwrotnej[6] = 
			-m[0] * m[6] * m[15] +
			m[0] * m[7] * m[14] +
			m[4] * m[2] * m[15] -
			m[4] * m[3] * m[14] -
			m[12] * m[2] * m[7] +
			m[12] * m[3] * m[6];

		elementyMacierzyOdwrotnej[10] = 
			m[0] * m[5] * m[15] -
			m[0] * m[7] * m[13] -
			m[4] * m[1] * m[15] +
			m[4] * m[3] * m[13] +
			m[12] * m[1] * m[7] -
			m[12] * m[3] * m[5];

		elementyMacierzyOdwrotnej[14] = 
			-m[0] * m[5] * m[14] +
			m[0] * m[6] * m[13] +
			m[4] * m[1] * m[14] -
			m[4] * m[2] * m[13] -
			m[12] * m[1] * m[6] +
			m[12] * m[2] * m[5];

		elementyMacierzyOdwrotnej[3] = 
			-m[1] * m[6] * m[11] +
			m[1] * m[7] * m[10] +
			m[5] * m[2] * m[11] -
			m[5] * m[3] * m[10] -
			m[9] * m[2] * m[7] +
			m[9] * m[3] * m[6];

		elementyMacierzyOdwrotnej[7] = 
			m[0] * m[6] * m[11] -
			m[0] * m[7] * m[10] -
			m[4] * m[2] * m[11] +
			m[4] * m[3] * m[10] +
			m[8] * m[2] * m[7] -
			m[8] * m[3] * m[6];

		elementyMacierzyOdwrotnej[11] = 
			-m[0] * m[5] * m[11] +
			m[0] * m[7] * m[9] +
			m[4] * m[1] * m[11] -
			m[4] * m[3] * m[9] -
			m[8] * m[1] * m[7] +
			m[8] * m[3] * m[5];

		elementyMacierzyOdwrotnej[15] = 
			m[0] * m[5] * m[10] -
			m[0] * m[6] * m[9] -
			m[4] * m[1] * m[10] +
			m[4] * m[2] * m[9] +
			m[8] * m[1] * m[6] -
			m[8] * m[2] * m[5];

		T wyznacznik = m[0] * elementyMacierzyOdwrotnej[0] + m[1] * elementyMacierzyOdwrotnej[4] + m[2] * elementyMacierzyOdwrotnej[8] + m[3] * elementyMacierzyOdwrotnej[12];
		if (wyznacznik == 0) throw std::exception("Macierz osobliwa");
		for (int i = 0; i < 16; i++) elementyMacierzyOdwrotnej[i] /= wyznacznik;
		return TMacierzGrafika3D(elementyMacierzyOdwrotnej);
	}
};

/* --------------------------------------------------------------------------------------- */

#include "glew.h"
#include "wglew.h"

struct MacierzOpenGL : public TMacierzGrafika3D<float>
{
private:
	GLint identyfikatorMacierzy; //uniform location

public:
	//rzutowanie na tablic� T[Rozmiar] - �amie zasady enkapsulacji, ale mo�e by� wygodne w OpenGL
	/*
	operator float*() const
	{
	return (float*)elementy;
	}
	*/

	MacierzOpenGL()
		: TMacierzGrafika3D<float>(), identyfikatorMacierzy(-1)
	{
	}

	MacierzOpenGL(const TMacierzKwadratowa<float, 4>& m)
		: TMacierzGrafika3D<float>(m), identyfikatorMacierzy(-1)
	{
	}

	MacierzOpenGL(const TMacierzGrafika3D<float>& m)
		: TMacierzGrafika3D<float>(m), identyfikatorMacierzy(-1)
	{
	}

	void ZwiazZIdentyfikatorem(GLint identyfikatorMacierzy, bool przeslijWartosc = false)
	{
		this->identyfikatorMacierzy = identyfikatorMacierzy;
		if (przeslijWartosc) PrzeslijWartosc();
	}

	GLint PobierzIdentyfikator()
	{
		return identyfikatorMacierzy;
	}

	void PrzeslijWartosc(bool zglaszajBladPrzyBrakuWiazania = false)
	{
		if (identyfikatorMacierzy < 0)
		{
			if (zglaszajBladPrzyBrakuWiazania) throw std::exception("Aby przes�a� macierz do programu shader�w ustaw wpierw jej identyfikator (uniform location)");
			else return;
		}
		glUniformMatrix4fv(identyfikatorMacierzy, 1, false, elementy);
	}

	static const MacierzOpenGL Jednostkowa;
	static const MacierzOpenGL Zerowa;
};

//const MacierzOpenGL MacierzOpenGL::Jednostkowa = MacierzOpenGL::Tw�rzJednostkow�();
//const MacierzOpenGL MacierzOpenGL::Zerowa = MacierzOpenGL();

typedef MacierzOpenGL Macierz4;

/* --------------------------------------------------------------------------------------- */

#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>


template<typename T>
bool TestyMacierzyGrafika3D(unsigned int ziarno, T tolerancjaBledu = (T)1E-9)
{
	bool wynik = true;
	const T zakres = 10;	
	//T tolerancjaBledu = (T)1E-9; //dla double
	//if (typeif(T) == typeid(float)) tolerancjaBledu = (T)1E-5; //dla float

	//zerowa macierz
	TMacierzGrafika3D<T> m;
	for (int i = 0; i < TMacierzGrafika3D<T>::Rozmiar; ++i) if (m[i] != 0) wynik = false;

	//skalowanie
	TWektor<T,4> wektor;
	T wsp�czynnikiSkalowania[3];
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
		wsp�czynnikiSkalowania[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	TMacierzGrafika3D<T> macierzSkalowania = TMacierzGrafika3D<T>::Skalowanie(wsp�czynnikiSkalowania[0], wsp�czynnikiSkalowania[1], wsp�czynnikiSkalowania[2]);
	TWektor<T,4> wynikWektor = macierzSkalowania.PrzetransformowanyWektor(wektor);
	for (int i = 0; i < 3; ++i) if (wynikWektor[i] != wsp�czynnikiSkalowania[i] * wektor[i]) wynik = false;

	//translacja
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	T przesuni�cie[3];
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
		przesuni�cie[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	TMacierzGrafika3D<T> macierzTranslacji = TMacierzGrafika3D<T>::Przesuniecie(przesuni�cie[0], przesuni�cie[1], przesuni�cie[2]);
	wynikWektor = macierzTranslacji.PrzetransformowanyWektor(wektor);
	for (int i = 0; i < 3; ++i) if (wynikWektor[i] != wektor[i] + przesuni�cie[i]) wynik = false;

	//obr�t OZ o 0 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	TMacierzGrafika3D<T> macierzObrotuZ = TMacierzGrafika3D<T>::ObrotZ(0);
	wynikWektor = macierzObrotuZ.PrzetransformowanyWektor(wektor);
	for (int i = 0; i < 3; ++i) if (wynikWektor[i] != wektor[i]) wynik = false;

	//obr�t OZ o 360 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuZ = TMacierzGrafika3D<T>::ObrotZ(360);
	wynikWektor = macierzObrotuZ.PrzetransformowanyWektor(wektor);
	for (int i = 0; i < 3; ++i) if (fabs(wynikWektor[i] - wektor[i])>tolerancjaBledu) wynik = false;

	//obr�t OZ o 180 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuZ = TMacierzGrafika3D<T>::ObrotZ(180);
	wynikWektor = macierzObrotuZ.PrzetransformowanyWektor(wektor);
	if (wynikWektor[2] != wektor[2]) wynik = false;
	if (fabs(wynikWektor[1] + wektor[1])>tolerancjaBledu) wynik = false;
	if (fabs(wynikWektor[0] + wektor[0])>tolerancjaBledu) wynik = false;

	//obr�t OZ o 90 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuZ = TMacierzGrafika3D<T>::ObrotZ(90);
	wynikWektor = macierzObrotuZ.PrzetransformowanyWektor(wektor);
	if (wynikWektor[2] != wektor[2]) wynik = false;
	if (fabs(wynikWektor[1] - wektor[0])>tolerancjaBledu) wynik = false;
	if (fabs(wynikWektor[0] + wektor[1])>tolerancjaBledu) wynik = false;

	//obr�t OZ o -90 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuZ = TMacierzGrafika3D<T>::ObrotZ(-90);
	wynikWektor = macierzObrotuZ.PrzetransformowanyWektor(wektor);
	if (wynikWektor[2] != wektor[2]) wynik = false;
	if (fabs(wynikWektor[1] + wektor[0])>tolerancjaBledu) wynik = false;
	if (fabs(wynikWektor[0] - wektor[1])>tolerancjaBledu) wynik = false;


	//obr�t OX o 0 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	TMacierzGrafika3D<T> macierzObrotuX = TMacierzGrafika3D<T>::ObrotX(0);
	wynikWektor = macierzObrotuX.PrzetransformowanyWektor(wektor);
	for (int i = 0; i < 3; ++i) if (wynikWektor[i] != wektor[i]) wynik = false;

	//obr�t OX o 360 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuX = TMacierzGrafika3D<T>::ObrotX(360);
	wynikWektor = macierzObrotuX.PrzetransformowanyWektor(wektor);
	for (int i = 0; i < 3; ++i) if (fabs(wynikWektor[i] - wektor[i])>tolerancjaBledu) wynik = false;

	//obr�t OX o 180 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuX = TMacierzGrafika3D<T>::ObrotX(180);
	wynikWektor = macierzObrotuX.PrzetransformowanyWektor(wektor);
	if (wynikWektor[0] != wektor[0]) wynik = false;
	if (fabs(wynikWektor[1] + wektor[1])>tolerancjaBledu) wynik = false;
	if (fabs(wynikWektor[2] + wektor[2])>tolerancjaBledu) wynik = false;

	//obr�t OX o 90 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuX = TMacierzGrafika3D<T>::ObrotX(90);
	wynikWektor = macierzObrotuX.PrzetransformowanyWektor(wektor);
	if (wynikWektor[0] != wektor[0]) wynik = false;
	if (fabs(wynikWektor[2] - wektor[1])>tolerancjaBledu) wynik = false;
	if (fabs(wynikWektor[1] + wektor[2])>tolerancjaBledu) wynik = false;

	//obr�t OX o -90 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuX = TMacierzGrafika3D<T>::ObrotX(-90);
	wynikWektor = macierzObrotuX.PrzetransformowanyWektor(wektor);
	if (wynikWektor[0] != wektor[0]) wynik = false;
	if (fabs(wynikWektor[2] + wektor[1])>tolerancjaBledu) wynik = false;
	if (fabs(wynikWektor[1] - wektor[2])>tolerancjaBledu) wynik = false;

	//obr�t OY o 0 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	TMacierzGrafika3D<T> macierzObrotuY = TMacierzGrafika3D<T>::ObrotY(0);
	wynikWektor = macierzObrotuY.PrzetransformowanyWektor(wektor);
	for (int i = 0; i < 3; ++i) if (wynikWektor[i] != wektor[i]) wynik = false;

	//obr�t OY o 360 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuY = TMacierzGrafika3D<T>::ObrotY(360);
	wynikWektor = macierzObrotuY.PrzetransformowanyWektor(wektor);
	for (int i = 0; i < 3; ++i) if (fabs(wynikWektor[i] - wektor[i])>tolerancjaBledu) wynik = false;

	//obr�t OY o 180 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuY = TMacierzGrafika3D<T>::ObrotY(180);
	wynikWektor = macierzObrotuY.PrzetransformowanyWektor(wektor);
	if (wynikWektor[1] != wektor[1]) wynik = false;
	if (fabs(wynikWektor[0] + wektor[0])>tolerancjaBledu) wynik = false;
	if (fabs(wynikWektor[2] + wektor[2])>tolerancjaBledu) wynik = false;

	//obr�t OY o 90 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuY = TMacierzGrafika3D<T>::ObrotY(90);
	wynikWektor = macierzObrotuY.PrzetransformowanyWektor(wektor);
	if (wynikWektor[1] != wektor[1]) wynik = false;
	if (fabs(wynikWektor[0] - wektor[2])>tolerancjaBledu) wynik = false;
	if (fabs(wynikWektor[2] + wektor[0])>tolerancjaBledu) wynik = false;

	//obr�t OY o -90 stopni
	for (int i = 0; i < 4; ++i)
	{
		wektor[i] = 0;
		wynikWektor[i] = 0;
	}
	for (int i = 0; i < 3; ++i)
	{
		wektor[i] = zakres*rand() / RAND_MAX;
	}
	wektor[3] = 1;
	macierzObrotuY = TMacierzGrafika3D<T>::ObrotY(-90);
	wynikWektor = macierzObrotuY.PrzetransformowanyWektor(wektor);
	if (wynikWektor[1] != wektor[1]) wynik = false;
	if (fabs(wynikWektor[0] + wektor[2])>tolerancjaBledu) wynik = false;
	if (fabs(wynikWektor[2] - wektor[0])>tolerancjaBledu) wynik = false;

	//zgodno�� obrot�w OX
	T k�t = 15;
	TMacierzGrafika3D<T> wzorzec = TMacierzGrafika3D<T>::ObrotX(k�t);
	TMacierzGrafika3D<T> por�wnywana = TMacierzGrafika3D<T>::ObrotXYZ(k�t, 0, 0); if (!wzorzec.R�wna(por�wnywana)) wynik = false;
	por�wnywana = TMacierzGrafika3D<T>::ObrotZXZ(0, k�t, 0); if (!wzorzec.Rowna(por�wnywana)) wynik = false;
	por�wnywana = TMacierzGrafika3D<T>::ObrotWokolOsi(k�t, 1, 0, 0); if (!wzorzec.Rowna(por�wnywana)) wynik = false;
	
	//zgodno�� obrot�w OY
	wzorzec = TMacierzGrafika3D<T>::ObrotY(k�t);
	por�wnywana = TMacierzGrafika3D<T>::ObrotXYZ(0, k�t, 0); if (!wzorzec.Rowna(por�wnywana)) wynik = false;
	por�wnywana = TMacierzGrafika3D<T>::ObrotWokolOsi(k�t, 0, 1, 0); if (!wzorzec.Rowna(por�wnywana)) wynik = false;
	
	//zgodno�� obrot�w OZ
	wzorzec = TMacierzGrafika3D<T>::ObrotZ(k�t);
	por�wnywana = TMacierzGrafika3D<T>::ObrotXYZ(0, 0, k�t); if (!wzorzec.Rowna(por�wnywana)) wynik = false;
	por�wnywana = TMacierzGrafika3D<T>::ObrotZXZ(k�t, 0, 0); if (!wzorzec.Rowna(por�wnywana)) wynik = false;
	por�wnywana = TMacierzGrafika3D<T>::ObrotZXZ(0, 0, k�t); if (!wzorzec.Rowna(por�wnywana)) wynik = false;
	por�wnywana = TMacierzGrafika3D<T>::ObrotWokolOsi(k�t, 0, 0, 1); if (!wzorzec.Rowna(por�wnywana)) wynik = false;


	//rzut izometryczny
	T l = -1;
	T r = 1;
	T b = (T)-0.71;
	T t = (T)0.71;
	T n = 1;
	T f = 10;
	TMacierzGrafika3D<T> macierzRzutu = TMacierzGrafika3D<T>::RzutIzometryczny(l, r, b, t, n, f);
	T elementyMacierzyWzorcowej[16];
	for (int i = 0; i < 16; ++i) elementyMacierzyWzorcowej[i] = 0;
	elementyMacierzyWzorcowej[0] = 2 / (r - l);
	elementyMacierzyWzorcowej[5] = 2 / (t - b);
	elementyMacierzyWzorcowej[10] = -2 / (f - n);
	elementyMacierzyWzorcowej[12] = -(r + l) / (r - l);
	elementyMacierzyWzorcowej[13] = -(t + b) / (t - b);
	elementyMacierzyWzorcowej[14] = -(f + n) / (f - n);
	elementyMacierzyWzorcowej[15] = 1;
	TMacierzGrafika3D<T> macierzWzorcowa(elementyMacierzyWzorcowej);
	for (int i = 0; i<TMacierzGrafika3D<T>::Rozmiar; ++i)
	{
		if (fabs(macierzRzutu[i] - macierzWzorcowa[i]) > tolerancjaBledu) wynik = false;
		if (fabs(macierzRzutu[i] - elementyMacierzyWzorcowej[i]) > tolerancjaBledu) wynik = false;
	}

	//rzut perspektywiczny
	macierzRzutu = TMacierzGrafika3D<T>::RzutPerspektywiczny(l, r, b, t, n, f);
	for (int i = 0; i < 16; ++i) elementyMacierzyWzorcowej[i] = 0;
	elementyMacierzyWzorcowej[0] = 2 * n / (r - l);
	elementyMacierzyWzorcowej[5] = 2 * n / (t - b);
	elementyMacierzyWzorcowej[8] = (r + l) / (r - l);
	elementyMacierzyWzorcowej[9] = (t + b) / (t - b);
	elementyMacierzyWzorcowej[10] = -(f + n) / (f - n);
	elementyMacierzyWzorcowej[11] = -1;
	elementyMacierzyWzorcowej[14] = -2 * n * f / (f - n);
	macierzWzorcowa.UstawElementy(elementyMacierzyWzorcowej);
	for (int i = 0; i<TMacierzGrafika3D<T>::Rozmiar; ++i)
	{
		if (fabs(macierzRzutu[i] - macierzWzorcowa[i]) > tolerancjaBledu) wynik = false;
		if (fabs(macierzRzutu[i] - elementyMacierzyWzorcowej[i]) > tolerancjaBledu) wynik = false;
	}

	//TMacierzGrafika3D<T> ig = TMacierzGrafika3D<T>::Jednostkowa*TMacierzGrafika3D<T>::OperatorGwiazdki(1, 0, 0);
	//if (ig.KopiaKolumny(0,))

	TMacierzGrafika3D<T> macierz = TMacierzGrafika3D<T>::ObrotXYZ(10, 15, 20);
	TMacierzGrafika3D<T> macierzOdwrotna = macierz.Odwrotna();
	TMacierzGrafika3D<T> macierzTransponowana = macierz.Transponowana();
	TMacierzGrafika3D<T> roznica = macierzOdwrotna - macierzTransponowana;
	if (!macierzTransponowana.Rowna(macierzOdwrotna, tolerancjaBledu)) wynik = false; //dla ortonormalnych odwrotna = transponowana
	TMacierzGrafika3D<T> macierzOdwrotna2 = macierzOdwrotna.Odwrotna();
	wynik = true;
	if (!macierzOdwrotna2.Rowna(macierz, tolerancjaBledu)) wynik = false;

	return wynik;
}

#endif