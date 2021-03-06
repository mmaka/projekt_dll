//Jacek Matulewski, e-mail: jacek@fizyka.umk.pl
//wersja 1.1

//pomoc w optymalizacji: Krzystof Chyziński, Michał Zieliński

#pragma warning(disable:4996)

#ifndef MACIERZ_H
#define MACIERZ_H

#include "Wektor.h"
#include <cassert>
#include <algorithm>
#include <functional>

template<typename T, int stopien> 
struct TMacierzKwadratowa
{
public:
	static const int Rozmiar = stopien * stopien;
	static const int Stopien = stopien;

protected: //dostęp do elementy bedzie potrzebny tylko w metodzie Macierz4::PrzeslijWartosc
	//przechowujemy jak w OpenGL (column-majored odrder, a nie row-majored order) m00, m01, m02, m10, m11, ... (wiersze,kolumny)
	T elementy[Rozmiar]; 

private:
	//iteratory
	T* poczatek()
	{
		return elementy;
	}
	T* koniec()
	{
		return elementy + Rozmiar;
	}
	const T* poczatek() const
	{
		return elementy;
	}
	const T* koniec() const
	{
		return elementy + Rozmiar;
	}

	//dostep do elementow macierz
	/*
	//TO WYWALAMY, A ZOSTAWIAMY TYLKO OPERATORY[]
	T& KontrolowanyDostepDoElementu(int indeks)
	{
		//if (indeks >= 0 && indeks < Rozmiar) return elementy[indeks];
		//else throw std::invalid_argument("Indeks spoza zakresu");

		assert(indeks >= 0 && indeks < Rozmiar);
		return elementy[indeks];
	}
	//K. Chyziński
	const T& KontrolowanyDostepDoElementu(int indeks) const
	{
		//if (indeks >= 0 && indeks < Rozmiar) return elementy[indeks];
		//else throw std::invalid_argument("Indeks spoza zakresu");

		assert(indeks >= 0 && indeks < Rozmiar);
		return elementy[indeks];
	}
	T& KontrolowanyDostepDoElementu(int wiersz, int kolumna)
	{
		//tu latwo zmienic zapis kolumnami na wierszami (jedyne miejsce)
		//if (kolumna >= 0 && kolumna < stopień && wiersz >= 0 && wiersz < stopień) return elementy[wiersz + stopień*kolumna];
		//else throw std::invalid_argument("Indeks spoza zakresu");

		assert(kolumna >= 0 && kolumna < stopień && wiersz >= 0 && wiersz < stopień);
		return elementy[wiersz + stopień*kolumna];
	}
	/* PRZEMYŚLEC: KontrolowaneUstawienieCalejTablicy + KontrolowaneOdczytanieCalejTablicy
	KontrolowanyDostepDoElementu(T bufor[Rozmiar])
	{
	assert()???
		std::copy(elementy, elementy + Rozmiar, bufor);
	}
	*/

public:
	//dostęp do elementów
	T& operator[](const int indeks)
	{
		assert(indeks >= 0 && indeks < Rozmiar);
		return elementy[indeks];
	};
	const T& operator[](const int indeks) const
	{
		assert(indeks >= 0 && indeks < Rozmiar);
		return elementy[indeks];
	};
	T& operator()(const int wiersz, const int kolumna)
	{
		assert(kolumna >= 0 && kolumna < stopien);
		assert(wiersz >= 0 && wiersz < stopien);
		return elementy[wiersz + stopien*kolumna];
	}
	const T& operator()(const int wiersz, const int kolumna) const
	{
		assert(0 <= kolumna && kolumna < stopien);
		assert(0 <= wiersz  && wiersz  < stopien);
		return elementy[wiersz + stopien*kolumna];
	}

	T PobierzElement(const int indeks) const
	{
		//return const_cast<TMacierzKwadratowa*>(this)->KontrolowanyDostepDoElementu(indeks);		
		return this->operator[](indeks);
	}
	T PobierzElement(const int wiersz, const int kolumna) const
	{
		//return const_cast<TMacierzKwadratowa*>(this)->KontrolowanyDostepDoElementu(wiersz, kolumna);		
		return this->operator()(wiersz, kolumna);
	}

	void ZerujElementy()
	{
		for (int i = 0; i < Rozmiar; ++i) this->operator[](i) = 0;
		//for (auto &element : elementy) element = 0;
	}
	void UstawJednostkowa()
	{
		ZerujElementy();
		for (int kolumnaWiersz = 0; kolumnaWiersz < stopien; ++kolumnaWiersz)
			this->operator()(kolumnaWiersz, kolumnaWiersz) = 1;
	}

	void UstawElement(const int indeksElementu, T wartosc)
	{
		this->operator[](indeksElementu) = wartosc;
	}
	void UstawElement(const int wiersz, const int kolumna, T wartosc)
	{
		this->operator()(wiersz, kolumna) = wartosc;
	}
	void UstawElementy(const T* elementyMacierzy)
	{
		//for (int i = 0; i < Rozmiar; ++i) UstawElement(i, elementy[i]);
		std::copy(elementyMacierzy, elementyMacierzy + Rozmiar, elementy);
	}
	void Ustaw(const TMacierzKwadratowa& m)
	{
		UstawElementy(m.elementy);
	}

	T* KopiaElementow(T* bufor) const
	{
		//for (int i = 0; i < Rozmiar; ++i) bufor[i] = PobierzElement(i);
		std::copy(poczatek(), koniec(), bufor);
		return bufor;
	}
	T* KopiaKolumny(const int indeksKolumny, T* bufor) const
	{		
		//if (indeksKolumny < 0 && indeksKolumny >= stopień) throw std::invalid_argument("Indeks kolumny spoza zakresu");
		assert(indeksKolumny >= 0 && indeksKolumny < stopien);
		//for (int i = 0; i < stopień; ++i) bufor[i] = PobierzElement(i, indeksKolumny);
		//*
		const T* poczatekKolumny = poczatek() + indeksKolumny * stopien;
		std::copy(poczatekKolumny, poczatekKolumny + stopien, bufor);
		//*/
		return bufor;
	}
	T* KopiaWiersza(const int indeksWiersza, T* bufor) const
	{
		assert(indeksWiersza >= 0 && indeksWiersza < stopien);
		for (int i = 0; i < stopien; ++i) bufor[i] = PobierzElement(indeksWiersza, i);
		return bufor;
	}

	//konstruktory
	TMacierzKwadratowa()
	{
		ZerujElementy();
	};
	TMacierzKwadratowa(const T* elementy)
	{
		UstawElementy(elementy);
	};

	//operatory relacji
	bool operator==(const TMacierzKwadratowa& m) const
	{
		/*
		for (int i = 0; i < Rozmiar; ++i)
			if (PobierzElement(i) != m.PobierzElement(i))
				return false;
		return true;
		*/
		return std::equal(poczatek(), koniec(), m.poczatek());
	}
	bool operator!=(const TMacierzKwadratowa& w) const
	{
		return !(*this == w);
	}
	bool Rowna(const TMacierzKwadratowa& m, T tolerancjaBledu = 0) const
	{
		for (int i = 0; i<Rozmiar; ++i)
			if (fabs(PobierzElement(i) - m.PobierzElement(i))>tolerancjaBledu)
				return false;
		return true;
	}

	//operatory arytmetyczne
	//NIE TESTOWANE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	TMacierzKwadratowa& operator+=(const TMacierzKwadratowa& m)
	{
		//for (int i = 0; i < Rozmiar; ++i) KontrolowanyDostepDoElementu(i) += m.PobierzElement(i);
		std::transform(poczatek(), koniec(), m.poczatek(), poczatek(), std::plus<T>());
		return *this;
	}
	//NIE TESTOWANE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	TMacierzKwadratowa operator+(const TMacierzKwadratowa& m) const
	{
		//return TMacierzKwadratowa(*this) += m;
		//*
		TMacierzKwadratowa wynik;
		std::transform(poczatek(), koniec(), m.poczatek(), wynik.poczatek(), std::plus<T>());
		return wynik;
		//*/
	};
	TMacierzKwadratowa& operator-=(const TMacierzKwadratowa& m)
	{
		//for (int i = 0; i < Rozmiar; ++i) this->operator[](i) -= m.PobierzElement(i);
		std::transform(poczatek(), koniec(), m.poczatek(), poczatek(), std::minus<T>());
		return *this;
	}
	TMacierzKwadratowa operator-(const TMacierzKwadratowa& m) const
	{
		//return TMacierzKwadratowa(*this) -= m;
		//*
		TMacierzKwadratowa wynik;
		std::transform(poczatek(), koniec(), m.poczatek(), wynik.poczatek(), std::minus<T>());
		return wynik;
		//*/
	};
	TMacierzKwadratowa& operator*=(const T a)
	{
		for (int i = 0; i < Rozmiar; ++i) this->operator[](i) *= a;
		//for (T &element : *this) element *= a; // C++11 
		return *this;
	};
	TMacierzKwadratowa operator*(const T a) const
	{
		return TMacierzKwadratowa(*this) *= a;
	};
	TMacierzKwadratowa& operator/=(const T a)
	{
		assert(a != 0);
		for (int i = 0; i < Rozmiar; ++i) this->operator[](i) /= a;
		return *this;
	};
	TMacierzKwadratowa operator/(const T a) const
	{
		return TMacierzKwadratowa(*this) /= a;
	};

	TMacierzKwadratowa& PomnozZPrawej(const TMacierzKwadratowa& m)
	{
		//tu wystarczy dodatkowy wiersz (wyzej wypelniac oryginalna)
		//*
		TMacierzKwadratowa kopia = *this;
		this->ZerujElementy();
		for (int kolumna = 0; kolumna < stopien; ++kolumna)
			for (int wiersz = 0; wiersz < stopien; ++wiersz)
				for (int i = 0; i < stopien; ++i)
					this->operator()(wiersz, kolumna) += kopia.PobierzElement(wiersz, i)*m.PobierzElement(i, kolumna);
		return *this;
		//*/
		/*
		TMacierzKwadratowa kopia = *this;
				
		const T* kopia_element = kopia.początek(); //A
		const T* m_element = m.początek(); //B
		T* this_element = początek(); //C
		
		while (this_element < this->początek()+stopień) //warunek na wiersze wyniku
		{
			while (kopia_element < kopia.koniec())
			{
				*this_element = 0;
				while (m_element < m.koniec()) //obliczanie jednego elementu wyniku
				{
					*this_element = (*kopia_element) * (*m_element);
					kopia_element += 1; //w dół (następny wiersz)
					m_element += stopień; //w prawo (następna kolumna)
				}
				this_element += stopień; //następny element w wierszu (w prawo)
				m_element -= Rozmiar; //od początku
			}
			kopia_element -= Rozmiar; //od początku
			m_element += 1; //w dół
			this_element -= Rozmiar - 1; //od początku, kolejny element
		}
		
		return *this;
		*/
	};
	TMacierzKwadratowa& PomnozZLewej(const TMacierzKwadratowa& m)
	{
		//tu wystarczy dodatkowy wiersz (wyzej wypelniac oryginalna)		
		TMacierzKwadratowa kopia = *this;
		this->ZerujElementy();
		for (int kolumna = 0; kolumna < stopien; ++kolumna)
			for (int wiersz = 0; wiersz < stopien; ++wiersz)
				for (int i = 0; i < stopien; ++i)
					this->operator()(wiersz, kolumna) += m.PobierzElement(wiersz, i)*kopia.PobierzElement(i, kolumna);
		return *this;
	};
	TMacierzKwadratowa& operator*=(const TMacierzKwadratowa& m)
	{
		return PomnozZPrawej(m);
	}
	TMacierzKwadratowa operator*(const TMacierzKwadratowa m) const
	{
		//to napisać od nowa
		return TMacierzKwadratowa(*this) *= m;

		/*
		TMacierzKwadratowa wynik; // C = A + B;

		const T* ptr_A = elementy;
		const T* ptr_B = m.elementy;
		T* ptr_C = wynik.elementy;

		const T* ptr_AE = ptr_A + Rozmiar;
		const T *ptr_BE = ptr_B + Rozmiar;
		const T *ptr_CE = ptr_C + Stopień;

		while (ptr_C < ptr_CE)
		{
			while (ptr_A < ptr_AE)
			{
				*ptr_C = 0;

				while (ptr_B < ptr_BE)
				{
					*ptr_C += *ptr_A * *ptr_B;
					ptr_A += 1;
					ptr_B += Stopień;
				}

				ptr_C += Stopień;
				ptr_B -= Rozmiar;
			}

			ptr_A -= Rozmiar;
			ptr_B += 1;
			ptr_C -= Rozmiar - 1;
		}

		return wynik;
		*/
	};

	/*
	T* TransformujWektor(const T wektor[stopień], T wynik[stopień]) const //zmiana out-of-place
	{
		for (int wiersz = 0; wiersz < stopień; ++wiersz)
		{
			wynik[wiersz] = 0;
			for (int kolumna = 0; kolumna < stopień; ++kolumna)
				wynik[wiersz] += this->PobierzElement(wiersz, kolumna)*wektor[kolumna];
		}
		return wynik;
	}
	T* TransformujWektor(T wektor[stopień]) const //zmiana in-place
	{
		T wynik[stopień];
		TransformujWektor(wektor, wynik);
		for (int wiersz = 0; wiersz < stopień; ++wiersz) wektor[wiersz] = wynik[wiersz];
		return wektor;
	}
	*/
	void TransformujWektor(TWektor<T, stopien>& wektor) const
	{
		TWektor<T, n> kopia = wektor;
		for (int wiersz = 0; wiersz < stopien; ++wiersz)
		{
			wektor[wiersz] = 0;
			for (int kolumna = 0; kolumna < stopien; ++kolumna)
				wektor[wiersz] += this->PobierzElement(wiersz, kolumna)*kopia[kolumna];
		}
	}
	TWektor<T, stopien> PrzetransformowanyWektor(const TWektor<T, stopien>& wektor) const
	{		
		TWektor<T, stopien> wynik = wektor;
		TransformujWektor(wynik);
		return wynik;
	}
		

	//metody		
	/*
	T Slad() const
	{
		T slad = 0;
		for (int kolumna = 0; kolumna < stopień; ++kolumna)
			slad += PobierzElement(kolumna, kolumna);
		return slad;
	}
	*/

	//pomijamy implementację funkcji na odwrócenie macierzy, bo w przypadku macierzy 
	//ortogonalnej macierz odwrotna to macierz transponowana (a znacznie szybsze)
	TMacierzKwadratowa Transponowana() const
	{
		TMacierzKwadratowa m;
		m.ZerujElementy();
		for (int kolumna = 0; kolumna < stopien; ++kolumna)
			for (int wiersz = 0; wiersz < stopien; ++wiersz)
				m(wiersz, kolumna) = PobierzElement(kolumna, wiersz);
		return m;
	}	

protected:
	//ta metoda jest ukryta, ze względu na zbyt wiele kopiowań,
	//umożliwia jednak łatwe zainicjowanie statycznego stałego pola
	static TMacierzKwadratowa TworzJednostkowa()
	{
		TMacierzKwadratowa m;
		m.UstawJednostkowa();
		return m;
	}
};

/* --------------------------------------------------------------------------------------- */

#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

template<typename T, int stopien> 
bool TestyMacierzyKwadratowej(unsigned int ziarno)
{
	bool wynik = true;

	const int rozmiar = stopien * stopien;
	if (TMacierzKwadratowa<T, stopien>::Rozmiar != rozmiar) wynik = false;

	//test kolejnosci elementow (do obejrzenia w Locals lub Watch)
	T elementyMacierzyKolejnosc[rozmiar];
	for (int i = 0; i < rozmiar; ++i) elementyMacierzyKolejnosc[i] = (T)i;
	TMacierzKwadratowa<T, stopien> mKolejnosc(elementyMacierzyKolejnosc);
	for (int i = 0; i < rozmiar; ++i)
	{
		if (mKolejnosc[i] != i) wynik = false;
		if (mKolejnosc.PobierzElement(i) != i) wynik = false;
	}
	for (int kolumna = 0; kolumna < stopien; ++kolumna)
	{
		for (int wiersz = 0; wiersz < stopien; ++wiersz)
		{
			if (mKolejnosc.PobierzElement(wiersz, kolumna) != wiersz + stopien*kolumna) wynik = false;
		}
	}

	//testy na macierzy jednostkowej
	TMacierzKwadratowa<T, stopien> mI;
	mI.UstawJednostkowa();
	//if (mI.Slad() != stopień) wynik = false;
	if (mI*mI != mI) wynik = false;
	if (mI*mI*mI != mI) wynik = false;
	if (mI.Transponowana() != mI) wynik = false;

	//tworzenie macierzy z elementami losowymi
	const T zakres = 10;
	//T tolerancjaBledu=(T)1E-9; //dla double
	T elementyMacierzy[rozmiar];
	for (int i = 0; i < rozmiar; ++i) elementyMacierzy[i] = zakres*rand() / RAND_MAX;
	TMacierzKwadratowa<T, stopien> m(elementyMacierzy);

	//kontrola konstruktora kopiującego
	TMacierzKwadratowa<T, stopien> m2(m);
	for (int i = 0; i < rozmiar; ++i)
	{
		if (m[i] != m2[i]) wynik = false;
		m2[i] += 1;
		if (m[i] == m2[i]) wynik = false;
	}

	//test kopiowania wskaźnika
	TMacierzKwadratowa<T, stopien> m3 = m;
	for (int i = 0; i < rozmiar; ++i)
	{
		if (m[i] != m3[i]) wynik = false;
		m2[i] += 1;
		if (m[i] != m3[i]) wynik = false;
	}

	//odczyt elementow
	for (int i = 0; i < rozmiar; ++i)
	{
		if (m.PobierzElement(i) != elementyMacierzy[i]) wynik = false;
		if (m[i] != elementyMacierzy[i]) wynik = false;
	}
	for (int kolumna = 0; kolumna < stopien; ++kolumna)
		for (int wiersz = 0; wiersz < stopien; ++wiersz)
			if (m.PobierzElement(wiersz, kolumna) != elementyMacierzy[wiersz + stopien*kolumna]) wynik = false;
	T kopiaElementow[rozmiar];
	m.KopiaElementow(kopiaElementow);
	for (int i = 0; i < rozmiar; ++i) if (kopiaElementow[i] != elementyMacierzy[i]) wynik = false;

	//odczyt kolumn
	for (int kolumna = 0; kolumna < stopien; ++kolumna)
	{
		T kopiaElementowKolumny[stopien];
		m.KopiaKolumny(kolumna, kopiaElementowKolumny);
		for (int wiersz = 0; wiersz < stopien; ++wiersz)
			if (kopiaElementowKolumny[wiersz] != elementyMacierzy[wiersz + stopien*kolumna]) wynik = false;
	}

	//zapis elementow
	m.ZerujElementy();
	if (m != TMacierzKwadratowa<T, stopien>()) wynik = false;
	for (int i = 0; i < rozmiar; ++i) m.UstawElement(i, elementyMacierzy[i]);
	for (int i = 0; i < rozmiar; ++i) if (m[i] != elementyMacierzy[i]) wynik = false;
	m.ZerujElementy();
	for (int kolumna = 0; kolumna < stopien; ++kolumna)
		for (int wiersz = 0; wiersz < stopien; ++wiersz)
			m.UstawElement(wiersz, kolumna, elementyMacierzy[wiersz + stopien*kolumna]);
	for (int i = 0; i < rozmiar; ++i) if (m[i] != elementyMacierzy[i]) wynik = false;

	//testy operatorow mnożenia i metod
	if (m*mI != m) wynik = false;
	if (mI*m != m) wynik = false;
	TMacierzKwadratowa<T, stopien> kopia = m;
	kopia *= mI;
	if (kopia != m) wynik = false;
	
	kopia = m; 
	if (kopia.PomnozZPrawej(mI) != m) wynik = false;
	if (mI.PomnozZPrawej(m) != m) wynik = false;
	mI.UstawJednostkowa();
	kopia = m; 
	if (kopia.PomnozZLewej(mI) != m) wynik = false;
	if (mI.PomnozZLewej(m) != m) wynik = false;
	mI.UstawJednostkowa();

	if (m.Transponowana().Transponowana() != m) wynik = false;
	
	TMacierzKwadratowa<T, stopien> m0;
	m0.ZerujElementy();
	if (m + m0 != m) wynik = false;
	if (m0 + m != m) wynik = false;
	kopia = m; kopia += m0;
	if (kopia != m) wynik = false;
	kopia = m; kopia -= m0;
	if (kopia != m) wynik = false;

	kopia = m + m;
	for (int i = 0; i < rozmiar; ++i) if (kopia[i] != 2 * m[i]) wynik = false;
	kopia = m; kopia += m;
	for (int i = 0; i < rozmiar; ++i) if (kopia[i] != 2 * m[i]) wynik = false;

	//transformacja wektora
	TWektor<T,stopien> wektor;
	TWektor<T, stopien> wektorWynik;
	for (int i = 0; i < stopien; ++i) wektor[i] = zakres*rand() / RAND_MAX;
	wektorWynik = mI.PrzetransformowanyWektor(wektor);
	for (int i = 0; i < stopien; ++i)
	{
		if (wektor[i] != wektorWynik[i]) wynik = false;
	}
	mI.TransformujWektor(wektor);
	for (int i = 0; i < stopien; ++i) if (wektor[i] != wektorWynik[i]) wynik = false;

	return wynik;
}

#endif