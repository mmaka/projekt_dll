#pragma once
#include"glew.h"
#include"OCTviewer.h"

enum TYP_PRZEKROJU
{
	BSKAN, PRZEKROJ_POPRZECZNY, PRZEKROJ_POZIOMY
};

class TomogramTekstury {

	unsigned int b_skan_rozmiar_x;
	unsigned int b_skan_rozmiar_y;
	unsigned int glebokoscPomiaru;
	unsigned int liczba_B_skanow;
	GLuint *indeksy_tekstur;
	unsigned int liczba_Przekrojow_Poprzecznych;
	unsigned int liczba_Przekrojow_Poziomych;
	bool zainicjalizowane = false;

public:
	explicit TomogramTekstury(visualizationParams params) {

		b_skan_rozmiar_x = params.bscanSize_px;
		b_skan_rozmiar_y = params.ascanSize_px;
		glebokoscPomiaru = params.depth_px;
		liczba_B_skanow = params.liczbaBskanow;
		liczba_Przekrojow_Poprzecznych = params.liczbaPrzekrojowPoprzecznych;
		liczba_Przekrojow_Poziomych = params.liczbaPrzekrojowPoziomych;
	}

	inline unsigned int liczbaTekstur() const { return liczba_B_skanow + liczba_Przekrojow_Poprzecznych +liczba_Przekrojow_Poziomych;}
	void inline parametryTekstur(unsigned int *indeksyTekstur, unsigned int ileTekstur, TYP_PRZEKROJU typ) {

		unsigned int szerokoscTekstury = 0;
		unsigned int wysokoscTekstury = 0;

		switch (typ) {

		case BSKAN:
			szerokoscTekstury = b_skan_rozmiar_x;
			wysokoscTekstury = b_skan_rozmiar_y;
			break;
		case PRZEKROJ_POPRZECZNY:
			szerokoscTekstury = b_skan_rozmiar_x;
			wysokoscTekstury = glebokoscPomiaru;
			break;
		case PRZEKROJ_POZIOMY:
			szerokoscTekstury = glebokoscPomiaru;
			wysokoscTekstury = b_skan_rozmiar_y;
			break;
		}

		for (unsigned int i = 0, stop = ileTekstur; i != stop; ++i) {

			glBindTexture(GL_TEXTURE_2D, indeksyTekstur[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA, szerokoscTekstury, wysokoscTekstury, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		}

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	GLuint* inicjalizacjaTekstur() {

		//przemyœleæ czy chcemy mieæ mo¿liwoœæ tworzenia samych bskanów
		GLuint* indeksy = new GLuint[liczbaTekstur()];
		glGenTextures(liczbaTekstur(), indeksy);
		parametryTekstur(indeksy, liczba_B_skanow, BSKAN);
		parametryTekstur(indeksy + liczba_B_skanow, liczba_Przekrojow_Poprzecznych, PRZEKROJ_POPRZECZNY);
		parametryTekstur(indeksy + liczba_B_skanow + liczba_Przekrojow_Poprzecznych, liczba_Przekrojow_Poziomych, PRZEKROJ_POZIOMY);
		return indeksy;
	}

	void init() {

		indeksy_tekstur = inicjalizacjaTekstur();
		zainicjalizowane = true;
	}

	GLuint* indeksyTekstur() { return indeksy_tekstur; }

	~TomogramTekstury() {

		if (zainicjalizowane) {

			glBindTexture(GL_TEXTURE_2D, 0);
			glDeleteTextures(liczbaTekstur(), indeksy_tekstur);
			delete[] indeksy_tekstur;
		}
	}
};