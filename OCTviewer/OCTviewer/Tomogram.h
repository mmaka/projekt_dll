#pragma once
#include"glew.h"
#include"OCTviewer.h"
#include<vector>


class TomogramTekstury {

	std::vector<GLuint> indeksyTekstur;
	int wyswietlanie2D_indeksBskanu{ 0 };
	void ustawienieParametrowTekstur(unsigned int idx_start, unsigned int idx_stop, unsigned int szerokoscTekstury, unsigned int wysokoscTekstury) const {

		for (unsigned int i = idx_start, stop = idx_stop; i != stop; ++i) {

			glBindTexture(GL_TEXTURE_2D, indeksyTekstur[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, szerokoscTekstury, wysokoscTekstury, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		}

		glBindTexture(GL_TEXTURE_2D, 0);

	}
	
public:

	void inicjalizacjaTekstur(visualizationParams params) {

		unsigned int ileTekstur = 0;
		unsigned int szerokoscTekstury = 0;
		unsigned int wysokoscTekstury = 0;

		switch (params.typ) {

		case WIZUALIZACJA::TYP_2D:
			ileTekstur = params.liczbaBskanow;
			indeksyTekstur.resize(ileTekstur);
			glGenTextures(ileTekstur, indeksyTekstur.data());
			szerokoscTekstury = params.bscanSize_px;
			wysokoscTekstury = params.ascanSize_px;
			ustawienieParametrowTekstur(0, ileTekstur, szerokoscTekstury, wysokoscTekstury);
			break;

		case WIZUALIZACJA::TYP_3D:
			ileTekstur = params.liczbaBskanow + params.liczbaPrzekrojowPoprzecznych + params.liczbaPrzekrojowPoziomych;
			indeksyTekstur.resize(ileTekstur);
			glGenTextures(ileTekstur, indeksyTekstur.data());
			szerokoscTekstury = params.bscanSize_px;
			wysokoscTekstury = params.ascanSize_px;
			ustawienieParametrowTekstur(0, params.liczbaBskanow, szerokoscTekstury, wysokoscTekstury);
			szerokoscTekstury = params.bscanSize_px;
			wysokoscTekstury = params.depth_px;
			ustawienieParametrowTekstur(params.liczbaBskanow, params.liczbaBskanow+params.liczbaPrzekrojowPoprzecznych, szerokoscTekstury, wysokoscTekstury);
			szerokoscTekstury = params.depth_px;
			wysokoscTekstury = params.ascanSize_px;
			ustawienieParametrowTekstur(params.liczbaBskanow + params.liczbaPrzekrojowPoprzecznych, params.liczbaBskanow + params.liczbaPrzekrojowPoprzecznych+params.liczbaPrzekrojowPoziomych, szerokoscTekstury, wysokoscTekstury);
			break;
		}
	}

	std::vector<GLuint>& tablicaIndeksow() { return indeksyTekstur; }
	GLuint indeks(size_t idx) const { return indeksyTekstur[idx]; }
	GLuint kolejnyIndeksBskanu() { 

		if (wyswietlanie2D_indeksBskanu != indeksyTekstur.size()-1)
			++wyswietlanie2D_indeksBskanu;
		else
			wyswietlanie2D_indeksBskanu = 0;
		
		return indeksyTekstur[wyswietlanie2D_indeksBskanu];
	}
	GLuint poprzedniIndeksBskanu() {
/*
		if (wyswietlanie2D_indeksBskanu != 0)
			--wyswietlanie2D_indeksBskanu;
		else
			wyswietlanie2D_indeksBskanu = indeksyTekstur.size() - 1;
			*/

		--wyswietlanie2D_indeksBskanu;
		wyswietlanie2D_indeksBskanu %= indeksyTekstur.size();


		return indeksyTekstur[wyswietlanie2D_indeksBskanu];
	}
	void usunTekstury() {

		glBindTexture(GL_TEXTURE_2D, 0);
		glDeleteTextures(indeksyTekstur.size(), indeksyTekstur.data());

	}
};