#pragma once
#include <windows.h>

bool CzyPlikIstnieje(const char* filename);
unsigned long* WczytajObrazZPlikuBitmap(HWND uchwytOkna, char* nazwaPliku, int& szerokoscObrazu, int& wysokoscObrazu, bool czytajZZasobow, unsigned char alfa);
