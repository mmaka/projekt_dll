#pragma once
#include"Okno.h"
#include "glew.h"
#include "wglew.h"
#include "Werteks.h"
#include "math.h"
#include "Wektor.h"
#include<sstream>
#include"Macierz.h"
#include"Shadery.h"


LRESULT CALLBACK __stdcall WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);


LRESULT __stdcall Okno::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {

	switch (message) {

	case WM_SIZE:
		RECT rect;
		GetClientRect(hWnd, &rect);
		szerokoscObszaruUzytkownika = rect.right - rect.left;
		wysokoscObszaruUzytkownika = rect.bottom - rect.top;
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return (DefWindowProc(hWnd, message, wParam, lParam));
	}

	return 0L;
}

bool Okno::ZmianaRozdzieloczosci(long szerokosc, long wysokosc, long glebiaKolorow) const {

	DEVMODE dmScreenSettings;
	memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
	dmScreenSettings.dmSize = sizeof(dmScreenSettings);
	dmScreenSettings.dmPelsWidth = szerokosc;
	dmScreenSettings.dmPelsHeight = wysokosc;
	dmScreenSettings.dmBitsPerPel = glebiaKolorow;
	dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;
	return ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) == DISP_CHANGE_SUCCESSFUL;

}


bool Okno::Inicjuj(HINSTANCE uchwytAplikacji, POINT polozenieOkna, POINT rozmiarOkna, bool trybPelnoekranowy, bool zmianaRozdzielczosci) {

	char nazwaOkna[] = "OCTviewer";

	WNDCLASSEX wc;
	wc.cbSize = sizeof(wc);
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wc.lpfnWndProc = (WNDPROC)::WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = uchwytAplikacji;
	wc.hIcon = NULL;
	wc.hIconSm = NULL;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = NULL;
	wc.lpszMenuName = NULL;
	wc.lpszClassName = nazwaOkna;

	if (RegisterClassEx(&wc) == 0) return false;


	DWORD stylOkna = WS_OVERLAPPEDWINDOW;

	if (trybPelnoekranowy) {

		stylOkna = WS_POPUP;
		polozenieOkna.x = 0;
		polozenieOkna.y = 0;

		if (zmianaRozdzielczosci) {

			rozmiarOkna.x = 1024;
			rozmiarOkna.y = 768;
			if (!ZmianaRozdzieloczosci(rozmiarOkna.x, rozmiarOkna.y)) return false;

		}
		else {

			RECT rozmiarEkranu;
			GetWindowRect(GetDesktopWindow(), &rozmiarEkranu);
			rozmiarOkna.x = rozmiarEkranu.right - rozmiarEkranu.left;
			rozmiarOkna.y = rozmiarEkranu.bottom - rozmiarEkranu.top;

		}

	}
	uchwytOkna = CreateWindow(nazwaOkna, nazwaOkna, stylOkna, polozenieOkna.x, polozenieOkna.y, rozmiarOkna.x, rozmiarOkna.y, NULL, NULL, uchwytAplikacji, NULL);

	if (uchwytOkna == NULL) return false;

	ShowWindow(uchwytOkna, SW_SHOW);
	UpdateWindow(uchwytOkna);

	return true;
}

WPARAM Okno::Uruchom() {

	MSG msg;
	while (GetMessage(&msg, NULL, 0, 0)) {

		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return msg.wParam;
}


bool OknoGL::UstalFomatPikseli(HDC uchwytDC) const {

	PIXELFORMATDESCRIPTOR opisFormatuPikseli;
	ZeroMemory(&opisFormatuPikseli, sizeof(opisFormatuPikseli));
	opisFormatuPikseli.nVersion = 1;
	opisFormatuPikseli.dwFlags = PFD_SUPPORT_OPENGL | PFD_DRAW_TO_WINDOW | PFD_DOUBLEBUFFER;
	opisFormatuPikseli.iPixelType = PFD_TYPE_RGBA;
	opisFormatuPikseli.cColorBits = 32;
	opisFormatuPikseli.cDepthBits = 32;
	opisFormatuPikseli.iLayerType = PFD_MAIN_PLANE;
	int formatPikseli = ChoosePixelFormat(uchwytDC, &opisFormatuPikseli);
	if (formatPikseli == 0) return false;
	if (!SetPixelFormat(uchwytDC, formatPikseli, &opisFormatuPikseli)) return false;

	return true;
}

bool OknoGL::InicjujWGL(HWND uchwytOkna) {

	uchwytDC = ::GetDC(uchwytOkna);

	if (!UstalFomatPikseli(uchwytDC)) return false;

	HGLRC uchwytTymczasowegoRC = wglCreateContext(uchwytDC);
	if (uchwytTymczasowegoRC == NULL) return false;

	if (!wglMakeCurrent(uchwytDC, uchwytTymczasowegoRC)) return false;

	GLenum err = glewInit();
	if (GLEW_OK != err) {

		MessageBox(NULL, "Inicjacja biblioteki GLEW nie powiodla sie", "Aplikacja OpenGL", MB_OK | MB_ICONERROR);
		return false;
	}

	const int major_min = 4;
	const int minor_min = 3;
	int major, minor;
	glGetIntegerv(GL_MAJOR_VERSION, &major);
	glGetIntegerv(GL_MINOR_VERSION, &minor);

	if (major < major_min || (major == major_min && minor < minor_min)) {

		MessageBox(NULL, "Wersja OpenGL jest niewystrczajaca", "Aplikacja OpenGL", MB_OK | MB_ICONERROR);
		return false;
	}

	int atrybuty[] = {

		WGL_CONTEXT_MAJOR_VERSION_ARB,major_min,
		WGL_CONTEXT_MINOR_VERSION_ARB,minor_min,
		//	WGL_CONTEXT_FLAGS_ARB,WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,//WGL_CONTEXT_DEBUG_BIT_ARB,//WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB, <-- dodane WGL_CONTEXT_DEBUG_BIT_ARG
		WGL_CONTEXT_PROFILE_MASK_ARB,WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
		0
	};

	uchwytRC = wglCreateContextAttribsARB(uchwytDC, 0, atrybuty);

	if (uchwytRC == NULL) return false;
	if (!wglMakeCurrent(uchwytDC, uchwytRC)) return false;

	wglDeleteContext(uchwytTymczasowegoRC);
	return true;
}


void OknoGL::UsunWGL() {

	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(uchwytRC);
	::ReleaseDC(uchwytOkna, uchwytDC);
}

LRESULT __stdcall OknoGL::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {

	long wynik = Okno::WndProc(hWnd, message, wParam, lParam);

	static POINT poprzedniaPozycjaKursoraMyszyLMB = { -1,-1 };
	static POINT poprzedniaPozycjaKursoraMyszyRMB = { -1,-1 };

	const bool swobodneObrotyKameryMozliwe = true;
	const int identyfikatorTimeraSwobodnychObrotowKamery = 1;
	const int okresTimeraSwobodnychObrotowKamery = 50;

	const int identyfikatorTimeraRysowania = 2;
	const int okresTimeraRysowania = 10;

	switch (message) {

	case WM_CREATE:
		if (!InicjujWGL(hWnd)) {

			MessageBox(NULL, "Pobranie kontekstu renderowania nie powodlo sie", "OCTviewer", MB_OK | MB_ICONERROR);
			return EXIT_FAILURE;
		}

		idProgramuShaderow = PrzygotujShadery(vertexShader, fragmentShader, false);

		if (idProgramuShaderow == NULL) {

			MessageBox(NULL, "Przygotowanie shaderow nie powiodlo sie", "OCTviewer", MB_OK | MB_ICONERROR);
			exit(EXIT_FAILURE);
		}

		UmiescInformacjeNaPaskuTytulu(hWnd);
		tekstury = new TomogramTekstury(parametryWyswietlania);
		tekstury->init();
		liczbaPrzekrojow = przygotujPrzekroje();
		cudaTekstury->init();
		ustanowienieWspolpracyCudaOpenGL(tekstury->indeksyTekstur(), cudaTekstury->cudaArray(), parametryWyswietlania.liczbaBskanow+ parametryWyswietlania.liczbaPrzekrojowPoprzecznych  + parametryWyswietlania.liczbaPrzekrojowPoziomych);
		cudaTekstury->przygotowanieTekstur();
		cudaTekstury->pobierzDaneCPU();
		cudaTekstury->tworzPrzekroje();

		UstawienieSceny();
		if (swobodneObrotyKameryMozliwe) {

			if (SetTimer(hWnd, identyfikatorTimeraSwobodnychObrotowKamery, okresTimeraSwobodnychObrotowKamery, NULL) == 0)
				MessageBox(hWnd, "Nie udalo sie ustawic timera swobodnych obrotow kamery", "Blad", MB_OK | MB_ICONERROR);
		}

		if (SetTimer(hWnd, identyfikatorTimeraRysowania, okresTimeraRysowania, NULL) == 0)
			MessageBox(hWnd, "nie udalo sie ustawic timera rysowanie", "Blad", MB_OK | MB_ICONERROR);


		QueryPerformanceFrequency(&countPerSec);
		QueryPerformanceCounter(&tim1);

		break;
	case WM_SIZE:
		UstawienieSceny();
		break;
	case WM_PAINT:
		RysujScene();
		ValidateRect(hWnd, NULL);
		break;
	case WM_KEYDOWN:
		ObslugaKlawiszy(wParam);
		break;
	case WM_MOUSEMOVE:
		if (wParam & MK_LBUTTON)
			ObliczaniePrzesunieciaMyszy(lParam, 3.0f, poprzedniaPozycjaKursoraMyszyLMB, &OknoGL::ObslugaMyszyZWcisnietymLewymPrzyciskiem);

		if (wParam & MK_RBUTTON)
			ObliczaniePrzesunieciaMyszy(lParam, 3.0f, poprzedniaPozycjaKursoraMyszyRMB, &OknoGL::ObslugaMyszyZWcisnietymPrawymPrzyciskiem);
		break;
	case WM_MOUSEWHEEL:
		ObslugaRolkiMyszy(wParam);
		break;
	case WM_LBUTTONUP:
		poprzedniaPozycjaKursoraMyszyLMB = { -1,-1 };
		break;
	case WM_RBUTTONUP:
		poprzedniaPozycjaKursoraMyszyRMB = { -1,-1 };
		break;
	case WM_TIMER:
		switch (wParam) {

		case identyfikatorTimeraSwobodnychObrotowKamery:
			if (swobodneObrotyKameryAktywne) SwobodneObrotyKamery(false);
			break;
		case identyfikatorTimeraRysowania:
			RysujScene();
			break;
		}
		wynik = 0;
		break;
	case WM_DESTROY:
	//	UsunTekstury();
		cudaTekstury->sprzatanie();
		delete tekstury;
		UsunAktorow();
		UsunWGL();
		KillTimer(uchwytOkna, identyfikatorTimeraSwobodnychObrotowKamery);
		break;
	}

	return wynik;
}


void OknoGL::UmiescInformacjeNaPaskuTytulu(HWND uchwytOkna) {

	char bufor[256];
	GetWindowText(uchwytOkna, bufor, 256);
	const GLubyte* wersja = glGetString(GL_VERSION);
	strcat_s(bufor, " | OpenGL ");
	strcat_s(bufor, (char*)wersja);
	const GLubyte* dostawca = glGetString(GL_VENDOR);
	strcat_s(bufor, " | ");
	strcat_s(bufor, (char*)dostawca);
	const GLubyte* kartaGraficzna = glGetString(GL_RENDERER);
	strcat_s(bufor, " | ");
	strcat_s(bufor, (char*)kartaGraficzna);
	const GLubyte* wersjaGLEW = glewGetString(GLEW_VERSION);
	strcat_s(bufor, " | GLEW ");
	strcat_s(bufor, (char*)wersjaGLEW);
	const GLubyte* wersjaGLSL = glGetString(GL_SHADING_LANGUAGE_VERSION);
	strcat_s(bufor, " | GLSL ");
	strcat_s(bufor, (char*)wersjaGLSL);

	SetWindowText(uchwytOkna, bufor);
}

void OknoGL::UstawienieSceny(bool rzutowanieIzometryczne) {

	glViewport(0, 0, szerokoscObszaruUzytkownika, wysokoscObszaruUzytkownika);
	glEnable(GL_TEXTURE_2D);

	switch (parametryWyswietlania.typ) {

	case WIZUALIZACJA::TYP_2D:

		break;
	case WIZUALIZACJA::TYP_3D:
		glEnable(GL_BLEND);
		glDisable(GL_DEPTH_TEST);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		break;

	}

	//Parametry shadera 
	GLint parametrUwzglednijKolorWerteksu = glGetUniformLocation(idProgramuShaderow, "UwzglednijKolorWerteksu");
	glUniform1i(parametrUwzglednijKolorWerteksu, true);
	GLint parametrKolor = glGetUniformLocation(idProgramuShaderow, "Kolor");
	float kolor[4] = { 0.0,1.0,0.0,1.0 };
	glUniform4fv(parametrKolor, 1, kolor);

	GLint parametrPrzezroczystosc = glGetUniformLocation(idProgramuShaderow, "przezroczystosc");
	float przezroczystosc = 0.02f;
	glUniform1f(parametrPrzezroczystosc, przezroczystosc);

	GLint parametrMacierzSwiata = glGetUniformLocation(idProgramuShaderow, "macierzSwiata");
	//macierzSwiata.ZwiazZIdentyfikatorem(parametrMacierzSwiata, true);

	GLint parametrMacierzWidoku = glGetUniformLocation(idProgramuShaderow, "macierzWidoku");
	macierzWidoku.UstawJednostkowa();
	macierzWidoku *= Macierz4::Przesuniecie(0, 0, -5);
	//	macierzWidoku.ZwiazZIdentyfikatorem(parametrMacierzWidoku, true);

	GLint parametrMacierzRzutowania = glGetUniformLocation(idProgramuShaderow, "macierzRzutowania");
	//	macierzRzutowania.ZwiazZIdentyfikatorem(parametrMacierzRzutowania, false);


	GLint parametrMVP = glGetUniformLocation(idProgramuShaderow, "mvp");
	MVP.ZwiazZIdentyfikatorem(parametrMVP, true);


	float wsp = wysokoscObszaruUzytkownika / (float)szerokoscObszaruUzytkownika;
	if (!rzutowanieIzometryczne)
		macierzRzutowania.UstawRzutPerspektywiczny(-1.0f, 1.0f, wsp*-1.0f, wsp*1.0f, 1.0f, 10.0f);
	else
		macierzRzutowania.UstawRzutIzometryczny(-1.0f, 1.0f, wsp*-1.0f, wsp*1.0f, 1.0f, 10.0f);

	VP.Ustaw(macierzRzutowania);
	VP.PomnozZPrawej(macierzWidoku);
	
}


void OknoGL::RysujScene() {

	glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT); //sprawdz czy GL_DEPTH_BUFFER_BIT jest w ogole potrzebny
	RysujAktorow();//uwaga na wiele watkow str 242 Nowoczesny OpenGL
	SwapBuffers(uchwytDC);
	glFinish();
	QueryPerformanceCounter(&tim2);
	double time = (double)(tim2.QuadPart - tim1.QuadPart) / countPerSec.QuadPart * 1000;
	tim1 = tim2;
	std::stringstream s;
	s << "Czas: " << time << " milisekund | FPS: " << 1 / (time / 1000) << std::endl;
	SetWindowText(uchwytOkna, s.str().c_str());

}

void PokazKomunikat(const char* tekst, UINT typ = 0) {

	MessageBoxA(NULL, tekst, "Aplikacja OpenGL - Kompilacja shaderow", MB_OK | typ);
}

unsigned int OknoGL::PrzygotujShadery(const char* vs, const char* fs, bool trybDebugowania) {

	//KomKompilacja shadera werteksow
	GLuint idShaderaWerteksow = KompilujShader(vs, GL_VERTEX_SHADER, trybDebugowania);

	if (idShaderaWerteksow == NULL) {

		PokazKomunikat("Kompilacja shadera werteksow nie powiodla sie", MB_ICONERROR);
		return NULL;

	}
	else if (trybDebugowania) PokazKomunikat("Kompilacja shadera werteksow zakonczyla sie sukcesem", MB_ICONINFORMATION);


	//Kompilacja shadera fragmentow
	GLuint idShaderaFragmentow = KompilujShader(fs, GL_FRAGMENT_SHADER, trybDebugowania);

	if (idShaderaFragmentow == NULL) {

		PokazKomunikat("Kompilacja shadera fragmentow nie powodila sie", MB_ICONERROR);
		return NULL;

	}
	else if (trybDebugowania) PokazKomunikat("Kompilacja shadera fragmentow zakonczyla sie sukcesem", MB_ICONINFORMATION);

	//Tworzenie obiektu programu
	GLuint idProgramu = glCreateProgram();

	//Przylaczenie shaderow
	glAttachShader(idProgramu, idShaderaWerteksow);
	glAttachShader(idProgramu, idShaderaFragmentow);

	//Linkowanie
	glLinkProgram(idProgramu);

	//Weryfikacja linkowania
	GLint powodzenie;
	glGetProgramiv(idProgramu, GL_LINK_STATUS, &powodzenie);

	if (!powodzenie) {

		const int maxInfoLogSize = 2048;
		GLchar infoLog[maxInfoLogSize];
		glGetProgramInfoLog(idProgramu, maxInfoLogSize, NULL, infoLog);
		char komunikat[maxInfoLogSize + 64] = "Uwaga! Linkowanie programu shaderow nie powiodlo sie:\n";
		strcat_s(komunikat, (char*)infoLog);
		PokazKomunikat(komunikat, MB_ICONERROR);
		return NULL;

	}
	else if (trybDebugowania) PokazKomunikat("Linkowanie programu shaderow powiodlo sie", MB_ICONINFORMATION);

	//Walidacja programu
	glGetProgramiv(idProgramu, GL_VALIDATE_STATUS, &powodzenie);

	if (!powodzenie) {

		const int maxInfoLogSize = 2048;
		GLchar infoLog[maxInfoLogSize];
		glGetProgramInfoLog(idProgramu, maxInfoLogSize, NULL, infoLog);
		char komunikat[maxInfoLogSize + 64] = "Uwaga! Walidacja programu shaderow nie powiodla sie:\n";
		strcat_s(komunikat, (char*)infoLog);
		PokazKomunikat(komunikat, MB_ICONERROR);
		return NULL;

	}
	else if (trybDebugowania) PokazKomunikat("Walidacja programu shaderow powiodla sie", MB_ICONINFORMATION);

	//Uzycie programu
	glUseProgram(idProgramu);
	//Usuwanie niepotrzebnych obiektow shadera
	glDeleteShader(idShaderaWerteksow);
	glDeleteShader(idShaderaFragmentow);

	return idProgramu;
}


unsigned int OknoGL::KompilujShader(const char* shader, GLenum typ, bool trybDebugowania) {

	const int maksymalnaWielkoscKodu = 65535;

	if(trybDebugowania) PokazKomunikat(shader);
	//Tworzenie obiektu shadera
	GLuint idShadera = glCreateShader(typ);
	if (idShadera == NULL) return NULL;

	//Dostarczenie zrodla do obiektu shadera
	const GLchar* zrodlo[1];
	zrodlo[0] = shader;
	glShaderSource(idShadera, 1, zrodlo, NULL);

	//Kompilacja shadera
	glCompileShader(idShadera);

	//Weryfikacja kompilacji
	GLint powodzenie;
	glGetShaderiv(idShadera, GL_COMPILE_STATUS, &powodzenie);

	if (!powodzenie) {

		const int maxInfoLogSize = 2048;
		GLchar infoLog[maxInfoLogSize];
		glGetShaderInfoLog(idShadera, maxInfoLogSize, NULL, infoLog);
		char komunikat[maxInfoLogSize + 64] = "Uwaga! Kompilacja shadera nie powodla sie:\n";
		strcat_s(komunikat, (char*)infoLog);
		PokazKomunikat(komunikat, MB_ICONERROR);
		return NULL;

	}
	else if (trybDebugowania) PokazKomunikat("Kompilacja shadera zakonczyla sie sukcesem", MB_ICONINFORMATION);

	return idShadera;
}

void OknoGL::ObslugaKlawiszy(WPARAM wParam) {

	const float kat = 5.0f;
	Macierz4 m = Macierz4::Jednostkowa;
	switch (wParam) {

	case 27:
		SendMessage(uchwytOkna, WM_CLOSE, 0, 0);
		break;
	case VK_F6:

		break;
	case VK_F5:
	{
		bool komunikatNaPaskuTytulu = true;
		switch (trybKontroliKamery) {

		case tkkFPP:
			trybKontroliKamery = tkkTPP;
			if (komunikatNaPaskuTytulu)
				SetWindowText(uchwytOkna, "Tryb kontroli kamery: TPP");

			break;

		case tkkTPP:
			trybKontroliKamery = tkkArcBall;
			if (komunikatNaPaskuTytulu)
				SetWindowText(uchwytOkna, "Tryb kontroli kamery: ArcBall");

			break;

		case tkkArcBall:
			trybKontroliKamery = tkkModel;
			if (komunikatNaPaskuTytulu)
				SetWindowText(uchwytOkna, "Tryb kontroli kamery: Model");

			break;

		case tkkModel:
			trybKontroliKamery = tkkFPP;
			if (komunikatNaPaskuTytulu)
				SetWindowText(uchwytOkna, "tryb kontroli kamery: FPP");

			break;
		}
	}
	MessageBeep(MB_OK);
	break;
	case 'A':
	case VK_LEFT:
		m = Macierz4::ObrotY(kat);
		break;
	case 'D':
	case VK_RIGHT:
		m = Macierz4::ObrotY(-kat);
		break;
	case 'W':
	case VK_UP:
		m = Macierz4::ObrotX(kat);
		break;
	case 'S':
	case VK_DOWN:
		m = Macierz4::ObrotX(-kat);
		break;
	case VK_OEM_COMMA:
		m = Macierz4::ObrotZ(kat);
		break;
	case VK_OEM_PERIOD:
		m = Macierz4::ObrotZ(-kat);
		break;
	case VK_OEM_MINUS:
		m = Macierz4::Przesuniecie(0, 0, 0.1f);
		break;
	case VK_OEM_PLUS:
		m = Macierz4::Przesuniecie(0, 0, -0.1f);
		break;
	}

	ModyfikujPolozenieKamery(m);
	RysujScene();

}
//zwraca polozenie kamery w ukladzie sceny
Wektor3 OknoGL::PobierzPolozenieKamery(bool pominObroty) const {

	Wektor3 w3;
	if (pominObroty) {

		Wektor4 w4 = macierzWidoku.KopiaKolumny(3);
		for (int i = 0; i < 3; ++i) w3[i] = -w4[i];

	}
	else {

		Wektor4 w4 = macierzWidoku.Odwrotna().KopiaKolumny(3);
		for (int i = 0; i < 3; ++i) w3[i] = w4[i];

	}

	return w3;
}

float OknoGL::OdlegloscKamery() const {

	return PobierzPolozenieKamery().Dlugosc();
}

void OknoGL::ModyfikujPolozenieKamery(Macierz4 macierzPrzeksztalcenia) {

	switch (trybKontroliKamery) {

	case tkkFPP:
		macierzWidoku.PomnozZLewej(macierzPrzeksztalcenia);
		break;
	case tkkTPP:
	{
		const float odlegloscCentrumOdKamery = 3.0f;
		macierzWidoku.PomnozZLewej(Macierz4::Przesuniecie(0, 0, -odlegloscCentrumOdKamery)*macierzPrzeksztalcenia*Macierz4::Przesuniecie(0, 0, odlegloscCentrumOdKamery));

	}
	break;
	case tkkArcBall:
	{
		Wektor3 polozenieKamery = PobierzPolozenieKamery(true);
		macierzWidoku.PomnozZLewej(Macierz4::Przesuniecie(-polozenieKamery)*macierzPrzeksztalcenia*Macierz4::Przesuniecie(polozenieKamery));
	}
	break;
	case tkkModel:
		macierzWidoku.PomnozZPrawej(macierzPrzeksztalcenia);
	}

	VP.Ustaw(macierzRzutowania);
	VP.PomnozZPrawej(macierzWidoku);
}

void OknoGL::ObliczaniePrzesunieciaMyszy(const LPARAM lParam, const float prog, POINT& poprzedniaPozycjaKursoraMyszy, TypMetodyObslugujacejPrzesuniecieMyszy MetodaObslugujacaPrzesuniecieMyszy) {

	if (poprzedniaPozycjaKursoraMyszy.x == -1 && poprzedniaPozycjaKursoraMyszy.y == -1) {
		poprzedniaPozycjaKursoraMyszy = { LOWORD(lParam), HIWORD(lParam) };
		return;
	}

	POINT biezacaPozycjaKursoraMyszy = { LOWORD(lParam),HIWORD(lParam) };
	POINT przesuniecie;
	przesuniecie.x = biezacaPozycjaKursoraMyszy.x - poprzedniaPozycjaKursoraMyszy.x;
	przesuniecie.y = biezacaPozycjaKursoraMyszy.y - poprzedniaPozycjaKursoraMyszy.y;

	float przesuniecieKursoraMyszy = (float)sqrt(przesuniecie.x*przesuniecie.x + przesuniecie.y*przesuniecie.y);

	if (przesuniecieKursoraMyszy > prog) {
		//prog w pikselach - zapobiega zbyt czestym zmianom - utrata dokladnosci

		(this->*MetodaObslugujacaPrzesuniecieMyszy)(biezacaPozycjaKursoraMyszy, przesuniecie);
		poprzedniaPozycjaKursoraMyszy = biezacaPozycjaKursoraMyszy;

	}
}

void OknoGL::ObslugaMyszyZWcisnietymLewymPrzyciskiem(POINT biezacaPozycjaKursoraMyszy, POINT przesuniecieKursoraMyszy) {

	const float wspX = 360.0f / szerokoscObszaruUzytkownika;
	const float wspY = 360.0f / wysokoscObszaruUzytkownika;
	float dx = przesuniecieKursoraMyszy.x *wspX;
	float dy = przesuniecieKursoraMyszy.y *wspY;
	Macierz4 m = Macierz4::ObrotXYZ(dy, dx, 0);
	ModyfikujPolozenieKamery(m);
	RysujScene();

	SwobodneObrotyKamery(true, dx, dy, 0.95f);
}

void OknoGL::ObslugaMyszyZWcisnietymPrawymPrzyciskiem(POINT biezacaPozycjaKursoraMyszy, POINT przesuniecieKursoraMyszy) {

	//skalowanie
	const float wspX = 5.0f / szerokoscObszaruUzytkownika;
	const float wspY = 5.0f / wysokoscObszaruUzytkownika;
	float dx = przesuniecieKursoraMyszy.x * wspX;
	float dy = przesuniecieKursoraMyszy.y * wspY;

	Macierz4 m = Macierz4::Przesuniecie(dx, -dy, 0);
	ModyfikujPolozenieKamery(m);
	RysujScene();

}

void OknoGL::ObslugaRolkiMyszy(WPARAM wParam) {

	const float czulosc = 10.0f;
	short zmianaPozycjiRolki = (short)HIWORD(wParam);
	float przesuniecie = zmianaPozycjiRolki / abs(zmianaPozycjiRolki) / czulosc;
	ModyfikujPolozenieKamery(Macierz4::Przesuniecie(0, 0, przesuniecie));
	RysujScene();
}

void OknoGL::SwobodneObrotyKamery(const bool inicjacja, const float poczatkowe_dx, const float poczatkowe_dy, const float wspolczynnikWygaszania) {

	static float dx = 0;
	static float dy = 0;
	static float wsp = 0;
	if (inicjacja) {

		swobodneObrotyKameryAktywne = true;
		dx = poczatkowe_dx;
		dy = poczatkowe_dy;
		wsp = wspolczynnikWygaszania;
		if (wsp < 0) wsp = 0;
		if (wsp > 1) wsp = 1;

		return;

	}
	else {

		dx *= wsp;
		dy *= wsp;
		if (fabs(dx) < 0.001f && fabs(dy) < 0.001f)
			swobodneObrotyKameryAktywne = false;

		Macierz4 m = Macierz4::ObrotXYZ(dy, dx, 0);
		ModyfikujPolozenieKamery(m);
		RysujScene();
	}

}

unsigned int OknoGL::przygotujPrzekroje() {

	GLuint atrybutPolozenie = glGetAttribLocation(idProgramuShaderow, "polozenie_in");
	if (atrybutPolozenie == (GLuint)-1) atrybutPolozenie = 0;

	GLuint atrybutWspolrzedneTeksturowania = glGetAttribLocation(idProgramuShaderow, "wspTekstur_in");
	if (atrybutWspolrzedneTeksturowania == (GLuint)-1) atrybutWspolrzedneTeksturowania = 2;

	GLuint atrybutKolor = glGetAttribLocation(idProgramuShaderow, "kolor_in");
	if (atrybutKolor == (GLuint)-1) atrybutKolor = 3;
	unsigned int liczbaPrzekrojow = 0;
	GLuint *indeksyTekstur = tekstury->indeksyTekstur();
	switch (parametryWyswietlania.typ) {

	case WIZUALIZACJA::TYP_2D:
		liczbaPrzekrojow = 1;
		przekroje = new CrossSection*[liczbaPrzekrojow];
		przekroje[0] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, parametryWyswietlania.x_mm*parametryWyswietlania.xSizeScale, parametryWyswietlania.y_mm*parametryWyswietlania.ySizeScale);
		przekroje[0]->IndeksTekstury = indeksyTekstur[0];
		break;
	case WIZUALIZACJA::TYP_3D:
		liczbaPrzekrojow = parametryWyswietlania.liczbaBskanow+ parametryWyswietlania.liczbaPrzekrojowPoprzecznych  + parametryWyswietlania.liczbaPrzekrojowPoziomych;
		przekroje = new CrossSection*[liczbaPrzekrojow];

		float rozmiarX = parametryWyswietlania.x_mm*parametryWyswietlania.xSizeScale;
		float rozmiarY = parametryWyswietlania.y_mm*parametryWyswietlania.ySizeScale;
		float rozmiarZ = parametryWyswietlania.z_mm*parametryWyswietlania.zSizeScale;

		float bskany_krok = rozmiarZ / (parametryWyswietlania.liczbaBskanow - 1);
		float przekroje_poprzeczne_krok = rozmiarY / (parametryWyswietlania.liczbaPrzekrojowPoprzecznych - 1);
		float przekroje_poziome_krok = rozmiarX / (parametryWyswietlania.liczbaPrzekrojowPoziomych - 1);

		for (size_t i = 0, end = parametryWyswietlania.liczbaBskanow; i != end; ++i) {

			przekroje[i] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, rozmiarX, rozmiarY);
			przekroje[i]->MacierzSwiata = Macierz4::Przesuniecie(0.0f, 0.0f, rozmiarZ / 2 - bskany_krok*(end - 1 - i));
			przekroje[i]->IndeksTekstury = indeksyTekstur[i];
		}

		for (size_t i = 0, end = parametryWyswietlania.liczbaPrzekrojowPoprzecznych; i != end; ++i) {

			przekroje[i + parametryWyswietlania.liczbaBskanow] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, rozmiarX, rozmiarZ);
			przekroje[i + parametryWyswietlania.liczbaBskanow]->MacierzSwiata = Macierz4::Przesuniecie(0.0f, przekroje_poprzeczne_krok*(i) - rozmiarY / 2, 0) *Macierz4::ObrotX(90);
			//przekroje[i + parametryWyswietlania.liczbaBskanow]->MacierzSwiata = Macierz4::Przesuniecie(2.5f, przekroje_poprzeczne_krok*(end - 1 - i) - rozmiarY / 2, 0) *Macierz4::ObrotX(90);
			przekroje[i + parametryWyswietlania.liczbaBskanow]->IndeksTekstury = indeksyTekstur[i + parametryWyswietlania.liczbaBskanow];

			//przekroje[i] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, rozmiarX, rozmiarZ);
			//przekroje[i]->MacierzSwiata = Macierz4::Przesuniecie(0.0f, przekroje_poprzeczne_krok*(end - 1 - i) - rozmiarY / 2, -rozmiarZ / 2) *Macierz4::ObrotX(-270);
			//przekroje[i]->IndeksTekstury = indeksyTekstur[i];

		}
		
		for (size_t i = 0, end = parametryWyswietlania.liczbaPrzekrojowPoziomych; i != end; ++i) {

			przekroje[i + parametryWyswietlania.liczbaBskanow + parametryWyswietlania.liczbaPrzekrojowPoprzecznych] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, rozmiarZ, rozmiarY);
			//przekroje[i + parametryWyswietlania.liczbaBskanow + parametryWyswietlania.liczbaPrzekrojowPoprzecznych]->MacierzSwiata = Macierz4::Przesuniecie(i*przekroje_poziome_krok - rozmiarX / 2, 0.0f, -((parametryWyswietlania.liczbaBskanow - 1)*bskany_krok - rozmiarZ / 2))*Macierz4::ObrotY(270);
			przekroje[i + parametryWyswietlania.liczbaBskanow + parametryWyswietlania.liczbaPrzekrojowPoprzecznych]->MacierzSwiata = Macierz4::Przesuniecie(i*przekroje_poziome_krok - rozmiarX / 2, 0.0f, -((parametryWyswietlania.liczbaBskanow - 1)*bskany_krok) + rozmiarZ)*Macierz4::ObrotY(270);
			przekroje[i + parametryWyswietlania.liczbaBskanow + parametryWyswietlania.liczbaPrzekrojowPoprzecznych]->IndeksTekstury = indeksyTekstur[i + parametryWyswietlania.liczbaBskanow + parametryWyswietlania.liczbaPrzekrojowPoprzecznych];

			//		przekroje[i] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, rozmiarZ, rozmiarY);
			//		przekroje[i]->MacierzSwiata = Macierz4::Przesuniecie(i*przekroje_poziome_krok - rozmiarX / 2, 0.0f, -((parametryWyswietlania.liczbaBskanow - 1)*bskany_krok - rozmiarZ / 2))*Macierz4::ObrotY(270);
			//		przekroje[i]->IndeksTekstury = indeksyTekstur[i];
		}
	
		break;
	}
	
	return liczbaPrzekrojow;
}

/*
unsigned int OknoGL::przygotujPrzekroje() {

	GLuint atrybutPolozenie = glGetAttribLocation(idProgramuShaderow, "polozenie_in");
	if (atrybutPolozenie == (GLuint)-1) atrybutPolozenie = 0;

	GLuint atrybutWspolrzedneTeksturowania = glGetAttribLocation(idProgramuShaderow, "wspTekstur_in");
	if (atrybutWspolrzedneTeksturowania == (GLuint)-1) atrybutWspolrzedneTeksturowania = 2;

	GLuint atrybutKolor = glGetAttribLocation(idProgramuShaderow, "kolor_in");
	if (atrybutKolor == (GLuint)-1) atrybutKolor = 3;
	unsigned int liczbaPrzekrojow = 0;
	GLuint *indeksyTekstur = tekstury->indeksyTekstur();
	switch (parametryWyswietlania.type) {

	case TYPE_2D:
		liczbaPrzekrojow = 1;
		przekroje = new CrossSection*[liczbaPrzekrojow];
		przekroje[0] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, parametryWyswietlania.x_mm*parametryWyswietlania.xSizeScale, parametryWyswietlania.y_mm*parametryWyswietlania.ySizeScale);
		przekroje[0]-> MacierzSwiata = Macierz4::ObrotY(180);
		przekroje[0]->IndeksTekstury = indeksyTekstur[0];
		break;
	case TYPE_3D:
		liczbaPrzekrojow = parametryWyswietlania.depth_px; +parametryWyswietlania.ascanSize_px + parametryWyswietlania.bscanSize_px;
		przekroje = new CrossSection*[liczbaPrzekrojow];
		float bskany_krok = ((float)parametryWyswietlania.z_mm*parametryWyswietlania.zSizeScale) / (parametryWyswietlania.depth_px-1);
		float przekroje_poziome_krok = ((float)parametryWyswietlania.x_mm*parametryWyswietlania.xSizeScale) / (parametryWyswietlania.bscanSize_px-1);
		float przekroje_poprzeczne_krok = ((float)parametryWyswietlania.y_mm*parametryWyswietlania.ySizeScale) / (parametryWyswietlania.ascanSize_px-1);

		float rozmiarX = parametryWyswietlania.x_mm*parametryWyswietlania.xSizeScale;
		float rozmiarY = parametryWyswietlania.y_mm*parametryWyswietlania.ySizeScale;
		float rozmiarZ = parametryWyswietlania.z_mm*parametryWyswietlania.zSizeScale;

		for (size_t i = 0, end = parametryWyswietlania.depth_px; i != end; ++i) {

			przekroje[i] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, rozmiarX, rozmiarY);
			przekroje[i]->MacierzSwiata = Macierz4::Przesuniecie(0.0f, 0.0f, -bskany_krok*(end-1- i));
			//przekroje[i]->MacierzSwiata = Macierz4::Przesuniecie(0.0f, 0.0f, -bskany_krok*i);
			przekroje[i]->IndeksTekstury = indeksyTekstur[i];
		}

		for (size_t i = 0, end = parametryWyswietlania.ascanSize_px; i != end; ++i) {

			przekroje[i + parametryWyswietlania.depth_px] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, rozmiarX, rozmiarZ);
			przekroje[i + parametryWyswietlania.depth_px]->MacierzSwiata = Macierz4::Przesuniecie(0.0f,przekroje_poprzeczne_krok*(end-1 - i)-rozmiarY/2, -rozmiarZ / 2) *Macierz4::ObrotX(-270);
			przekroje[i + parametryWyswietlania.depth_px]->IndeksTekstury = indeksyTekstur[i+parametryWyswietlania.depth_px];

		}
		
		for (size_t i = 0, end = parametryWyswietlania.bscanSize_px; i != end; ++i) {

			przekroje[i + parametryWyswietlania.depth_px+ parametryWyswietlania.ascanSize_px] = new CrossSection(atrybutPolozenie, atrybutWspolrzedneTeksturowania, atrybutKolor, rozmiarZ, rozmiarY);
			przekroje[i + parametryWyswietlania.depth_px + parametryWyswietlania.ascanSize_px]->MacierzSwiata = Macierz4::Przesuniecie(i*przekroje_poziome_krok-rozmiarX/2, 0.0f, -((parametryWyswietlania.depth_px-1)*bskany_krok-rozmiarZ/2))*Macierz4::ObrotY(270);
			przekroje[i + parametryWyswietlania.depth_px + parametryWyswietlania.ascanSize_px]->IndeksTekstury = indeksyTekstur[i + parametryWyswietlania.depth_px + parametryWyswietlania.ascanSize_px];
		}
		
		break;
	}
	
	return liczbaPrzekrojow;
}
*/
void OknoGL::RysujAktorow() {

	for (unsigned int i = 0; i < liczbaPrzekrojow; ++i) {

		MVP.Ustaw(VP);
		MVP.PomnozZPrawej(przekroje[i]->MacierzSwiata);
		MVP.PrzeslijWartosc();
	
		if (teksturowanieWlaczone) {

			if (przekroje[i]->IndeksTekstury != -1) {

				glUniform1i(glGetUniformLocation(idProgramuShaderow, "Teksturowanie"), true);
				glBindTexture(GL_TEXTURE_2D, przekroje[i]->IndeksTekstury);
			
			}

		//	glUniform1f(glGetUniformLocation(idProgramuShaderow, "przezroczystosc"), przekroje[i]->przezroczystosc);

		} else glUniform1i(glGetUniformLocation(idProgramuShaderow, "Teksturowanie"), false);

		przekroje[i]->Rysuj();
	}
}

void OknoGL::UsunAktorow() {

	for (unsigned int i = 0; i < liczbaPrzekrojow; ++i)
		delete przekroje[i];

	delete[] przekroje;
	liczbaPrzekrojow = 0;
}

void OknoGL::UsunTekstury() {

	glBindTexture(GL_TEXTURE_2D, NULL);
	glDeleteTextures(liczbaTekstur, indeksyTekstur);
	delete[] indeksyTekstur;
	liczbaTekstur = 0;
}