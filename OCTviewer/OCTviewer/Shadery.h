#pragma once


const GLchar *vertexShader = { "#version 330 core\n"
"layout(location =0) in vec3 polozenie_in;\n"
"layout(location = 2) in vec2 wspTekstur_in;\n"
"layout(location =3) in vec4 kolor_in;\n"
"uniform bool UwzglednijKolorWerteksu = true;\n"
"uniform vec4 Kolor = vec4(1.0,1.0,0.0,1.0);\n"
"uniform float przezroczystosc = 1.0f;\n"
"const mat4 macierzJednostkowa = mat4(1.0);\n"
"uniform mat4 mvp = macierzJednostkowa;\n"
"out vec4 polozenie;\n"
"out vec4 kolor;\n"
"out vec2 wspTekstur;\n"
"out vec3 polozenie_scena;\n"
"void main(void){\n"
"polozenie = vec4(polozenie_in,1.0);\n"
"gl_Position = mvp*polozenie;\n"
"kolor = vec4(kolor_in.r,kolor_in.g,kolor_in.b,przezroczystosc);\n"
"wspTekstur = wspTekstur_in;\n"
"}\0" };

const GLchar *fragmentShader = { "#version 330 core\n"
"out vec4 kolor_out;\n"
"in vec4 kolor;\n"
"in vec2 wspTekstur;\n"
"uniform bool Teksturowanie = true;\n"
"uniform bool flaga = true;\n"
"uniform sampler2D ProbnikTekstury;\n"
"void main(void){\n"
"vec4 teksel = vec4(1.0f,1.0f,1.0f,1.0f);\n"
"if(Teksturowanie){\n"
"teksel = texture2D(ProbnikTekstury,wspTekstur);\n"
"teksel.a = kolor.a;\n"
"kolor_out = teksel;\n"
"} else kolor_out = kolor;\n"
"//kolor_out = vecr4(1.0f-kolor_out.r,1.0f-kolor_out.g,1.0f-kolor_out.b,kolor_out.a);\n"
"}\0" };