@echo off
setlocal
python -m venv .venv
if errorlevel 1 goto error
call .venv\Scripts\activate.bat
if errorlevel 1 goto error
python -m pip install --upgrade pip
if errorlevel 1 goto error
python -m pip install -r requirements.txt
if errorlevel 1 goto error
echo Entorno listo. Activalo en tu terminal con: .venv\Scripts\activate
endlocal
exit /b 0

:error
echo Error al crear el entorno o instalar las dependencias.
endlocal
exit /b 1