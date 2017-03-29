@echo off

set XPJ="xpj4.exe"

%XPJ% -v 1 -t VC11 -p WIN32 -x SSE2NEON.xpj
%XPJ% -v 1 -t VC11 -p WIN64 -x SSE2NEON.xpj

cd ..
cd vc11win64

goto cleanExit

:pauseExit
pause

:cleanExit

