@echo off
setlocal enabledelayedexpansion
cd %~dp0

py plot2.py %1 %2
py plot3.py %1 %2
py count_evalution.py %1 %2
py rating_count.py %1 %2