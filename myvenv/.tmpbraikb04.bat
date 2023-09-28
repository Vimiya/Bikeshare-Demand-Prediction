@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\Users\Admin\anaconda3\condabin\conda.bat" activate "d:\Vimiya Folders\Data science interview\Internships\ineuron\Bikeshare Demand Prediction\myvenv"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@python c:\Users\Admin\.vscode\extensions\ms-python.python-2023.16.0\pythonFiles\get_output_via_markers.py c:/Users/Admin/.vscode/extensions/ms-python.python-2023.16.0/pythonFiles/printEnvVariables.py
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
