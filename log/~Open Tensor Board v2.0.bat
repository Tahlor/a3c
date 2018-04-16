REM set /p dir="Enter save dir: "
SET var=%cd%

call activate tensorflow
call tensorboard --logdir="%var%"
start "127.0.0.1:6006"

pause

pause

