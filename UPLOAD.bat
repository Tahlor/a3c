set /p ext="Enter file extension: "

"D:\OneDrive\Documents\Graduate School\PSCP.exe" -l tarch -P 22 -pw =-09][po=-09 -r ./*%ext% ssh.fsl.byu.edu:/fslhome/tarch/compute/678/a3c/

"D:\OneDrive\Documents\Graduate School\PSCP.exe" -l tarch -P 22 -pw =-09][po=-09 -r ./model/*%ext% ssh.fsl.byu.edu:/fslhome/tarch/compute/678/a3c/model

"D:\OneDrive\Documents\Graduate School\PSCP.exe" -l tarch -P 22 -pw =-09][po=-09 -r ./process_data/*%ext% ssh.fsl.byu.edu:/fslhome/tarch/compute/678/a3c/process_data

pause
