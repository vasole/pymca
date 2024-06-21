datadir=data/test_data

for f in $datadir/*.dat;
do (cat "${f}"; echo) >> temp_data2.dat;
done;
#dos2unix temp_data2.dat; would need to install?
sed 's/\r//' temp_data2.dat > temp_data.dat
rm temp_data2.dat

