datadir=../data/test_data
resdir=../data

for f in $datadir/*.dat;
do (cat "${f}"; echo) >> $resdir/temp_data2.dat;
done;
#dos2unix temp_data2.dat; would need to install?
sed 's/\r//' $resdir/temp_data2.dat > $resdir/temp_data.dat # only if ^M character as EOL from windows
rm $resdir/temp_data2.dat

