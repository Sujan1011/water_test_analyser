@echo off
echo Testing pH with red color...
python test_water_quality.py --rgb 255 0 0 --test ph
echo.
pause

echo Testing Iron with light yellow...
python test_water_quality.py --rgb 255 255 200 --test iron
echo.
pause

echo Testing Nitrate with pink...
python test_water_quality.py --rgb 255 192 203 --test nitrate
echo.
pause

echo Testing Chlorine with golden yellow...
python test_water_quality.py --rgb 255 215 0 --test chlorine
echo.
pause