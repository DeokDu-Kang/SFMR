# SFMR
#25.07.09
#Chatgpt의 Codex 기능을 사용해 보기위해 만든 Github repository
#이름을 SFMR로 지었으며, 유용하다고 판단되면 SFMR 관련 코딩을 여기서 해보려고 한다.

## ERA5 Download Script

`download_era5.py` uses the [CDS API](https://cds.climate.copernicus.eu/api-how-to) to download ERA5 reanalysis data. You must create an account on the Copernicus Climate Data Store and place your API key in `~/.cdsapirc` before running.

Example:

```bash
python download_era5.py --variable 2m_temperature --year 2024 --month 01 --day 01 --time 00:00 --outfile era5.nc
```

