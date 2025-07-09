import argparse
import cdsapi


def download_era5(out_file: str,
                  variable: str = '2m_temperature',
                  year: str = '2024',
                  month: str = '01',
                  day: str = '01',
                  time: str = '00:00') -> None:
    """Download ERA5 reanalysis data from the CDS.

    A valid CDS API key must be configured in ~/.cdsapirc.
    """
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variable,
            'year': year,
            'month': month,
            'day': day,
            'time': time,
        },
        out_file,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ERA5 data using CDS API.')
    parser.add_argument('--variable', default='2m_temperature', help='Variable to download')
    parser.add_argument('--year', default='2024', help='Year of data')
    parser.add_argument('--month', default='01', help='Month of data')
    parser.add_argument('--day', default='01', help='Day of data')
    parser.add_argument('--time', default='00:00', help='Time of data')
    parser.add_argument('--outfile', default='era5.nc', help='Output file path')
    args = parser.parse_args()

    download_era5(
        out_file=args.outfile,
        variable=args.variable,
        year=args.year,
        month=args.month,
        day=args.day,
        time=args.time,
    )
