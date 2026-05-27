Data format:
The user works with monthly Parquet files containing KNMI station-level observations.
path structure is as follows: "C:\Users\floris\Desktop\MSC\thesis_msc\data\yearly_aggregated_FH_FX\year=1966\month=05\knmi_fh_1966_05.parquet" which is all the data for May 1966.

Typical columns:
time, station, stationname, lat, lon, height, FH, FX

We focus on wind gust.

Relevant variables:
- time: timestamp of the hourly observation
- station: KNMI station identifier
- stationname: station name
- lat: station latitude
- lon: station longitude
- height: station elevation or sensor height, depending on KNMI definition
- FX: maximum wind gust over the past hour

Primary empirical variable:
- FX, because the thesis focuses on extreme wind gusts.

Frequency:
Hourly or converted from 10-minute observations depending on preprocessing stage.

Core tasks:
- Load monthly parquet files
- Filter Dutch stations
- Select seasons, especially DJF and JJA
- Handle missingness
- Construct block maxima
- Create station-pair datasets
- Compute distances between stations using lat / lon 
- Estimate pairwise extremal dependence 
- Compare winter versus summer 