# Japan Weather Data
*From the Japanese Meteorological Agency (scraper + data)*

## Get started

```bash
git clone git@github.com:philipperemy/japan-weather-forecast.git && cd japan-weather-forecast
pip install -r requirements.txt
cd scraper && python scraper.py # data will be located in ../output
```

## Data format

Examples are provided [here](output).

```json
{
  "station": "WAKKANAIWMO Station ID:47401",
  "view": "Monthly total of sunshine duration",
  "data": {
    "Jan": {
      "1938": "59.2",
      "1939": "45.9",
      "1940": "59.5",
      "1941": "44.1",
      "1942": "36.3",
      "1943": "37.7",
      "1944": "29.3",
      "1945": "29.2",
      [...]
      "2012": "1377.4",
      "2013": "1384.1",
      "2014": "1640.6",
      "2015": "1437.5",
      "2016": "1401.1",
      "2017": "1502.0",
      "2018": "1471.0",
      "2019": "316.9 ]"
    }
  }
}
```
