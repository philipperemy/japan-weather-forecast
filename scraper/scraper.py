import json
import os

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_soup(url: str):
    print(url)
    resp = requests.get(url)
    assert resp.status_code == 200
    return BeautifulSoup(resp.content, 'lxml')


def fetch_files(output_dir='output', view='13'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open('stations.txt') as r:
        stations = r.read().strip().split('\n')
    for station in stations:
        d = fetch_file(station, view)
        if d is None:
            continue
        output_filename = os.path.join(output_dir, 'STATION_' + station + '_VIEW_' + view) + '.json'
        with open(output_filename, 'w') as w:
            json.dump(d, w, indent=2, ensure_ascii=True)
        print(f'-> {output_filename}.')


def fetch_file(station_id='47897', view='13'):
    # https://stackoverflow.com/questions/14863495/find-the-selected-option-using-beautifulsoup
    url = f'https://www.data.jma.go.jp/obd/stats/etrn/view/monthly_s3_en.php?block_no={station_id}&view={view}'
    html = get_soup(url)

    if 'This item is not observed' in html.text:
        return None

    station = str(html.find('select', {'label': 'station'}).find('option', {'selected': True}).contents[0])
    view = str(html.find('select', {'name': 'view'}).find('option', {'selected': True}).contents[0])
    raw_records = html.find('table', {'class': 'data2_s'}).find_all('tr')
    raw_headers, raw_rows = raw_records[0], raw_records[1:]

    headers = [str(s.contents[0]) for s in raw_headers.find_all('th')]
    headers = [str(h) for h in headers if h.lower() != 'year']
    # data_1t_0_0_0 or data_0_0_0_0
    records = [[str(s.contents[0]) for s in
                r.find_all('td')]
               # list(r.find_all('td', {'class': 'data_0_0_0_0'})) +
               # list(r.find_all('td', {'class': 'data_1t_0_0_0'}))]
               for r in raw_rows]
    indexes = [str(r.contents[0].contents[0]) for r in raw_rows]
    assert len(records) == len(indexes)
    np_records = np.array(records)
    assert list(np_records[:, 0]) == indexes
    records = np_records[:, 1:]
    d = pd.DataFrame(data=records, columns=headers, index=indexes)
    output = {'station': station.encode('ascii', 'ignore').decode('utf8'), 'view': view, 'data': d.to_dict()}
    return output


def main():
    views = ['12', '13']  # see views.txt
    for view in views:
        fetch_files(os.path.join('..', 'output'), view)


if __name__ == '__main__':
    main()
