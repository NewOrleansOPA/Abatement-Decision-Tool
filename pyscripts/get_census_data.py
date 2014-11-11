import requests 
import csv 
from StringIO import StringIO
from zipfile import ZipFile 
import ujson 
from inspect import isgenerator

import re
from string import punctuation 

punct = frozenset(punctuation)
re_wht = re.compile('\s+')

def unzip(contents, format):
  """
  Unzip a file, extracting a particular format.
  """

  fobj = StringIO()
  fobj.write(contents)
  zobj = ZipFile(fobj)
  filenames = zobj.namelist()
  output = []
  for filename in filenames:
    if filename.endswith(format):
      fobj = zobj.open(filename)
      output.append((filename, fobj))
  return output

def write_csv(data, filename='data/census_data.csv'):
  """
  Parse a csv.
  """
  
  if isgenerator(data):
    data = list(data)

  fieldnames = data[0].keys()
  writer = csv.DictWriter(open(filename, 'wb'), fieldnames=fieldnames)
  writer.writeheader()
  writer.writerows(data)


def slugify(s, sep="_"):
  """
  Slugify a string.
  """

  s = "".join([c if c not in punct else " " for c in s])
  s = re_wht.sub(" ", s).strip().lower()
  return "census{}{}".format(sep, s.replace(' ', sep))


def parse_geojson(fobj, tables, incl_error=False, percentiles=True):
  """
  Parse a geojson file.
  """

  raw = ujson.loads(fobj.read())
  for row in raw['features']:
    clean_row = {}
    for k,v in row['properties'].items():
      
      for t in tables:
        k = k.replace(t[0], t[1] + " ")
      
      k = slugify(k)

      if k.endswith('error'):
        if incl_error:
          clean_row[k] = v 

      else:
        clean_row[k] = v

    if percentiles:
      for k,v in clean_row.items():
        k_per = "{}_per".format(k)
        if k not in ['census_geoid', 'census_name', 'census_population_001_total']:
          if clean_row['census_population_001_total'] > 0 and v > 0:
            clean_row[k_per] = v / clean_row['census_population_001_total']
          elif v == 0:
            clean_row[k_per] = 0
          else:
            clean_row[k_per] = None

      
    yield clean_row


def get_nola_tables(tables, format='geojson'):
  """
  Get nola data.
  """

  url = "http://api.censusreporter.org/1.0/data/download/latest"

  params = {
    'table_ids': ','.join([t[0] for t in tables]),
    'geo_ids': '16000US2255000,140|16000US2255000',
    'format': format
  }

  r = requests.get(url, params=params)
  files = unzip(r.content, format)
  data = parse_geojson(files[0][1], tables)
  return write_csv(data)

def table_suggest(q, start=0, max_pages=100):
  
  url = "http://api.censusreporter.org/1.0/table/elasticsearch"
  page = 0
  
  while True:
  
    start += 25
    page += 1 

    params = {
      'q': q,
      'start': start
    }
    r = requests.get(url, params=params)
    raw = r.json()

    if len(raw['results']):    
      for res in raw['results']:
        yield res
    
    else:
      break

    if page >= max_pages:
      break

if __name__ == '__main__':
  TABLES = [
    ('B01003', 'population'),
    ('B25058', 'median_contract'),
    ('B02001', 'race'),
    ('B25081', 'mortgage_status')
  ]
  get_nola_tables(TABLES)

