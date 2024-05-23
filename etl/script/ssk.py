import pathlib

from etl.parse_xml import parse_into_path

PATH = "CIN1314662258-2016-11-02-15-42-25V6.5.ssk"

if __name__ == "__main__":
  xml_dir = pathlib.Path("/Users/abe/Desktop/SSK/")
  data_path = pathlib.Path("/Users/abe/Desktop/OUT/")
  meta_path = pathlib.Path("/Users/abe/Desktop/meta.csv")
  for xml in xml_dir.glob("*.ssk"):
    parse_into_path(xml, data_path, meta_path, overwrite_meta=False)