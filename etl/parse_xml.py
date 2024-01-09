#@title def ssk_file_to_pd, def fix_ssk_columns
from lxml import etree
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

def validate_xml(xml_path: pathlib.Path, xsd_path: pathlib.Path):
    # Load the XSD file
    with open(xsd_path, 'rb') as schema_file:
        schema_root = etree.XML(schema_file.read())
        schema = etree.XMLSchema(schema_root)

    # Load the XML file
    with open(xml_path, 'rb') as xml_file:
        xml_doc = etree.parse(xml_file)

    # Validate XML against the XSD
    try:
        schema.assertValid(xml_doc)
        return True, "XML is valid against the XSD"
    except etree.DocumentInvalid as e:
        return False, str(e)

def get_drawing_strokes(drawing: etree, symbol_digit: str):
  strokes = drawing.find('.//strokes').findall('.//stroke')
  points = pd.DataFrame([
      dict(
          stroke_id=int(stroke.attrib['index']),
          # this is a bug in the file format
          x=float(point.attrib['y']),
          y=float(point.attrib['x']),
          # ms -> seconds
          t=1e-3 * float(point.attrib['timestamp']),
          symbol_digit=symbol_digit
          # pressure attribute has very little sensitivity on the digitizing pen, unfortunately
          )
      for stroke in strokes
      for point in stroke.findall('.//point')
  ])
  return points

def get_clock_sketch_data(clock_sketch_data: etree):
  vals = {}
  for tag in clock_sketch_data:
    for k, v in tag.attrib.items():
      if k == 'value':
        vals[tag.tag] = v
      else:
        vals[tag.tag + '_' + k] = v
  return vals
        

def parse_xml(xml_path: pathlib.Path):
  xml_doc = etree.parse(xml_path)
  symbol_digit = xml_doc.find('.//drawing[@structureID="symbol-digit"]')
  symbol_digit = get_drawing_strokes(symbol_digit, symbol_digit=True)
  digit_digit = xml_doc.find('.//drawing[@structureID="digit-digit"]')
  digit_digit = get_drawing_strokes(digit_digit, symbol_digit=False)

  min_t = symbol_digit.t.min()
  symbol_digit.t -= min_t
  digit_digit.t -= min_t
  # make sure stroke ids are unique
  symbol_digit.stroke_id += 9999
  coords = pd.concat([symbol_digit, digit_digit]).set_index("stroke_id")

  metadata = get_clock_sketch_data(xml_doc.find('.//ClockSketchData'))
  return coords, metadata


if __name__ == "__main__":
  x = validate_xml(pathlib.Path('etl/test/test.xml'), pathlib.Path('etl/schema/top.xsd'))
  print(x)
  c, m = parse_xml(pathlib.Path('etl/test/test.xml'))
  print(m)
  for idx, cc in c.groupby("stroke_id"):
    cc = cc.sort_values(by="t")
    plt.scatter(cc.y, cc.x)
  plt.show()