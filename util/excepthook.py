import traceback

# Define the modules you want to filter out
IGNORED_MODULES = [
    'Traceback (most recent call last)',
    '/Documents/GitHub/dsst/.venv/lib/python3.9/site-packages/torch/nn/modules/',
    '/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/runpy.py',
]

def filter_traceback(tb):
  filtered_tb = []
  for line in tb:
    if not any(ignored_module in line for ignored_module in IGNORED_MODULES):
      line = line.replace('/Users/abe/Documents/GitHub/dsst/', '~/')
      if line[:2] == '  ':
        line = line[2:]
      filtered_tb.append(line)
  return filtered_tb

def custom_excepthook(type, value, tb):
    # Format the traceback
    tb_lines = traceback.format_exception(type, value, tb)
    
    # Filter the traceback
    filtered_tb = filter_traceback(tb_lines)
    
    # Print the filtered traceback
    print("\n*********************")
    for line in filtered_tb:
        print(line, end='')