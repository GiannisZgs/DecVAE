import json
import sys

if len(sys.argv) < 4:
    print("Usage: python update_config.py <config_file> <param_name> <value>")
    sys.exit(1)

config_file = sys.argv[1]
param_name = sys.argv[2]
value = sys.argv[3]

# Convert string values to appropriate types
if value.lower() == 'true':
    value = True
elif value.lower() == 'false':
    value = False
elif value.isdigit():
    value = int(value)
elif value.replace('.', '', 1).isdigit():
    value = float(value)

with open(config_file, 'r') as f:
    config = json.load(f)

config[param_name] = value

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Updated '{param_name}' to {value} in {config_file}")