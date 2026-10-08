import json
import sys

if len(sys.argv) < 4:
    print("Usage: python update_config.py <config_file> <param_name> <value>")
    sys.exit(1)

config_file = sys.argv[1]
param_name = sys.argv[2]
value = sys.argv[3]

# Convert string values to appropriate types
lower_value = value.lower()
if lower_value == 'true':
    value = True
elif lower_value == 'false':
    value = False
elif lower_value in ('none', 'null'):
    value = None
else:
    # Try JSON parsing first (handles lists, dicts, etc.)
    try:
        value = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        # If not valid JSON, try numeric conversion (ints, floats, scientific notation)
        try:
            float_val = float(value)
            # Convert to int if it's a whole number
            value = int(float_val) if float_val.is_integer() else float_val
        except (ValueError, TypeError):
            # Keep as string if conversion fails
            pass

with open(config_file, 'r') as f:
    config = json.load(f)

config[param_name] = value

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Updated '{param_name}' to {value} in {config_file}")