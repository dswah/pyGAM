from pygam import LinearGAM

gam = LinearGAM()

params = gam.get_params()

print("Before change:", params["callbacks"])

# Try modifying returned params
params["callbacks"].append("hack")

# Check again
params2 = gam.get_params()
print("After change:", params2["callbacks"])