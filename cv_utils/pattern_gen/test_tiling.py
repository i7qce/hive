# %%
import tiling
import numpy as np
import matplotlib.pyplot as plt

# %%
a =  tiling.generate_periodic_data(placer=tiling.example_placer)
# %%
plt.imshow(a[0])

# %%
def placer(img, label, loc):
    if np.random.randint(0,100) > 95:
        tiling.place_circle(img, loc, 13)
        tiling.place_circle(label, loc, 13)
    else:
        tiling.place_circle(img, loc, 15)
# %%
a =  tiling.generate_periodic_data(placer=placer)
# %%
plt.imshow(a[0])
# %%
