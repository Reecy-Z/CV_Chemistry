import pandas as pd
import numpy as np
a = np.empty()
b = np.zeros((1,2,3))
c = np.concatenate([a,b],1)
print(c.shape)
