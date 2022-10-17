import pandas as pd
import numpy as np
ans = pd.read_csv('./answers_v2.3.csv')
ans = ans.to_numpy()/np.sum(ans.to_numpy(), axis=1, keepdims = True)#.to_csv('./answers_v2.2.csv', index = False)
pd.DataFrame(ans, columns = ['r', 'g', 'b']).to_csv('./answers_v2.3(norm).csv', index = False)