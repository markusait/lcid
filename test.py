# from PIL import Image
# # import Image
# import numpy as np

# w, h = 512, 512
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
# img = Image.fromarray(data, 'RGB')
# img.save('my.png')
# img.show()




# df = pd.DataFrame({'age':    [ 3,  29],
#                     'height': [94, 170],
#                     'weight': [31, 115]})
# df
#    age  height  weight
# 0    3      94      31
# 1   29     170     115
# print(df.dtypes)
# age       int64
# height    int64
# weight    int64
# dtype: object
# df.values
# array([[  3,  94,  31],
#        [ 29, 170, 115]], dtype=int64)

# import numpy as np
# x = np.array([[2,3,4,8], [5,6,7,9]])
# print("a", np.reshape(x, (4,-1)))



# import numpy as np
# import pandas as pd
# import pandas_profiling

# df = pd.DataFrame(
#     np.random.rand(100, 5),
#     columns=['a', 'b', 'c', 'd', 'e']
# )
# df.profile_report()