#pip install mat73
import mat73
PATH = r'X:\Widefield\mSM65\SpatialDisc\05-Sep-2018\Vc.mat'
mat = mat73.loadmat(PATH)
print(mat['Vc'].shape)