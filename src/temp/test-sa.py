from saliecny_map import SaliencyMap
from saliecny_map.utils import OpencvIo

oi = OpencvIo()
src = oi.imread("test.png")
sm = SaliencyMap(src)
oi.imshow_array([sm.map])
