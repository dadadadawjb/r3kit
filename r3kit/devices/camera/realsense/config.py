try:
    import pyrealsense2 as rs
except ImportError:
    print("Camera RealSense needs `pyrealsense2`")
    raise ImportError

L515_ID = 'f0172289'
L515_STREAMS = [
    (rs.stream.depth, 1024, 768, rs.format.z16, 30), 
    (rs.stream.color, 1280, 720, rs.format.bgr8, 30)
]

T265_ID = '230222110234'
T265_STREAMS = [
    (rs.stream.pose, rs.format.six_dof, 200), 
    (rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30), 
    (rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)
]
