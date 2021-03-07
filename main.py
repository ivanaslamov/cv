from app.calibration import calibrate

matrix, distortion, r_vecs, t_vecs = calibrate("data/checkerboards/", preview=False)

print(distortion)
