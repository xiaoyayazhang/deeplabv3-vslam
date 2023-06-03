import cv2,math
import numpy as np
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    orb = cv2.ORB_create()
    principal_point = (320.1, 247.6)  #光心 TUM dataset标定值
    focal_length = 535.4          # 焦距TUM dataset标定值
    K = np.array((535.4, 0, 320.1, 0, 539.2, 247.6, 0, 0, 1)).reshape((3,3))
    p_world = np.zeros((3,1))


    reference_image = cv2.imread("img/data6/1341846314.325981.png")
    test_image = cv2.imread("img/data6/1341846314.357905.png")

    timestamp = "1341846314.3259"

    # Detect features in both images
    reference_keypoints, reference_descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(reference_image, None)
    test_keypoints, test_descriptors = cv2.xfeatures2d.SIFT_create().detectAndCompute(test_image, None)

    # Find correspondences between the features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)
    matches = matcher.match(reference_descriptors, test_descriptors, None)

    # Extract the coordinates of the matched keypoints
    reference_points = np.array([reference_keypoints[m.queryIdx].pt for m in matches])
    test_points = np.array([test_keypoints[m.trainIdx].pt for m in matches])

    # Compute the essential matrix
    E, _ = cv2.findEssentialMat(test_points, reference_points, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover the relative camera pose from the essential matrix
    _, R, t, _ = cv2.recoverPose(E, test_points, reference_points, cameraMatrix=K)

    # Convert the rotation matrix to a quaternion representation
    p_world = np.dot(R, p_world) + t
    print(f'p_world:{p_world}')
    
    # Convert the rotation matrix to a quaternion representation
    tr = np.trace(R)
    qw = np.sqrt(1 + tr) / 2
    qx = (R[2,1] - R[1,2]) / (4 * qw)
    qy = (R[0,2] - R[2,0]) / (4 * qw)
    qz = (R[1,0] - R[0,1]) / (4 * qw)
    q = np.array([qx, qy, qz, qw])
    q = q / np.linalg.norm(q)

    # Write the camera pose to a file in TUM format
    print(f'{timestamp} {"%.6f" % p_world[0]} {"%.6f" % p_world[1]} {"%.6f" % p_world[2]} {"%.6f" % q[0]} {"%.6f" % q[1]} {"%.6f" % q[2]} {"%.6f" % q[3]}')
        
    