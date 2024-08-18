from general_robotics_toolbox import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy, math

ex=np.array([[1.],[0.],[0.]])
ey=np.array([[0.],[1.],[0.]])
ez=np.array([[0.],[0.],[1.]])

def Rx(theta):
	return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
	return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
	return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

def rotate_vector_at_angle(u, v, theta_rad):
    ###rotate u to v at angle theta
    # Compute the unit normal to the plane defined by u and v
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)
    
    # Compute the vector w that lies at angle theta from u in the plane of u and v
    w = u * np.cos(theta_rad) + np.cross(n, u) * np.sin(theta_rad)
    
    return w / np.linalg.norm(w)  # Return as unit vector

def H_inv(H):
	R=H[:3,:3].T
	p=-R@H[:3,-1]
	return H_from_RT(R,p)
def transform_curve(curve,H):
	###trasform spatial curve with transformation H
	R=H[:3,:3]
	T=H[:-1,-1]
	curve_new=np.dot(R,curve[:,:3].T).T+np.tile(T,(len(curve),1))
	if len(curve[0])>3:
		curve_normal_new=np.dot(R,curve[:,3:].T).T
		return np.hstack((curve_new,curve_normal_new))
	else:
		return curve_new

def vector_to_plane(point, centroid, normal):		###find the vector from point to plane
	return  np.dot(centroid - point,normal)*normal

def point2plane_distance(p,centroid,normal):
	return np.abs(np.dot(p - centroid, normal) / np.linalg.norm(normal))

def rodrigues_rot(curve, n0, n1):

    if np.all(n0==n1):
        return curve
    
    # If curve is only 1d array (coords of single point), fix it to be matrix
    if curve.ndim == 1:
        curve = curve[np.newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0,n1))
    
    # Compute rotated points
    curve_rot = np.zeros((len(curve),3))
    for i in range(len(curve)):
        curve_rot[i] = curve[i]*np.cos(theta) + np.cross(k,curve[i])*np.sin(theta) + k*np.dot(k,curve[i])*(1-np.cos(theta))

    return curve_rot


def fit_plane(points):
	# Calculate the centroid of the points
	centroid = np.mean(points, axis=0)

	# Center the points by subtracting the centroid
	centered_points = points - centroid

	# Calculate the SVD of the centered points
	u, s, vh = np.linalg.svd(centered_points)

	# The normal vector of the plane is the last column of vh
	normal = vh[-1]

	return normal, centroid

def project_onto_plane(points, normal):
	points2d = rodrigues_rot(points, normal, [0,0,1])

	return points2d[:,:2]


def pose_regression(A,B):
	###find transformation between ordered point lists A and B with regression
	center_A = np.mean(A,axis=0)
	center_B = np.mean(B,axis=0)

	A_centered = A-center_A
	B_centered = B-center_B
	H = np.matmul(A_centered.T,B_centered)
	u,s,vT = np.linalg.svd(H)

	R = np.matmul(vT.T,u.T)
	if np.linalg.det(R)<0:
		vT[2,:] *= -1
		R = vT.T @ u.T

	t = center_B-np.dot(R,center_A)

	return H_from_RT(R,t)


def point2line_vec(p1,p2,p3):
	#find normal vector from p1 pointing to line of p2p3
	p2p1=p2-p1
	p3p1=p3-p1
	p2p3=p2-p3
	vec=np.cross(np.cross(p2p1,p3p1),p2p3)
	return vec/np.linalg.norm(vec)


def clip_joints(robot,curve_js,relax=0.05):
	curve_js_clipped=np.zeros(curve_js.shape)
	for i in range(len(curve_js[0])):
		curve_js_clipped[:,i]=np.clip(curve_js[:,i],robot.lower_limit[i]+relax,robot.upper_limit[i]-relax)

	return curve_js_clipped



def interplate_timestamp(curve,timestamp,timestamp_d):

	curve_new=[]
	for i in range(len(curve[0])):
		# curve_new.append(np.interp(timestamp_d,timestamp,curve[:,i]))
		curve_new.append(scipy.interpolate.CubicSpline(timestamp, curve[:,i])(timestamp_d))

	return np.array(curve_new).T

	
def replace_outliers2(data,rolling_window=30,threshold=0.0001):
	###replace outlier with rolling average
	rolling_window=30
	rolling_window_half=int(rolling_window/2)
	for i in range(rolling_window_half,len(data)-rolling_window_half):
		rolling_avg=np.mean(data[i-rolling_window_half:i+rolling_window_half])
		if np.abs(data[i]-rolling_avg)>threshold*rolling_avg:
			rolling_avg=(rolling_avg*rolling_window-data[i])/(rolling_window-1)
			data[i]=rolling_avg
	return data
def replace_outliers(data, m=2):
	###replace outlier with average
	data[abs(data - np.mean(data)) > m * np.std(data)] = np.mean(data)
	return data

def identify_outliers2(data,rolling_window=30,threshold=0.0001):
	###detect outlier with rolling average
	indices=[]
	rolling_window=30
	rolling_window_half=int(rolling_window/2)
	for i in range(rolling_window_half,len(data)-rolling_window_half):
		rolling_avg=np.mean(data[i-rolling_window_half:i+rolling_window_half])
		if np.abs(data[i]-rolling_avg)>threshold*rolling_avg:
			indices.append(i)
	return indices

def identify_outliers(data,m=2):
	return np.argwhere(abs(data - np.mean(data)) > m * np.std(data)).flatten()

def quadrant(q,robot):
	cf146=np.floor(np.array([q[0],q[3],q[5]])/(np.pi/2))
	eef=fwdkin(robot.robot_def_nT,q).p
	
	REAR=(1-np.sign((Rz(q[0])@np.array([1,0,0]))@np.array([eef[0],eef[1],eef[2]])))/2

	LOWERARM= q[2]<-np.pi/2
	FLIP= q[4]<0


	return np.hstack((cf146,[4*REAR+2*LOWERARM+FLIP])).astype(int)


def direction2R_x(v_norm,v_tang):
	###form rotation matrix from a normal and a tangent vector
	v_norm=v_norm/np.linalg.norm(v_norm)
	v_tang=VectorPlaneProjection(v_tang,v_norm)
	y=np.cross(v_norm,v_tang)
	y=y/np.linalg.norm(y)
	R=np.vstack((v_tang,y,v_norm)).T
	return R

def direction2R_y(v_norm,v_tang):
	###form rotation matrix from a normal and a tangent vector
	v_norm=v_norm/np.linalg.norm(v_norm)
	v_tang=VectorPlaneProjection(v_tang,v_norm)
	x=np.cross(v_tang,v_norm)
	x=x/np.linalg.norm(x)
	R=np.vstack((x,v_tang,v_norm)).T
	return R

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
 
	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")
 
	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi

def VectorPlaneProjection(v,n):
	temp = (np.dot(v, n)/np.linalg.norm(n)**2)*n
	v_out=v-temp
	v_out=v_out/np.linalg.norm(v_out)
	return v_out

def find_j_det(robot,curve_js):
	j_det=[]
	for q in curve_js:
		j=robot.jacobian(q)
		# j=j/np.linalg.norm(j)
		j_det.append(np.linalg.det(j))

	return j_det

def find_condition_num(robot,curve_js):
	cond=[]
	for q in curve_js:
		u, s, vh = np.linalg.svd(robot.jacobian(q))
		cond.append(s[0]/s[-1])

	return cond


def find_j_min(robot,curve_js):
	sing_min=[]
	for q in curve_js:
		u, s, vh = np.linalg.svd(robot.jacobian(q))
		sing_min.append(s[-1])

	return sing_min

def get_angle2(v1,v2,k=None):
	#signed rotational angle from v1 to v2, rotation about k if provided
	if k is not None:
		return np.arctan2(np.dot(np.cross(v1, v2),k), np.dot(v1,v2))
	else:
		return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1,v2))
		
def get_angle(v1,v2,less90=False):
	v1=v1/np.linalg.norm(v1)
	v2=v2/np.linalg.norm(v2)
	dot=np.dot(v1,v2)
	if dot>0.99999999999:
		return 0
	elif dot<-0.99999999999:
		return np.pi
	angle=np.arccos(dot)
	if less90 and angle>np.pi/2:
		angle=np.pi-angle
	return angle


def lineFromPoints(P, Q):
	#return coeff ax+by+c=0
	a = Q[1] - P[1]
	b = P[0] - Q[0]
	c = -(a*(P[0]) + b*(P[1]))
	return a,b,c

def line_intersection(p1, v1, p2, v2):
    n = np.cross(v1, v2)        #closest points on both line will lie on the perpendicular vector of both lines
    n1 = np.cross(v1, n)        #define normal vector of plane formed by v1 and n
    n2 = np.cross(v2, n)        #define normal vector of plane formed by v2 and n

    #see my write up
    c1 = p1 + np.dot(p2 - p1, n2) / np.dot(v1, n2) * v1
    c2 = p2 + np.dot(p1 - p2, n1) / np.dot(v2, n1) * v2

    # Return the midpoint on both lines as the intersection point
    return (c1 + c2) / 2

def visualize_curve_w_normal(curve,curve_normal,stepsize=500,equal_axis=False):
	curve=curve[::stepsize]
	curve_normal=curve_normal[::stepsize]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
	ax.quiver(curve[:,0],curve[:,1],curve[:,2],30*curve_normal[:,0],30*curve_normal[:,1],30*curve_normal[:,2])
	ax.set_xlabel('x (mm)')
	ax.set_ylabel('y (mm)')
	ax.set_zlabel('z (mm)')
	set_axes_equal(ax)

	plt.show()

def visualize_curve(curve,stepsize=10):
	curve=curve[::stepsize]
	plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1],curve[:,2], 'gray')
	set_axes_equal(ax)

	plt.show()

def linear_interp(x,y):
	###avoid divided by 0 problem
	x,unique_indices=np.unique(x,return_index=True)
	if (len(unique_indices)<len(y)-2):
		print('Duplicate in interpolate, check timestamp')
	y=y[unique_indices]
	f=interp1d(x,y.T)
	x_new=np.linspace(x[0],x[-1],len(x))
	return x_new, f(x_new).T

def moving_averageNd(a,n=5,padding=False):
	a_avg = []

	for dim in range(a.shape[1]):
		a_temp=a[:, dim]
		if padding:
			a_temp=np.hstack(([np.mean(a_temp[:int(n/2)])]*int(n/2),a_temp,[np.mean(a_temp[-int(n/2):])]*int(n/2)))
		
		a_avg.append(np.convolve(a_temp, np.ones(n), mode='valid') / n)
	return np.array(a_avg).T

def moving_average(a, n=11, padding=False):
	#n needs to be odd for padding
	if padding:
		a=np.hstack(([np.mean(a[:int(n/2)])]*int(n/2),a,[np.mean(a[-int(n/2):])]*int(n/2)))
	ret = np.cumsum(a, axis=0)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def lfilter(x, y):
	x,y=linear_interp(x,y)
	n=10
	y1=moving_average(y,n)
	y2=moving_average(np.flip(y,axis=0),n)

	return x[int(n/2):-int(n/2)+1], (y1+np.flip(y2,axis=0))/2

def orientation_interp(R_init,R_end,steps):
	curve_fit_R=[]
	###find axis angle first
	R_diff=np.dot(R_init.T,R_end)
	k,theta=R2rot(R_diff)
	for i in range(steps):
		###linearly interpolate angle
		angle=theta*i/(steps-1)
		R=rot(k,angle)
		curve_fit_R.append(np.dot(R_init,R))
	curve_fit_R=np.array(curve_fit_R)
	return curve_fit_R

def H_from_RT(R,T):
	return np.hstack((np.vstack((R,np.zeros(3))),np.append(T,1).reshape(4,1)))



def R2w(curve_R,R_constraint=[]):
	###convert rotation matrix to axis-angle, starting initial rotation matrix as 0
	if len(R_constraint)==0:
		R_init=curve_R[0]
		curve_w=[np.zeros(3)]
	else:
		R_init=R_constraint
		R_diff=np.dot(curve_R[0],R_init.T)
		k,theta=R2rot(R_diff)
		k=np.array(k)
		curve_w=[k*theta]
	
	for i in range(1,len(curve_R)):
		R_diff=np.dot(curve_R[i],R_init.T)
		k,theta=R2rot(R_diff)
		k=np.array(k)
		curve_w.append(k*theta)
	return np.array(curve_w)

def w2R(curve_w,R_init):
	###convert axis-angle to rotation matrix, starting initial rotation matrix as 0
	curve_R=[]
	for i in range(len(curve_w)):
		theta=np.linalg.norm(curve_w[i])
		if theta==0:
			curve_R.append(R_init)
		else:
			curve_R.append(np.dot(rot(curve_w[i]/theta,theta),R_init))

	return np.array(curve_R)

def rotation_matrix_from_vectors(vec1, vec2):	#https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
	""" Find the rotation matrix that aligns vec1 to vec2
	:param vec1: A 3d "source" vector
	:param vec2: A 3d "destination" vector
	:return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
	"""
	a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
	return rotation_matrix


def rotationMatrixToEulerAngles(R) :
	###https://learnopencv.com/rotation-matrix-to-euler-angles/
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

	singular = sy < 1e-6

	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
	return [x, y, z]


def unwrapped_angle_check(q_init,q_all):

	temp_q=q_all-q_init
	temp_q = np.unwrap(temp_q)
	order=np.argsort(np.linalg.norm(temp_q,axis=1))

	return temp_q[order[0]]+q_init

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    ax.set_box_aspect([1,1,1])
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
	
    