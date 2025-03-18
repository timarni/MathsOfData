import time 
import copy
from math import sqrt
import random

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.sparse.linalg import eigsh, svds, eigs
import numpy.matlib
from tqdm.notebook import trange


from lib.part2.proj import projL1, euclidean_proj_simplex

def projSDP(Z, kappa):
    #PROJSDP This function implements the projection onto SDP cone with bounded trace.

    s,u = np.linalg.eigh(Z)
    u = u.real
    s = euclidean_proj_simplex(s.real,kappa)

    return u@np.diag(s)@u.T
    
def lmo(grad, kappa):
    q, u = eigsh(grad, k=1, tol=1e-16, which='SA')
    u = sqrt(kappa)*u
    return np.outer(u,u) 
    
## FUNCTIONS FOR THE SDP ROUNDING
def remap_centers(assign,k):
    class_vec = np.zeros([10,10])
    max_class = np.zeros([10,2])
    remap_vec = -1*np.ones([10])
    
    # Computing the number of assignments per class (the classes go 0-99, 100-199,...) 
    for l in range(10):
        class_loc = assign[100*(l):100*(l+1)-1]
        for i in range(10):
            class_vec[l,i] = sum(class_loc==i).item()
        max_class[l] = [np.max(class_vec[l,:]), np.argmax(class_vec[l,:])]

    # Remapping the cluster with the largest number of elements to the actual one iteratively
    # Plotting the evolution of class_vec, pos_remap and max_class helps to understand it 
    # easily ;)
    for l in range(10):  
        pos_remap = np.argmax(max_class[:,0])
        remap_vec[int(max_class[pos_remap,1])] = pos_remap
        class_vec[:,int(max_class[pos_remap,1])] = 0
        class_vec[pos_remap,:] = 0
        for i in range(10):
            max_class[i] = [np.max(class_vec[i,:]), np.argmax(class_vec[i,:])]

    return remap_vec


def sdp_rounding(X, k, digits):
    X = digits.dot(X)
    
    N=X.shape[1]
    # computation of an affinity matrix identifying repeated denoised points
    affinity=np.zeros([N,N]);
    for i in range(N): 
        for j in range(N):
            if np.linalg.norm(X[:,i]-X[:,j])<1e-3:
                affinity[i,j]=1;
                affinity[j,i]=1;
    # centers are k most popular points
    centers=np.zeros([k,k])
    for t in range(k):
        s = np.sum(affinity,0)
        idx = np.argmax(s)

        aux = copy.deepcopy(affinity[:,idx])
        centers[t,:]=X[:, idx]
        for i in range(N): 
            if aux[i]==1:
                affinity[i,:]=0
                affinity[:,i]=0
        
    # assignment of points to closest center
    ind=np.zeros([N,1]);
    for i in range(N):
        aux=np.zeros([k,1]);
        for t in range(k):
            aux[t,0]=np.linalg.norm(X[:,i].T - centers[t,:],2);
        ind[i,0]= np.argmin(aux);
    assignment=ind
    
    # remapping to the correct clusters, i.e. first cluster should be 0, ...
    assignment_remap = np.zeros([N,1]);
    remap_vec = remap_centers(assignment,k)
    centers_remap =np.zeros([k,k]); 
    for i in range(N):
            assignment_remap[i] = remap_vec[int(assignment[i])]
    for loc, map_ in enumerate(remap_vec):
        centers_remap[loc,:] = centers[int(map_),:]
    
    return centers_remap, assignment_remap


def misclassification_rate(assignment, labels):
    labels = labels-1
    return np.sum(assignment!=labels)/len(assignment)


def vis_samples(assignment, images, labels):
    assignment=assignment.astype(int)
    classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    labels = labels-1
    rand_samp = np.random.randint(0,1000,25)
    plt.figure(figsize=(7,7))
    for i,samp in enumerate(rand_samp):
        plt.subplot(5,5,i+1)
        plt.imshow(1-np.reshape(images[samp],[28,28]), cmap=plt.cm.gray)
        plt.title('Pred. {0}\n Orig. {1}'.format(classes[assignment[samp].item()],classes[labels[samp].item()]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def value_kmeans(points, labels):
    # This function computes the kmeans value (sum of squared
    # mean deviations from cluster centroid) of a 
    # provided partition of a provided set of points.
    
    k = 10
    points = points.T
    idxx = np.argsort(labels.T)
    idxx = np.squeeze(idxx)
    points = points[idxx,:]
            
    count = np.zeros([k,], int)
    
    for i in range(k):
        count[i] = int(np.sum(labels==(i))) # Changed because of error
        
    idx = 0
    value = 0
    
    for t in range(k):
        cluster = points[idx:(idx+count[t]),:]
        center = np.matmul(np.ones([1, cluster.shape[0]]),cluster)/count[t]
        for i in range(count[t]):
            value = value + np.linalg.norm(cluster[i,:] - center)**2
        idx = (idx + count[t])
        
    return value



Problem = sio.loadmat('lib/part2/data/clustering_data.mat')
C = np.double(Problem['C']) # euclidean distance matrix
N = int(Problem['N']) # number of data points
k = Problem['k'] # number of clusters
opt_val = Problem['opt_val'] # optimum value 
images = Problem['images'] # images for visualization after clustering
labels = Problem['labels'] # true labels
digits = Problem['digits'] # input data points
kappa = 10


A1 = lambda x: np.sum(x, axis = 1)
At1 = lambda y: np.transpose(np.matlib.repmat(y, N, 1))
A2 = lambda x: np.sum(x, axis = 0)
At2 = lambda y: (np.matlib.repmat(y, N, 1))

b = np.double(np.ones(N))


def run_PD3O(alg, composite, maxit=int(1e3)):
    state = alg.init_state(composite, np.zeros((N,N)))
    
    
    feasibility1 = [] # norm(A1(X)-b1)/norm(b1)
    feasibility2 = [] # dist(X, \mathcal{K})
    objective    = [] # f(x)
    cur_iter    = [] 
    t    = [] 
    
    
    iter_track = np.unique(np.ceil(np.power(2, np.linspace(0,20,50))))
    
    start = time.time()
    
    bar = trange(1, maxit+1)
    for iteration in bar:
        
        # Primal variable update
        state = alg.state_update(composite, state)
                
        # Update A*X - b
        AX1_b = A1(state.x_k)-b
        
        if any(iteration == iter_track) or iteration==maxit:
            feasibility1.append(np.linalg.norm(AX1_b)/N)
            feasibility2.append(np.linalg.norm(np.minimum(state.x_k,0), ord='fro'))
            objective.append(np.sum(C.flatten()*state.x_k.flatten()))
            cur_iter.append(iteration)
            t.append(time.time()-start)
            bar.set_description('{:03d} | {:.4e}| {:.4e}| {:.4e}|'.format(iteration, feasibility1[-1], feasibility2[-1],objective[-1]))
            
    return state.x_k, feasibility1, feasibility2, objective, cur_iter, t


def run_HCGM(alg, problem, maxit=int(1e3), beta0=1):
    # Initialize
    p = problem
    X = np.zeros((N,N))
    state = alg.init_state(p, X, beta0)
    
    feasibility1 = [] # norm(A1(X)-b1)/norm(b1)
    feasibility2 = [] # norm(A2(X)-b2)/norm(b2)
    feasibility3 = [] # dist(X, \mathcal{K})
    objective    = [] # f(x)
    cur_iter    = [] 
    t    = [] 
    
    #u = np.zeros((N,1))
    iter_track = np.unique(np.ceil(np.power(2, np.linspace(0,20,50))))
    
    start = time.time()
    
    bar = trange(1, maxit+1)
    for iteration in bar:
        
        state = alg.state_update(p, state)
                
        if any(iteration == iter_track) or iteration==maxit:
            feasibility1.append(p.penalties[0](state.x_k)/N)
            feasibility2.append(p.penalties[1](state.x_k)/N)
            feasibility3.append(p.penalties[2](state.x_k))
            objective.append(p.f(state.x_k))
            cur_iter.append(iteration)
            t.append(time.time()-start)
            bar.set_description('{:03d} | {:.4e}| {:.4e}| {:.4e}|'.format(iteration, feasibility1[-1], feasibility3[-1],objective[-1]))
            
    return state.x_k, feasibility1, feasibility2, feasibility3, objective, cur_iter, t
